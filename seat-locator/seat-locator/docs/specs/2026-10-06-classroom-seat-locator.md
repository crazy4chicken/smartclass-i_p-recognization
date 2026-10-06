# nsc-seatlocator 需求定义（v2 · 集群微服务版）

> 状态：已与需求方达成共识（grilling 三轮共 20 项决策 + 3 项补充约束，2026-10-06）
> v1（独立流分析 PoC）已作废，本文档为唯一规格来源。
> 实施计划：`docs/superpowers/plans/2026-10-06-classroom-seat-locator.md`

## 1. 问题陈述

Nekostick/SmartClass 服务集群中的**无状态分析微服务**：接收一张教室照片的 URL 与 classroom_id，经 YOLO26 人员检测与座位网格映射得出每人的 (排, 列)，经 face-backend 人脸识别得出 user_id，合并后返回 JSON。同时对外提供教室布局管理 API（双向交互），持久化采用 PostgreSQL。

## 2. 术语

| 术语 | 定义 |
|---|---|
| 排/列 | 第 1 排最靠近讲台；列从画面左→右 1..C 连续，走道不断列 |
| 锚点 anchor | 标定时点选的座位中心像素 (x, y) |
| 定位点 foot point | 人员检测框底边中点 |
| non_seated | 定位点未落入任何座位阈值的状态（站立/走动/讲台） |
| 布局 layout | 一间教室的锚点集合 + 行列数，按 classroom_id 存储 |

## 3. 集群上下文（双向交互）

```
smartclass-web(前端)/运营方        dispatchub(调度)
        │ POST /api/v1/locate           │ photos[].download_url
        │ {image_url, classroom_id}     │ (15min TTL，不缓存)
        ▼                               ▼
┌────────────────── nsc-seatlocator :18001 ──────────────────┐
│ 入站 API：locate + 布局 CRUD + healthz/readyz（API Reference）│
│ 出站调用：face-backend /recognize（Bearer 透传）             │──→ face-backend :18000
│           teamusers JWKS 验签 + 权限判定（SDK 本地）          │──→ teamusers  :8080
│ 持久化：PostgreSQL（schema: smartclass_seatlocator）          │
└──────────────────────────────────────────────────────────────┘
        ▲ PUT /api/v1/classrooms/{id}/layout
        │
  calibrate 工具（本地 cv2 点选 → 经 API 写入，不再直写文件）
```

image_url 实际来源：`GET /api/v1/sessions/{id}/artifacts` 返回的 `photos[].download_url`（webcam-server 动态签发，15 分钟 TTL——**本服务绝不缓存或持久化该 URL/图像**）。

## 4. API Reference

FastAPI 自动生成 OpenAPI（`/openapi.json` + `/docs`）；权限以 `x-teamusers-permission` 扩展标注（对齐 dispatchub）。

### 4.1 分析端点

```
POST /api/v1/locate          权限 seat:check:any
Body: {"image_url": "https://...", "classroom_id": "room-301"}
200 →
{
  "classroom_id": "room-301",
  "persons_found": 32,
  "persons": [
    {"user_id": "u-1001", "matched": true, "similarity": 0.71,
     "row": 3, "col": 5, "status": "seated",
     "bbox": [x1,y1,x2,y2], "face_bbox": [x1,y1,x2,y2]},
    {"user_id": null, "matched": false, "row": null, "col": null,
     "status": "non_seated", "bbox": [...], "face_bbox": null}
  ],
  "faces_unmatched": 1
}
```
语义：person 为主键（检到人必有条目）；未匹配/无脸 → `user_id: null`；非就座 → `row/col: null`；未关联到 person 的人脸仅计入 `faces_unmatched`。

### 4.2 布局管理端点

```
GET    /api/v1/classrooms                       seat:read:any    布局摘要列表
GET    /api/v1/classrooms/{classroom_id}/layout seat:read:any    完整布局
PUT    /api/v1/classrooms/{classroom_id}/layout seat:manage:any  创建/覆盖（幂等）
DELETE /api/v1/classrooms/{classroom_id}/layout seat:manage:any  删除
GET    /healthz  /readyz                        公开
```

布局 JSON（PG jsonb 存储）：
```json
{"classroom_id": "room-301", "rows": 7, "cols": 8,
 "anchors": [{"row":1,"col":1,"x":120,"y":450}, ...],
 "created_at": "...", "updated_at": "..."}
```

### 4.3 失败语义（对齐 dispatchub）

| 情形 | 响应 |
|---|---|
| 令牌缺失/无效 | 401 + teamusers decision body + `WWW-Authenticate: Bearer realm="teamusers"` |
| 已认证无权限 | 403 + RFC 9457 problem detail `"permission denied"` |
| teamusers 不可达 | 503 problem detail，**fail closed** |
| classroom_id 无布局 | 404 problem `classroom_not_found` |
| 图片下载失败 | 502 problem `image_fetch_failed` |
| 图片超限/无法解码 | 413 `image_too_large` / 422 `invalid_image` |
| face-backend 失败 | 503 `upstream_unavailable`（透传其错误细节） |
| 布局数据损坏 | 500 `layout_invalid` |

### 4.4 权限目录

三键 any-scope（team/own 阶梯为后续扩展点，布局暂无属主语义）：
`seat:check:any` / `seat:read:any` / `seat:manage:any`
注册方式：`nsc-seatlocator register-permissions` 子命令 → `POST /permissions/ {key, description, registered_by: "nsc-seatlocator"}`（需 TEAMUSERS_ADMIN_TOKEN）。

## 5. 功能需求

- **FR1 输入**：图片 URL（http/https）+ classroom_id；下载防御：大小上限（Content-Length 预拒 + 实读兜底）、像素上限、私有网段 SSRF 防护（可配置放行内网，因 webcam-server 在内网）。
- **FR2 人员检测**：yolo26s（COCO person，AGPL 已接受）；`Detector` 协议接口隔离，可替换。
- **FR3 布局管理**：PG 持久化（schema `smartclass_seatlocator`，`application_name=nsc-seatlocator`）；PUT 幂等覆盖；标定工具经 API 写入。
- **FR4 落座判定**：最近锚点 + 局部自适应阈值（吸收透视）；一座一人冲突消解；不满足 → `non_seated`。
- **FR5 身份关联**：原图 multipart 转发 face-backend `/recognize`（调用方 Bearer 透传）；face bbox 中心包含 + IoU 双策略关联到 person bbox。
- **FR6 输出**：§4.1 契约；调试字段（bbox/similarity）保留。
- **FR7 鉴权**：复刻 face-backend auth.py 模式（teamusers-sdk：Verifier + PermissionsClient + Authenticate）。
- **FR8 运维端点**：healthz（进程活）/readyz（模型加载 + PG 连通）。
- **FR9 请求日志**：locate_log 表记录 classroom_id/时间/人数/状态/错误码/actor_id——**不含任何图像数据**。
- **FR10 评测**：公开教室图片基准，座位级准确率报告（目标 ≥95%），`evaluate.py` 产出。

## 6. 非功能需求

- **NFR1 精度**：座位级准确率 ≥95%（公开图片基准；真实数据到位后复测）。
- **NFR2 隐私**：图像仅内存中转，不落盘、不入库；download_url 不缓存。
- **NFR3 许可**：AGPL-3.0（ultralytics）内部使用已接受；buffalo_l 权重商用需 insightface 授权（face-backend 侧责任，本模块继承该风险）。
- **NFR4 性能**：单请求 P95 < 3s（3060 级 GPU，yolo26s 单帧 + 转发）。
- **NFR5 部署**：Nekostick 托管；`SEAT_LISTEN_ADDR=:18001`；配置全环境变量。
- **NFR6 多教室**：classroom_id 天然隔离；布局 CRUD API 化。
- **NFR7 文档**：OpenAPI 自动文档 + README（对齐 dispatchub 文档结构）。

## 7. 配置参考（SEAT_* 前缀，对齐 DISPATCH_* 风格）

| 变量 | 默认 | 说明 |
|---|---|---|
| SEAT_LISTEN_ADDR | :8081→**18001** | HTTP 监听地址 |
| SEAT_DSN | -（必填） | PG DSN；自动设 application_name/search_path |
| SEAT_TEAMUSERS_URL / ISSUER / AUD | -/teamusers/nekostick | IAM 对接 |
| SEAT_TEAMUSERS_CLIENT_ID/SECRET | - | 权限查询凭证 |
| SEAT_TEAMUSERS_TIMEOUT | 5s | |
| SEAT_FACE_BACKEND_URL/TIMEOUT | -/10s | 人脸服务 |
| SEAT_IMAGE_MAX_BYTES / MAX_PIXELS | 20MB / 40MP | 下载防御 |
| SEAT_YOLO_MODEL | yolo26s.pt | |
| SEAT_SCALE_FACTOR | 0.45 | 座位阈值调参 |
| SEAT_DEV | false | 开发旁路（合成 claims，本地联调用） |
| SEAT_LOG_LEVEL | info | |

## 8. 决策记录

v1 的 Q1-Q14 见 v1 存档；v2 生效决策：

| # | 决策 |
|---|---|
| Q15 | 令牌透传（调用方需 seat:check:any + face:check:any） |
| Q16 | 输出契约见 §4.1（保留调试字段） |
| Q17 | 砍除：RTSP/MJPEG/前端点选/SQLite/时序投票/常驻循环；保留：映射核心/检测协议/标定/评测 |
| Q18 | classroom_id 必带；布局 PG 存储 |
| Q19 | Nekostick 化：nsc-seatlocator、:18001、SEAT_* 环境变量、healthz/readyz |
| Q20 | teamusers 仅验签+权限，不查用户资料 |
| 补1 | 持久化 PostgreSQL（pskl 为误写），schema 隔离 |
| 补2 | FastAPI |
| 补3 | 双向交互：定义完整 API Reference（布局 CRUD + locate + 运维端点 + OpenAPI），标定工具改为 API 客户端 |

## 9. 验收标准

1. OpenAPI 文档（/docs）完整呈现 §4 全部端点与权限标注。
2. `pytest` 全绿（单元 + 集成，含 face-backend 桩与 PG 测试库）。
3. 评测报告：公开图片基准座位级准确率 ≥95%（或附差距归因）。
4. E2E：模拟令牌 + 本地图片服务 → locate 返回正确 (排,列) + user_id（face-backend 桩）。
5. locate_log 与全数据目录无任何图像字节。
6. `register-permissions` 子命令可注册 3 个 seat:* 键。

## 10. 风险

| 风险 | 缓解 |
|---|---|
| 透视 + 后排小目标漏检 | yolo26s STAL；不足时 yolo26-p2 微调（后续项） |
| face↔person 关联歧义（低头/侧脸） | 中心包含优先 + IoU 兜底 + faces_unmatched 计数暴露 |
| 15min TTL URL 过期 | 下载失败即返 502 image_fetch_failed，语义清晰交由调用方重取 |
| SSRF | 私有网段默认拒绝 + 白名单放行 webcam-server/files 域 |
| buffalo_l 权重商用授权 | face-backend 侧责任，README 风险声明 |
