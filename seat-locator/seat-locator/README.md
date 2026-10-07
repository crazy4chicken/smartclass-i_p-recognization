# nsc-seatlocator

SmartClass / Nekostick 集群中的**教室座位定位微服务**：接收一张教室照片 URL 与
classroom_id，输出照片中每个人的 `(row, col)` 座位坐标与（经 face-backend
识别的）`user_id`。

```
POST /api/v1/locate {"image_url": "...", "classroom_id": "rm-101"}
→ {"persons_found": 32, "persons": [
     {"user_id": "u-1001", "matched": true, "similarity": 0.71,
      "row": 3, "col": 5, "status": "seated",
      "bbox": [...], "face_bbox": [...],
      "assoc_iou": 0.42, "assoc_rejected": false},
     {"user_id": null, "matched": false, "row": null, "col": null,
      "status": "non_seated", "bbox": [...], "face_bbox": null,
      "assoc_iou": null, "assoc_rejected": false}],
   "faces_unmatched": 1, "assoc_rejected": 0}
```

`assoc_iou` = 脸↔人体框关联的 IoU（可信度）；`assoc_rejected` = 被**行一致性校验**
拒配（脸的隐含排与脚点排差 > 1 → 疑似透视遮挡串座，身份清空、调试字段保留），
防"后排学生的脸落进前排大框"导致的前排人被安上后排身份。

## 架构

```
smartclass-web / 运营方            dispatchub（photos[].download_url, 15min TTL）
        │ POST /api/v1/locate              │
        ▼                                  ▼
┌──────────────── nsc-seatlocator :18001 ────────────────┐
│ 鉴权(teamusers-sdk 验签+seat:* 权限) → 下载图片(防SSRF) │
│ → YOLO26s 检人 → 框底中点 → 布局映射(局部自适应阈值)    │
│ → (排,列)/non_seated                                    │
│ → 原图 multipart 转发 face-backend /recognize(Bearer 透传)│
│ → face bbox ↔ person bbox 关联(中心包含+IoU) → 合并 JSON │
└────────────────────────────────────────────────────────┘
     │                                    ▲
     ▼                                    │ JWKS / 权限快照
 PostgreSQL（layouts + locate_log，        │
   schema=smartclass_seatlocator）   teamusers :8080
```

- **无状态**：单次请求完成分析，不做流接入/留存图像（隐私：仅结构化数据落库）
- **布局标定**：`tools/calibrate.py` 在抓帧上点选 R×C 锚点 → `PUT` 布局 API
- **检测器可替换**：`Detector` 协议隔离 ultralytics（AGPL 见下）

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
# 检测依赖（torch CPU 版亦可）：
.venv/bin/pip install "ultralytics>=8.3.200" torch

# 本地开发（合成 claims + pgserver 内嵌 PG）：
SEAT_DEV=true SEAT_DSN=<pg-uri> python -m app.main

# 集群部署（Nekostick 托管）：
SEAT_DSN=postgres://... \
SEAT_TEAMUSERS_URL=http://teamusers:8080 \
SEAT_TEAMUSERS_SERVICE_TOKEN=... \
SEAT_FACE_BACKEND_URL=http://face-backend:18000 \
python -m app.main        # 监听 :18001
```

首次接入集群后注册权限目录（幂等 upsert）：

```bash
SEAT_TEAMUSERS_ADMIN_TOKEN=<admin-token> \
python -m app.cli register-permissions
# 注册 seat:check:any / seat:read:any / seat:manage:any
```

## API Reference

启动后访问 `/docs`（OpenAPI 3.1，每个端点带 `x-teamusers-permission` 标注）。

| 端点 | 权限 | 说明 |
|---|---|---|
| `POST /api/v1/locate` | `seat:check:any` | 图片 URL → 排座 + 身份 JSON |
| `GET /api/v1/classrooms` | `seat:read:any` | 已标定教室列表 |
| `GET /api/v1/classrooms/{id}/layout` | `seat:read:any` | 读取布局 |
| `PUT /api/v1/classrooms/{id}/layout` | `seat:manage:any` | 新建/替换布局（201/200） |
| `DELETE /api/v1/classrooms/{id}/layout` | `seat:manage:any` | 删除布局（204/404） |
| `GET /healthz` / `GET /readyz` | 无 | 存活 / 就绪（readyz 探 PG） |

失败响应统一 RFC 9457 problem details：
`401 unauthorized` / `403 permission_denied` / `404 layout_not_found` /
`413 image_too_large` / `422 invalid_url|invalid_image|invalid_layout` /
`502 image_fetch_failed|face_backend_error` / `503 config_unavailable`。

## 布局标定

```bash
python tools/calibrate.py --classroom rm-101 --rows 6 --cols 8 \
    --image classroom_101.png \
    --api http://127.0.0.1:18001 --token <seat:manage 令牌>
```

窗口内按提示**从第 1 排（近讲台）左→右**依次点击每个座位中心。
建议用有人上课的抓帧（定位点特征对齐）；空教室可用桌面中心粗标。

排/列语义：排号从讲台侧数起；列号画面左→右连续；走道不断列。

## 精度评测

```bash
.venv/bin/python tools/gen_benchmark.py   # 重建合成基准（10 图，含精确 GT）
.venv/bin/python evaluate.py benchmarks/synthetic
```

当前基准结果（`benchmarks/synthetic/report.md`，2026-10-06）：

| 模式 | seat_accuracy | 说明 |
|---|---|---|
| mapping（GT 定位点直接进映射） | **100.0%** (344/344) | 几何核心精度 |
| full（YOLO26s 检测 + 映射全管线） | **97.0%** | 达成 ≥95% 目标（NFR1） |

调参记录：`SEAT_SCALE_FACTOR` 0.35/0.40/0.45 消融无差异（维持默认 0.45）；
`conf` 0.2 vs 0.25 召回差异 <2%（维持默认 0.25）。
**真实教室数据到位后需重标定并复测**（Q9b 交付关卡）。

## 测试

```bash
.venv/bin/python -m pytest tests/ -q     # 91 项（单元+集成+E2E）
```

E2E（`tests/test_e2e.py`）：真实 yolo26s + 本地图片服务 + 真实 PostgreSQL
（pgserver）+ 桩 face-backend，验证完整 locate 链路。

## 配置参考（SEAT_*）

| 变量 | 默认 | 说明 |
|---|---|---|
| `SEAT_LISTEN_ADDR` | `:18001` | 监听地址 |
| `SEAT_DSN` | —（必填*） | PostgreSQL DSN |
| `SEAT_SCHEMA` | `smartclass_seatlocator` | PG schema 隔离 |
| `SEAT_TEAMUSERS_URL` | —（必填*） | teamusers 基地址 |
| `SEAT_TEAMUSERS_AUDIENCE` | `nekostick` | 期望 aud |
| `SEAT_TEAMUSERS_SERVICE_TOKEN` | —（必填*） | 本服务权限查询凭证 |
| `SEAT_FACE_BACKEND_URL` | —（必填*） | face-backend 基地址 |
| `SEAT_IMAGE_MAX_BYTES` | `10485760` | 图片大小上限 |
| `SEAT_IMAGE_MAX_PIXELS` | `30000000` | 像素上限 |
| `SEAT_ALLOW_PRIVATE_IMAGE_HOSTS` | `false` | SSRF 白名单开关（内网文件域） |
| `SEAT_SCALE_FACTOR` | `0.45` | 座位阈值系数 |
| `SEAT_DETECTOR_WEIGHTS` | `yolo26s.pt` | 检测权重 |
| `SEAT_DETECTOR_CONF` | `0.25` | 检测置信度 |
| `SEAT_DETECT_WORKERS` | `2` | 推理线程池大小（多教室并行度） |
| `SEAT_DEV` | `false` | 开发旁路（合成 claims，仅本地） |

\* `SEAT_DEV=true` 时全部可省略。

## 隐私与许可

- **隐私**：不落盘任何图像字节；`locate_log` 仅存 URL/计数/actor（user_id）。
- **AGPL-3.0**：本项目与 ultralytics（YOLO26）均为 AGPL；闭源商用需相应
  企业许可。检测器已协议隔离，可替换为 MIT 许可的 RT-DETR 等实现。
- **face-backend 侧风险**：InsightFace buffalo_l 权重商用需向 insightface.ai
  申请授权（学习/课程项目不受限）——人脸识别能力的商用可行性以该授权为准。

## 多教室扩展

每间教室 = 一行 `layouts` 记录（`classroom_id` 主键）。新增教室：标定 →
PUT 布局 → 即刻可查。无代码改动。
