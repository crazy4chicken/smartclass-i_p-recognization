# Implementation Plan: nsc-seatlocator（v2 · 集群微服务版）

**Feature name**: nsc-seatlocator
**Date**: 2026-10-06
**Spec**: `docs/specs/2026-10-06-classroom-seat-locator.md`（v2）
**REQUIRED SUB-SKILL**: Use superpowers:subagent-driven-development or superpowers:executing-plans

## Goal

无状态 FastAPI 微服务（:18001）：`POST /api/v1/locate {image_url, classroom_id}` → YOLO26s 检人 → PG 布局映射 (排,列) → face-backend /recognize（Bearer 透传）→ face↔person 关联 → 合并 JSON。附布局 CRUD API、teamusers 鉴权（seat:* 权限点）、请求审计表、公开图片精度评测（≥95%）。

## Architecture

```
app/
├── seating.py    # 纯几何：Layout/assign_seats（零 CV 依赖）
├── detector.py   # Detector 协议 + Yolo26Detector（AGPL 隔离点）
├── imagefetch.py # URL 下载 + 大小/像素/SSRF 防御
├── faceclient.py # FaceClient 协议 + HttpFaceClient + StubFaceClient
├── associate.py  # face↔person bbox 关联（中心包含+IoU）
├── storage.py    # asyncpg：layouts 表 + locate_log 表 + 迁移
├── auth.py       # teamusers-sdk 复刻 face-backend 模式（seat:*）
├── problems.py   # RFC 9457 problem detail 统一错误体
├── config.py     # SEAT_* 环境变量（dataclass）
└── main.py       # FastAPI：/api/v1/* + healthz/readyz + register-permissions 子命令
tools/calibrate.py # cv2 点选 → PUT /api/v1/classrooms/{id}/layout
evaluate.py        # 公开图片基准评测
```

依赖单向：main → (auth, storage, imagefetch, detector, seating, faceclient, associate)；seating/associate 纯函数可全单测；faceclient/httpx 仅 main 集成测试用。

**冻结接口**（贯穿任务）：
```python
# seating.py
@dataclass(frozen=True) class Layout: classroom_id: str; rows: int; cols: int; anchors: tuple[Anchor, ...]
@dataclass(frozen=True) class Anchor: row: int; col: int; x: float; y: float
def assign_seats(foot_points: list[tuple[float,float]], layout: Layout, scale_factor: float = 0.45
                ) -> list[tuple[int, int] | None]   # 与输入同序；None = non_seated
# detector.py
class Detection(TypedDict): bbox: tuple[float,float,float,float]; conf: float; label: str
class Detector(Protocol):
    def detect_persons(self, image: np.ndarray) -> list[Detection]: ...
def foot_point(bbox) -> tuple[float, float]  # 底边中点
# faceclient.py
@dataclass(frozen=True) class Face: bbox: tuple[float,float,float,float]; det_score: float; similarity: float | None; user_id: str | None
class FaceClient(Protocol):
    def recognize(self, image_bytes: bytes, bearer: str | None) -> list[Face]: ...
# associate.py
def associate(person_bboxes: list[tuple[...]], faces: list[Face]) -> list[tuple[Face | None, ...]]  # 一人一脸，贪心IoU+中心包含
```

## Tech Stack

Python ≥3.11；`fastapi`、`uvicorn[standard]`、`httpx`、`opencv-python`（解码+标定）、`numpy`、`ultralytics`、`teamusers-sdk`、`asyncpg`、`pytest`（+ `pytest-asyncio`）。PG 测试用独立 schema（`smartclass_seatlocator_test`）。标定工具交互依赖 cv2 GUI。

## Tasks

### Task 1 — 座位映射核心（seating.py）
- [ ] `pytest tests/test_seating.py` 先行覆盖：7×8 单人全落座；走道中点不落任何格；站立者（y 偏移大）→ None；一座两人冲突 → 最近者胜、另一人 None；透视模拟（后排锚距缩小）仍正确；空布局/锚重复坐标 → Layout 校验异常。
- [ ] 实现 `Layout.from_dict`/`to_dict`（与 spec §4.2 布局 JSON 互转）+ `assign_seats`（最近锚点距离 ≤ scale_factor×局部中位锚距，否则 None；一人一格贪心消解）。
- [ ] 提交 "feat: seating core with adaptive-threshold assignment"。

### Task 2 — 检测器封装（detector.py）
- [ ] `tests/test_detector.py`：协议可替换性（注入桩返回固定 Detection 列表）；`foot_point` 数学正确性；图像字节→np.ndarray 解码失败 → `InvalidImageError`。
- [ ] 实现 `Yolo26Detector`（ultralytics YOLO("yolo26s.pt"), classes=[0], conf 阈值可配）；模块导入延迟加载模型（测试环境无权重文件时桩可跑）。
- [ ] 提交 "feat: detector protocol + yolo26 adapter"。

### Task 3 — 图片获取防御（imagefetch.py）
- [ ] `tests/test_imagefetch.py`（httpx MockTransport）：Content-Length 预拒 413；实读超限 413；解码像素超限 422；私有 IP 默认拒绝、白名单放行；非 2xx → ImageFetchError（→502）；重定向后目标仍校验。
- [ ] 实现 `fetch_image(url, *, max_bytes, max_pixels, allow_private_hosts)`。
- [ ] 提交 "feat: hardened image fetcher"。

### Task 4 — face-backend 客户端（faceclient.py）
- [ ] `tests/test_faceclient.py`（MockTransport）：multipart 组装正确（原图字节不变形）；Bearer 透传到 Authorization；非 2xx → UpstreamError 透传 status/detail；faces_found=0 → 空列表；StubFaceClient 按 bbox 注入假脸。
- [ ] 实现 `HttpFaceClient(base_url, timeout)` + `StubFaceClient`。
- [ ] 提交 "feat: face-backend client with token passthrough"。

### Task 5 — face↔person 关联（associate.py）
- [ ] `tests/test_associate.py`：脸中心落在唯一 person 框内 → 关联；跨两框 → IoU 大者；无人脸 person → None；多脸同人 → det_score 高者，余者计 unmatched；零重叠 → faces_unmatched。
- [ ] 实现 `associate()`（中心包含优先，IoU 兜底，贪心 + 不确定度阈值）。
- [ ] 提交 "feat: face-person association"。

### Task 6 — PG 存储（storage.py + config.py）
- [ ] `tests/test_storage.py`（连接测试库）：建表迁移幂等；PUT 布局 upsert；GET/DELETE/列表；locate_log 写入且不含图像字段；DSN 缺失 → 明确报错。
- [ ] 实现 `config.py`（SEAT_* 全量环境变量 + `SEAT_DEV` 旁路开关）与 `storage.py`（asyncpg pool，`SET search_path TO smartclass_seatlocator`，lifespan 建池建表）。
- [ ] 提交 "feat: postgres layout store and request log"。

### Task 7 — 鉴权（auth.py + register-permissions）
- [ ] `tests/test_auth.py`：无令牌 → 401 + WWW-Authenticate；无效令牌 → 401；有令牌无权限 → 403 problem "permission denied"；teamusers 不可达 → 503 fail closed；SEAT_DEV=true → 合成 claims 放行。
- [ ] 实现 `require_check`/`require_read`/`require_manage` FastAPI 依赖（复刻 face-backend auth.py：Verifier + PermissionsClient + Authenticate）+ `problems.py`（RFC 9457）。
- [ ] 实现 CLI 子命令 `register-permissions`（POST /permissions/ 三键，registered_by="nsc-seatlocator"）。
- [ ] 提交 "feat: teamusers authz with seat:* permission catalog"。

### Task 8 — API 层组装（main.py）
- [ ] `tests/test_api.py`（TestClient + 全桩 + 测试 PG）：locate 全链路（桩检测+桩脸 → 响应契约逐字段断言，含 non_seated/user_id null/faces_unmatched）；布局 CRUD 权限矩阵（read/manage 键）；未知 classroom → 404 problem；图片下载失败 → 502；face-backend 500 → 503 upstream_unavailable；healthz/readyz 公开；OpenAPI 含 x-teamusers-permission 扩展。
- [ ] 实现 `/api/v1/locate`（下载→检测→foot_point→assign_seats→转发 recognize→associate→合并）、布局 4 端点、healthz/readyz、CORS、lifespan（模型预热+PG 池）、locate_log 写入。
- [ ] 手动验证：`uvicorn app.main:app` + /docs 交互。
- [ ] 提交 "feat: locate api with layout crud and problem details"。

### Task 9 — 标定工具（tools/calibrate.py）
- [ ] `tests/test_calibrate_cli.py`：CLI 参数解析；布局 JSON → PUT 请求体组装正确（不依赖 cv2 GUI）。
- [ ] 实现：cv2 窗口点选锚点（有人帧优先）→ 行列编号 → `PUT /api/v1/classrooms/{id}/layout`（带 Bearer）。
- [ ] 提交 "feat: interactive calibrator as api client"。

### Task 10 — 评测与收尾
- [ ] `benchmarks/classroom_public/`：收集 ≥10 张公开教室图（手动基准座位标注 GT JSON）。
- [ ] `tests/test_evaluate.py` + `evaluate.py`：座位级准确率、non_seated 错误分类计数；跑基准出 `report.md`，<95% 按 SEAT_SCALE_FACTOR 0.35–0.55 / conf 0.2–0.35 网格调参记录归因。
- [ ] E2E 冒烟（真实 yolo26s + StubFaceClient + 本地图片服务 + SEAT_DEV=true）：端到端断言 persons 结构。
- [ ] `README.md`：架构图、部署（SEAT_* 表）、API 摘要（指向 /docs）、权限注册、评测方法、隐私声明、AGPL + buffalo_l 许可风险。
- [ ] 全量 pytest 绿；提交 "feat: benchmark, e2e and docs"；tag `v0.2.0`。

## Definition of Done（对照 Spec §9）

1. /docs 完整呈现 §4 端点+权限 → 验收 1
2. pytest 全绿 → 验收 2
3. 基准报告 ≥95% 或归因 → 验收 3
4. E2E 桩链路返回正确 (排,列)+user_id → 验收 4
5. locate_log/数据目录无图像字节 → 验收 5
6. register-permissions 三键可注册 → 验收 6
