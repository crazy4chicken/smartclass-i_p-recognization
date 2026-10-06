"""服务组装：locate + 布局 CRUD + 运维端点。

路由前缀 /api/v1（对齐 dispatchub）；OpenAPI 每个业务端点标注
x-teamusers-permission；失败统一 RFC 9457 problem details。

依赖注入点（测试/E2E）：detector、face_client、fetch_client。
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import auth
from app.associate import associate_faces
from app.detector import Detection, Detector
from app.faceclient import FaceBackendError, FaceClient, HttpFaceClient
from app.imagefetch import ImageFetchError, fetch_image
from app.problems import ProblemError, install_problem_handler
from app.seating import LayoutError, assign_seats, load_layout
from app.storage import DEFAULT_SCHEMA, Storage

log = logging.getLogger("seatlocator")


# ---------------- 请求/响应模型 ----------------

class LocateRequest(BaseModel):
    image_url: str = Field(..., description="教室照片下载 URL"
                                   "（dispatchub artifacts 签发，15 分钟 TTL）")
    classroom_id: str = Field(..., min_length=1, max_length=64)


class PersonOut(BaseModel):
    user_id: str | None
    matched: bool
    similarity: float
    det_score: float
    row: int | None
    col: int | None
    status: str
    bbox: list[float]
    face_bbox: list[float] | None


class LocateResponse(BaseModel):
    persons_found: int
    persons: list[PersonOut]
    faces_unmatched: int


class ClassroomsOut(BaseModel):
    classrooms: list[str]


# ---------------- 惰性检测器 ----------------

class _LazyDetector:
    """首次 locate 时才加载 ultralytics，避免阻塞进程启动。"""

    def __init__(self, weights: str, conf: float):
        self._weights = weights
        self._conf = conf
        self._inner: Detector | None = None

    def detect_persons(self, image_bgr):
        if self._inner is None:
            from app.detector import Yolo26Detector
            self._inner = Yolo26Detector(weights=self._weights,
                                         conf=self._conf)
        return self._inner.detect_persons(image_bgr)


# ---------------- app 工厂 ----------------

def create_app(
    *,
    dsn: str,
    schema: str = DEFAULT_SCHEMA,
    dev: bool = False,
    detector: Detector | None = None,
    face_client: FaceClient | None = None,
    fetch_client: Any = None,
    fetch_resolver: Any = None,
    scale_factor: float = 0.45,
    image_max_bytes: int = 10 * 1024 * 1024,
    image_max_pixels: int = 30_000_000,
    detector_weights: str = "yolo26s.pt",
    detector_conf: float = 0.25,
    allow_private_image_hosts: bool = False,
    face_backend_url: str = "",
    face_backend_timeout: float = 10.0,
    detect_workers: int = 2,
) -> FastAPI:
    """组装 FastAPI 应用。测试通过 detector/face_client/fetch_client 注入桩。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.storage = Storage(dsn, schema=schema)
        await app.state.storage.connect()
        await app.state.storage.migrate()
        app.state.detector = detector or _LazyDetector(detector_weights,
                                                       detector_conf)
        app.state.face_client = face_client or HttpFaceClient(
            face_backend_url, timeout=face_backend_timeout)
        app.state.fetch_client = fetch_client
        app.state.fetch_resolver = fetch_resolver
        app.state.scale_factor = scale_factor
        app.state.image_max_bytes = image_max_bytes
        app.state.image_max_pixels = image_max_pixels
        app.state.allow_private_image_hosts = allow_private_image_hosts
        # CPU 推理线程池：torch/numpy 在 C 层释放 GIL，
        # 有界池防过载（多教室并行度 = detect_workers）
        app.state.detect_executor = ThreadPoolExecutor(
            max_workers=max(1, detect_workers),
            thread_name_prefix="detect")
        # 预热（face-backend 同款模式）：惰性 detector 提前加载权重，
        # 失败不阻断启动（首个真实请求会给出可操作错误）
        if detector is None:
            try:
                import numpy as np
                app.state.detector.detect_persons(
                    np.zeros((64, 64, 3), dtype=np.uint8))
            except Exception as e:  # pragma: no cover - 环境相关
                log.warning("detector warmup skipped: %s", e)
        auth.set_client(None, dev=dev)
        if not dev:
            # 非 dev 模式：teamusers 客户端在 main 入口注入（见 app/main.py）
            from app.config import from_env
            cfg = from_env()
            if cfg.teamusers_url and cfg.teamusers_service_token:
                auth.set_client(auth.build_client(
                    cfg.teamusers_url, cfg.teamusers_service_token,
                    audience=cfg.teamusers_audience,
                    issuer=cfg.teamusers_issuer), dev=False)
        yield
        app.state.detect_executor.shutdown(wait=False)
        await app.state.storage.close()

    app = FastAPI(
        title="nsc-seatlocator",
        version="0.2.0",
        description="教室座位定位：image_url + classroom_id → (row, col) + user_id",
        lifespan=lifespan,
    )
    install_problem_handler(app)

    # ---------- 运维端点（无鉴权，对齐集群约定） ----------

    @app.get("/healthz", tags=["ops"])
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz():
        try:
            await app.state.storage.pool.fetchval("SELECT 1")
        except Exception as e:  # pragma: no cover - 依赖故障路径
            raise ProblemError(503, "dependency_unavailable",
                               f"PostgreSQL 不可用: {e}")
        return {"status": "ready"}

    # ---------- 布局管理 ----------

    @app.get("/api/v1/classrooms", tags=["layouts"],
             openapi_extra={"x-teamusers-permission": auth.PERM_READ})
    async def list_classrooms(
            claims=Depends(auth.require_read())):
        ids = await app.state.storage.list_classrooms()
        return ClassroomsOut(classrooms=ids)

    @app.get("/api/v1/classrooms/{classroom_id}/layout", tags=["layouts"],
             openapi_extra={"x-teamusers-permission": auth.PERM_READ})
    async def get_layout(classroom_id: str,
                         claims=Depends(auth.require_read())):
        data = await app.state.storage.get_layout(classroom_id)
        if data is None:
            raise ProblemError(404, "layout_not_found",
                               f"教室 {classroom_id} 未标定布局")
        return data

    @app.put("/api/v1/classrooms/{classroom_id}/layout", tags=["layouts"],
             openapi_extra={"x-teamusers-permission": auth.PERM_MANAGE})
    async def put_layout(classroom_id: str, body: dict,
                         claims=Depends(auth.require_manage())):
        body = dict(body)
        body.setdefault("classroom_id", classroom_id)
        if body["classroom_id"] != classroom_id:
            raise ProblemError(
                422, "invalid_layout",
                f"body.classroom_id 与路径 classroom_id 不一致")
        try:
            layout = load_layout(body)
        except LayoutError as e:
            raise ProblemError(422, "invalid_layout", str(e)) from e
        created = await app.state.storage.put_layout(layout_to_dict(layout))
        return JSONResponse(status_code=201 if created else 200,
                            content=layout_to_dict(layout))

    @app.delete("/api/v1/classrooms/{classroom_id}/layout", tags=["layouts"],
                status_code=204,
                openapi_extra={"x-teamusers-permission": auth.PERM_MANAGE})
    async def delete_layout(classroom_id: str,
                            claims=Depends(auth.require_manage())):
        deleted = await app.state.storage.delete_layout(classroom_id)
        if not deleted:
            raise ProblemError(404, "layout_not_found",
                               f"教室 {classroom_id} 未标定布局")
        return None

    # ---------- locate 分析 ----------

    @app.post("/api/v1/locate", response_model=LocateResponse,
              tags=["locate"],
              openapi_extra={"x-teamusers-permission": auth.PERM_CHECK})
    async def locate(body: LocateRequest, request: Request,
                     claims=Depends(auth.require_check())):
        # 1) 布局
        layout_data = await app.state.storage.get_layout(body.classroom_id)
        if layout_data is None:
            raise ProblemError(404, "layout_not_found",
                               f"教室 {body.classroom_id} 未标定布局")
        layout = load_layout(layout_data)

        # 2) 下载图片
        import httpx
        client = app.state.fetch_client or httpx.AsyncClient()
        owns_client = app.state.fetch_client is None
        try:
            img = await fetch_image(
                body.image_url, client=client,
                max_bytes=app.state.image_max_bytes,
                max_pixels=app.state.image_max_pixels,
                resolver=app.state.fetch_resolver,
                allow_private_hosts=app.state.allow_private_image_hosts,
            )
        except ImageFetchError as e:
            raise ProblemError(e.status, e.code, e.detail) from e
        finally:
            if owns_client:
                await client.aclose()

        # 3) 人员检测（线程池：CPU 推理不阻塞事件循环，多教室并行）
        #    + 座位映射
        loop = asyncio.get_running_loop()
        detections: list[Detection] = await loop.run_in_executor(
            app.state.detect_executor,
            app.state.detector.detect_persons, img.bgr)
        seats = assign_seats([d.foot_point for d in detections], layout,
                             scale_factor=app.state.scale_factor)

        # 4) 人脸识别（Bearer 透传）
        bearer = request.headers.get("authorization", "")
        if bearer.lower().startswith("bearer "):
            bearer = bearer[7:]
        else:
            bearer = ""
        try:
            faces = await app.state.face_client.recognize(img.raw,
                                                          bearer=bearer)
        except FaceBackendError as e:
            raise ProblemError(e.status, e.code, e.detail) from e

        # 5) 关联合并
        assoc = associate_faces([d.bbox for d in detections], faces)
        matched_face_boxes = {tuple(a.face_bbox) for a in assoc if a.face_bbox}
        faces_unmatched = sum(
            1 for f in faces if tuple(f.bbox) not in matched_face_boxes)

        persons = []
        for det, seat, face in zip(detections, seats, assoc):
            persons.append(PersonOut(
                user_id=face.user_id,
                matched=face.matched,
                similarity=face.similarity,
                det_score=face.det_score,
                row=seat.row, col=seat.col, status=seat.status,
                bbox=list(det.bbox),
                face_bbox=list(face.face_bbox) if face.face_bbox else None,
            ))

        # 6) 审计（best-effort，不阻断响应）
        try:
            await app.state.storage.insert_locate_log(
                classroom_id=body.classroom_id,
                image_url=body.image_url,
                actor=getattr(claims, "subject", None),
                persons_found=len(persons),
                faces_unmatched=faces_unmatched,
            )
        except Exception:  # pragma: no cover
            log.exception("locate_log 写入失败")

        return LocateResponse(persons_found=len(persons),
                              persons=persons,
                              faces_unmatched=faces_unmatched)

    return app


def layout_to_dict(layout) -> dict:
    """Layout 对象 → 可存储/响应的 dict。"""
    return {
        "classroom_id": layout.classroom_id,
        "image_width": layout.image_width,
        "image_height": layout.image_height,
        "rows": layout.rows,
        "cols": layout.cols,
        "anchors": [{"row": a.row, "col": a.col, "x": a.x, "y": a.y}
                    for a in layout.anchors],
    }
