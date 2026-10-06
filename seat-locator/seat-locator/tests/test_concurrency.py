"""并发 locate：CPU 推理不得阻塞事件循环。

多教室并行检测场景：两个请求各带 0.4s 的"推理延迟"，
若推理阻塞事件循环则总耗时 >= 0.8s（串行）；
线程池化后应接近并行（< 0.7s）。
"""
from __future__ import annotations

import asyncio
import os
import socket
import tempfile
import threading
import time

import httpx
import pgserver
import pytest

from app.api import create_app
from app.detector import Detection
from app.faceclient import StubFaceClient

PG_DIR = os.path.join(tempfile.gettempdir(), "seatlocator_conc")

LAYOUT = {
    "classroom_id": "conc-room",
    "image_width": 640, "image_height": 480,
    "rows": 1, "cols": 1,
    "anchors": [{"row": 1, "col": 1, "x": 100.0, "y": 100.0}],
}


class SlowDetector:
    """模拟 CPU 推理耗时（sleep 释放 GIL，与 torch C 层行为一致）。"""

    def __init__(self, delay: float):
        self.delay = delay
        self.call_count = 0
        self._lock = threading.Lock()

    def detect_persons(self, image_bgr):
        with self._lock:
            self.call_count += 1
        time.sleep(self.delay)
        return [Detection(bbox=(80, 40, 120, 100), confidence=0.9,
                          foot_point=(100.0, 100.0))]


async def _resolve_public(host, port=0):
    return [(socket.AF_INET, None, None, "", ("93.184.216.34", port))]


def _png_bytes() -> bytes:
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".png",
                           np.zeros((240, 320, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def pg_uri():
    pg = pgserver.get_server(PG_DIR)
    return pg.get_uri()


async def test_parallel_locate_not_serialized(pg_uri, monkeypatch):
    import cv2  # noqa: F401
    png = _png_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=png,
                              headers={"Content-Type": "image/png"})

    slow = SlowDetector(0.4)
    app = create_app(
        dsn=pg_uri, dev=True,
        detector=slow,
        face_client=StubFaceClient([]),
        fetch_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler)),
        fetch_resolver=_resolve_public,
    )

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        import asyncpg as _apg
        conn = await _apg.connect(pg_uri)
        try:
            await conn.execute(
                "TRUNCATE smartclass_seatlocator.layouts, "
                "smartclass_seatlocator.locate_log")
        finally:
            await conn.close()

        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://t") as c:
            await c.put("/api/v1/classrooms/conc-room/layout", json=LAYOUT)

            t0 = time.monotonic()
            r1, r2 = await asyncio.gather(
                c.post("/api/v1/locate", json={
                    "image_url": "https://files.example.com/a.png",
                    "classroom_id": "conc-room"}),
                c.post("/api/v1/locate", json={
                    "image_url": "https://files.example.com/b.png",
                    "classroom_id": "conc-room"}),
            )
            dt = time.monotonic() - t0

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    assert slow.call_count == 2
    # 两个 0.4s 推理并行完成；串行则 >= 0.8s。留余量防 CI 抖动。
    assert dt < 0.7, f"推理被串行化：两请求总耗时 {dt:.2f}s >= 0.7s"


async def test_executor_bounded(pg_uri):
    """detect 线程池有界：可配置且正确初始化。"""
    png = _png_bytes()

    def handler(request):
        return httpx.Response(200, content=png)

    app = create_app(
        dsn=pg_uri, dev=True,
        detector=SlowDetector(0.01),
        face_client=StubFaceClient([]),
        fetch_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler)),
        fetch_resolver=_resolve_public,
        detect_workers=3,
    )
    async with app.router.lifespan_context(app):
        assert app.state.detect_executor._max_workers == 3
