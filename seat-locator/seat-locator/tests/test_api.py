"""T8：api.py——服务组装（locate + 布局 CRUD + 运维端点）。

桩注入：FakeDetector / StubFaceClient / MockTransport 图片服务。
真实依赖：pgserver PostgreSQL。
"""
from __future__ import annotations

import os
import socket
import tempfile

import httpx
import pgserver
import pytest
from fastapi.testclient import TestClient

from app.detector import Detection
from app.faceclient import FaceResult, StubFaceClient

PG_DIR = os.path.join(tempfile.gettempdir(), "seatlocator_apitest")

LAYOUT = {
    "classroom_id": "rm-101",
    "image_width": 640, "image_height": 480,
    "rows": 2, "cols": 3,
    "anchors": [
        {"row": 1, "col": 1, "x": 100.0, "y": 100.0},
        {"row": 1, "col": 2, "x": 300.0, "y": 100.0},
        {"row": 1, "col": 3, "x": 500.0, "y": 100.0},
        {"row": 2, "col": 1, "x": 100.0, "y": 300.0},
        {"row": 2, "col": 2, "x": 300.0, "y": 300.0},
        {"row": 2, "col": 3, "x": 500.0, "y": 300.0},
    ],
}


class FakeDetector:
    """确定性检测器：返回固定人员框。"""

    def __init__(self, script: list[Detection]):
        self.script = script

    def detect_persons(self, image_bgr):
        return self.script


def _png_bytes(w=640, h=480) -> bytes:
    import cv2
    import numpy as np
    ok, buf = cv2.imencode(".png", np.zeros((h, w, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _handler(request: httpx.Request) -> httpx.Response:
    """files.example.com 服务 PNG；其余域名连接失败。"""
    if request.url.host == "files.example.com":
        return httpx.Response(200, content=_png_bytes())
    raise httpx.ConnectError("dead host")


async def _public_resolver(host, port=0):
    return [(socket.AF_INET, None, None, "", ("93.184.216.34", port))]


@pytest.fixture(scope="session")
def pg_uri():
    pg = pgserver.get_server(PG_DIR)
    return pg.get_uri()


async def _truncate_tables(dsn: str):
    import asyncpg
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(
            "TRUNCATE smartclass_seatlocator.layouts, "
            "smartclass_seatlocator.locate_log")
    finally:
        await conn.close()


@pytest.fixture
def client(pg_uri):
    from app.api import create_app
    import asyncio
    asyncio.run(_truncate_tables(pg_uri))


    fetch_client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    app = create_app(
        dsn=pg_uri,
        dev=True,  # DEV 旁路：合成 claims
        detector=FakeDetector([
            Detection(bbox=(80, 40, 120, 100), confidence=0.9,
                      foot_point=(100.0, 100.0)),    # → (1,1)
            Detection(bbox=(280, 240, 320, 300), confidence=0.85,
                      foot_point=(300.0, 300.0)),    # → (2,2)
            Detection(bbox=(600, 430, 640, 478), confidence=0.7,
                      foot_point=(620.0, 478.0)),    # 走道 → non_seated
        ]),
        face_client=StubFaceClient([
            FaceResult(bbox=(88, 48, 112, 72), matched=True,
                       user_id="u-1001", similarity=0.71, det_score=0.92),
            FaceResult(bbox=(288, 248, 312, 272), matched=False,
                       user_id=None, similarity=0.2, det_score=0.9),
            FaceResult(bbox=(500, 50, 520, 70), matched=True,
                       user_id="u-9999", similarity=0.66, det_score=0.88),
        ]),
        fetch_client=fetch_client,
        fetch_resolver=_public_resolver,
    )
    with TestClient(app) as c:
        yield c


@pytest.fixture
def empty_detector_client(pg_uri):
    """空检测器：图中无人。"""
    from app.api import create_app
    import asyncio
    asyncio.run(_truncate_tables(pg_uri))
    fetch_client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    app = create_app(
        dsn=pg_uri, dev=True,
        detector=FakeDetector([]),
        face_client=StubFaceClient([]),
        fetch_client=fetch_client,
        fetch_resolver=_public_resolver,
    )
    with TestClient(app) as c:
        yield c


class TestOps:
    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_readyz_pg(self, client):
        r = client.get("/readyz")
        assert r.status_code == 200


class TestLayoutCRUD:
    def test_put_and_get(self, client):
        r = client.put("/api/v1/classrooms/rm-101/layout", json=LAYOUT)
        assert r.status_code == 201, r.text
        r = client.get("/api/v1/classrooms/rm-101/layout")
        assert r.status_code == 200
        assert r.json()["rows"] == 2

    def test_put_id_mismatch_422(self, client):
        r = client.put("/api/v1/classrooms/other-room/layout", json=LAYOUT)
        assert r.status_code == 422
        assert "invalid_layout" in r.text

    def test_put_invalid_layout_422(self, client):
        bad = dict(LAYOUT)
        bad["anchors"] = []  # 空锚点
        r = client.put("/api/v1/classrooms/rm-101/layout", json=bad)
        assert r.status_code == 422
        assert "invalid_layout" in r.text

    def test_get_unknown_404(self, client):
        r = client.get("/api/v1/classrooms/ghost/layout")
        assert r.status_code == 404
        assert "layout_not_found" in r.text

    def test_list_and_delete(self, client):
        client.put("/api/v1/classrooms/rm-101/layout", json=LAYOUT)
        r = client.get("/api/v1/classrooms")
        assert r.status_code == 200
        assert "rm-101" in r.json()["classrooms"]
        r = client.delete("/api/v1/classrooms/rm-101/layout")
        assert r.status_code == 204
        r = client.get("/api/v1/classrooms/rm-101/layout")
        assert r.status_code == 404

    def test_delete_unknown_404(self, client):
        r = client.delete("/api/v1/classrooms/ghost/layout")
        assert r.status_code == 404


class TestLocate:
    def test_locate_flow(self, client):
        client.put("/api/v1/classrooms/rm-101/layout", json=LAYOUT)
        r = client.post("/api/v1/locate", json={
            "image_url": "https://files.example.com/photo/1?expires=x",
            "classroom_id": "rm-101",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["persons_found"] == 3

        by_seat = {(p["row"], p["col"]): p for p in body["persons"]
                   if p["status"] == "seated"}
        p11 = by_seat[(1, 1)]
        assert p11["user_id"] == "u-1001"
        assert p11["matched"] is True
        assert p11["similarity"] == 0.71
        assert p11["face_bbox"] == [88.0, 48.0, 112.0, 72.0]

        p22 = by_seat[(2, 2)]
        assert p22["user_id"] is None
        assert p22["matched"] is False

        non_seated = [p for p in body["persons"]
                      if p["status"] == "non_seated"]
        assert len(non_seated) == 1
        assert non_seated[0]["row"] is None

        # u-9999 的脸没有关联到任何 person 框 → 只计数
        assert body["faces_unmatched"] == 1

    def test_locate_unknown_classroom_404(self, client):
        r = client.post("/api/v1/locate", json={
            "image_url": "https://files.example.com/p/1",
            "classroom_id": "ghost",
        })
        assert r.status_code == 404

    def test_locate_empty_classroom(self, empty_detector_client):
        c = empty_detector_client
        c.put("/api/v1/classrooms/rm-101/layout", json=LAYOUT)
        r = c.post("/api/v1/locate", json={
            "image_url": "https://files.example.com/photo/1",
            "classroom_id": "rm-101",
        })
        assert r.status_code == 200
        assert r.json() == {"persons_found": 0, "persons": [],
                            "faces_unmatched": 0}

    def test_locate_image_fetch_fail_502(self, client):
        client.put("/api/v1/classrooms/rm-101/layout", json=LAYOUT)
        r = client.post("/api/v1/locate", json={
            "image_url": "https://dead.example.com/photo/1",
            "classroom_id": "rm-101",
        })
        assert r.status_code == 502
        assert "image_fetch_failed" in r.text


class TestOpenAPI:
    def test_openapi_permission_annotations(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        locate = spec["paths"]["/api/v1/locate"]["post"]
        assert locate["x-teamusers-permission"] == "seat:check:any"
        put = spec["paths"]["/api/v1/classrooms/{classroom_id}/layout"]["put"]
        assert put["x-teamusers-permission"] == "seat:manage:any"
