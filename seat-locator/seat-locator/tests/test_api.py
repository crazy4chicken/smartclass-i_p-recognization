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


# P0 测试用的别名（与 FakeDetector 同实现）
_FakeDet = FakeDetector


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

        # P1a：关联可信度字段
        assert p11["assoc_iou"] is not None and p11["assoc_iou"] > 0
        assert p11["assoc_rejected"] is False
        # p22：带脸但未识别（matched=False）→ assoc_iou 仍有值
        assert p22["assoc_iou"] is not None and p22["assoc_iou"] > 0
        # 无脸的 person（non_seated 走道者）→ assoc_iou null
        assert non_seated[0]["assoc_iou"] is None
        # P0：顶层拒配计数
        assert body["assoc_rejected"] == 0

    def test_locate_row_consistency_rejection(self, client):
        """P0 端到端：后排脸落进前排框 → 拒配，身份清空但调试字段保留。"""
        # 布局：3 排（row1 y=400 前 / row2 y=250 / row3 y=100 后），
        # 排差 2 才会触发拒配（ROW_SLACK=1）
        layout2 = {
            "classroom_id": "rm-102",
            "image_width": 640, "image_height": 480,
            "rows": 3, "cols": 2,
            "anchors": [
                {"row": 1, "col": 1, "x": 150.0, "y": 400.0},
                {"row": 1, "col": 2, "x": 450.0, "y": 400.0},
                {"row": 2, "col": 1, "x": 150.0, "y": 250.0},
                {"row": 2, "col": 2, "x": 450.0, "y": 250.0},
                {"row": 3, "col": 1, "x": 150.0, "y": 100.0},
                {"row": 3, "col": 2, "x": 450.0, "y": 100.0},
            ],
        }
        from app.detector import Detection as _Det
        from app.faceclient import FaceResult as _FR, StubFaceClient as _Stub

        # 前排人：高框（透视拉到 y=100），脚点 (150,400) → row1
        front = _Det(bbox=(50, 100, 250, 400), confidence=0.9,
                     foot_point=(150.0, 400.0))
        from app.api import create_app as _create
        # 复用 client 的 PG（同库不同教室 id 即可）
        r = client.put("/api/v1/classrooms/rm-102/layout", json=layout2)
        assert r.status_code == 201, r.text

        # 换检测器/脸桩：直接操作 app.state（TestClient 持有的 app）
        app = client.app
        app.state.detector = _FakeDet([front])
        app.state.face_client = _Stub([
            _FR(bbox=(135, 115, 165, 145), matched=True,
                user_id="u-back", similarity=0.8, det_score=0.9),
        ])
        r = client.post("/api/v1/locate", json={
            "image_url": "https://files.example.com/photo/1?expires=x",
            "classroom_id": "rm-102",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["persons_found"] == 1
        p = body["persons"][0]
        # 脸中心 y=130 → 隐含 row3，脚点 row1，差 2 > 1 → 拒配
        assert p["row"] == 1 and p["col"] == 1
        assert p["user_id"] is None
        assert p["matched"] is False
        assert p["assoc_rejected"] is True
        assert p["face_bbox"] == [135.0, 115.0, 165.0, 145.0]  # 调试保留
        assert body["assoc_rejected"] == 1
        # 拒配不算 faces_unmatched（它挂上了，是被否决）
        assert body["faces_unmatched"] == 0

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
                            "faces_unmatched": 0, "assoc_rejected": 0}

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
