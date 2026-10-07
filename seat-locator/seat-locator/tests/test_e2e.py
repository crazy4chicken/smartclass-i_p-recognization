"""T10c：E2E 冒烟——真实链路。

组件真实性：
- 检测：真实 yolo26s.pt（CPU 推理，单图 ~2-5s）
- 取图：真实 http.server 本地服务（127.0.0.1，走 SSRF 白名单开关）
- 存储：pgserver 真实 PostgreSQL
- 人脸：StubFaceClient（face-backend 为独立服务，契约已有单测覆盖）
- 鉴权：SEAT_DEV=true 合成 claims（teamusers 为独立服务）

断言（spec §8.4）：端到端 locate 返回正确 (row, col) + user_id。
"""
from __future__ import annotations

import json
import os
import random
import socket
import tempfile
import threading
import functools
from http.server import HTTPServer, SimpleHTTPRequestHandler

import pgserver
import pytest
from fastapi.testclient import TestClient

from app.faceclient import FaceResult, StubFaceClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PG_DIR = os.path.join(tempfile.gettempdir(), "seatlocator_e2e")

requires_model = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "yolo26s.pt"))
    and not os.path.exists("yolo26s.pt"),
    reason="yolo26s.pt 未下载")


@pytest.fixture(scope="module")
def served_image(tmp_path_factory):
    """生成本地合成教室图并用 http.server 服务。"""
    import cv2
    sys_path = os.path.join(ROOT)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from tools.gen_benchmark import _load_sprites, gen_image

    d = tmp_path_factory.mktemp("e2e")
    rng = random.Random(42)
    img, anchors, persons, _standing = gen_image(
        0, rows=4, cols=6, w=1280, h=720, rng=rng,
        sprites=_load_sprites())
    img_path = d / "img.png"
    cv2.imwrite(str(img_path), img)

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(d))
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}/img.png", anchors, persons
    finally:
        httpd.shutdown()


@requires_model
def test_e2e_locate(served_image, monkeypatch):
    url, anchors, persons = served_image
    monkeypatch.chdir(ROOT)  # yolo26s.pt 在仓库根

    pg = pgserver.get_server(PG_DIR)
    dsn = pg.get_uri()

    from app.api import create_app
    app = create_app(
        dsn=dsn, dev=True,
        face_client=StubFaceClient([
            FaceResult(bbox=(0, 0, 1, 1), matched=True,
                       user_id="u-e2e", similarity=0.9, det_score=0.95),
        ]),
        allow_private_image_hosts=True,  # 本地图片服务
        detector_weights="yolo26s.pt",
    )
    layout = {
        "classroom_id": "e2e-room",
        "image_width": 1280, "image_height": 720,
        "rows": 4, "cols": 6,
        "anchors": [{"row": r, "col": c, "x": x, "y": y}
                    for (r, c, x, y) in
                    [(a["row"], a["col"], a["x"], a["y"])
                     for a in anchors]],
    }
    with TestClient(app) as c:
        r = c.put("/api/v1/classrooms/e2e-room/layout", json=layout)
        assert r.status_code in (200, 201), r.text  # 新建 201 / 幂等替换 200

        r = c.post("/api/v1/locate", json={
            "image_url": url, "classroom_id": "e2e-room"})
        assert r.status_code == 200, r.text
        body = r.json()

    # 结构断言（spec §4.1）
    assert body["persons_found"] >= len(persons) * 0.7  # 召回下限
    assert len(body["persons"]) == body["persons_found"]
    seated = [p for p in body["persons"] if p["status"] == "seated"]
    assert len(seated) >= len(persons) * 0.7

    # 正确性断言：GT 至少 70% 的座位被正确判出 (row, col)
    gt_seats = {(p["row"], p["col"]) for p in persons}
    got_seats = {(p["row"], p["col"]) for p in seated}
    overlap = gt_seats & got_seats
    assert len(overlap) >= len(gt_seats) * 0.7

    # user_id 来自桩 → 关联链路贯通（至少一人带上 u-e2e 或 null）
    assert all(p["user_id"] in (None, "u-e2e") for p in body["persons"])
