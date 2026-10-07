"""T9：tools/calibrate.py——交互式标定工具（API 客户端）。

可单测部分：参数解析、点击点 → 布局 JSON 组装、PUT 请求体。
GUI 点选循环（cv2 窗口）在无显示器环境跳过。
"""
from __future__ import annotations

import httpx
import pytest

from tools.calibrate import (
    build_layout,
    collect_clicks,
    main,
    upload_layout,
)


class TestBuildLayout:
    def test_assembles_from_clicks(self, tmp_path):
        import cv2
        import numpy as np
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        p = tmp_path / "frame.png"
        assert cv2.imwrite(str(p), img)

        clicks = [(100, 100), (300, 100), (500, 100),
                  (100, 300), (300, 300), (500, 300)]
        layout = build_layout(clicks, rows=2, cols=3,
                              classroom_id="rm-1",
                              image_path=str(p))
        assert layout["classroom_id"] == "rm-1"
        assert layout["image_width"] == 640
        assert layout["image_height"] == 480
        assert layout["rows"] == 2 and layout["cols"] == 3
        assert layout["anchors"][4] == {"row": 2, "col": 2,
                                        "x": 300.0, "y": 300.0}

    def test_wrong_click_count_raises(self, tmp_path):
        import cv2
        import numpy as np
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        p = tmp_path / "f.png"
        cv2.imwrite(str(p), img)
        with pytest.raises(SystemExit):
            build_layout([(1, 1)], rows=2, cols=3,
                         classroom_id="rm-1", image_path=str(p))


class TestUploadLayout:
    def test_put_with_bearer(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("Authorization")
            captured["body"] = __import__("json").loads(request.content)
            return httpx.Response(201, json={})

        layout = {"classroom_id": "rm-1", "rows": 1, "cols": 1,
                  "image_width": 10, "image_height": 10,
                  "anchors": [{"row": 1, "col": 1, "x": 5.0, "y": 5.0}]}
        import asyncio
        ok = asyncio.run(upload_layout(
            "http://seat:18001", "tok-1", layout,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))
        assert ok
        assert captured["url"] == "http://seat:18001/api/v1/classrooms/rm-1/layout"
        assert captured["auth"] == "Bearer tok-1"
        assert captured["body"] == layout

    def test_server_error_raises(self):
        def handler(request):
            return httpx.Response(422, json={"detail": "bad"})

        import asyncio
        with pytest.raises(SystemExit):
            asyncio.run(upload_layout(
                "http://seat:18001", "t",
                {"classroom_id": "rm-1", "rows": 1, "cols": 1,
                 "image_width": 1, "image_height": 1, "anchors": []},
                client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))


class TestMain:
    def test_usage_without_args(self, capsys):
        with pytest.raises(SystemExit) as ei:
            main([])
        assert ei.value.code == 2

    def test_missing_image_exits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":0")  # 绕过无显示器检查看参数校验
        with pytest.raises(SystemExit):
            main(["--classroom", "rm-1", "--rows", "2", "--cols", "3",
                  "--image", str(tmp_path / "nope.png"),
                  "--api", "http://x", "--token", "t"])
