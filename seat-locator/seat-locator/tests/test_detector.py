"""T2：detector.py——Detector 协议 + Yolo26Detector 适配器。

单测不依赖 ultralytics/torch：通过注入 FakeUltralytics 验证适配逻辑。
真实模型冒烟由 T10 的 E2E 覆盖（可跳过标记）。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.detector import Detection, Yolo26Detector


class FakeBoxes:
    """模拟 ultralytics Results[0].boxes：xyxy + conf + cls。"""

    def __init__(self, rows):
        self.xyxy = np.array([r[0] for r in rows], dtype=np.float32).reshape(-1, 4)
        self.conf = np.array([r[1] for r in rows], dtype=np.float32)
        self.cls = np.array([r[2] for r in rows], dtype=np.int64)


class FakeResults(list):
    pass


def make_fake_model(rows, *, classes_names=None):
    """rows: list of (xyxy, conf, cls)。返回可调用假模型。"""
    names = classes_names or {0: "person", 15: "rock"}

    def _call(image_bgr, **kwargs):
        res = SimpleNamespace(boxes=FakeBoxes(rows), names=names)
        return FakeResults([res])

    return _call


class TestYolo26Detector:
    def test_person_detection_and_foot_point(self, monkeypatch):
        rows = [([100, 200, 140, 360], 0.9, 0)]  # person
        fake = make_fake_model(rows)
        monkeypatch.setattr("app.detector._load_yolo", lambda self: fake)
        det = Yolo26Detector()
        out = det.detect_persons(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(out) == 1
        d = out[0]
        assert d.bbox == (100.0, 200.0, 140.0, 360.0)
        assert d.foot_point == (120.0, 360.0)
        assert d.confidence == pytest.approx(0.9, abs=1e-6)

    def test_non_person_filtered(self, monkeypatch):
        rows = [
            ([10, 10, 50, 60], 0.99, 15),   # rock → 过滤
            ([100, 200, 140, 360], 0.9, 0),  # person → 保留
        ]
        fake = make_fake_model(rows)
        monkeypatch.setattr("app.detector._load_yolo", lambda self: fake)
        det = Yolo26Detector()
        out = det.detect_persons(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(out) == 1
        assert out[0].foot_point == (120.0, 360.0)

    def test_conf_threshold(self, monkeypatch):
        rows = [
            ([100, 200, 140, 360], 0.30, 0),  # > 0.25 → 保留
            ([200, 200, 240, 360], 0.10, 0),  # < 0.25 → 过滤
        ]
        fake = make_fake_model(rows)
        monkeypatch.setattr("app.detector._load_yolo", lambda self: fake)
        det = Yolo26Detector(conf=0.25)
        out = det.detect_persons(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(out) == 1
        assert out[0].confidence == pytest.approx(0.30)

    def test_empty_and_no_person(self, monkeypatch):
        for rows in ([], [([10, 10, 50, 60], 0.99, 15)]):
            fake = make_fake_model(rows)
            monkeypatch.setattr("app.detector._load_yolo", lambda self, f=fake: f)
            det = Yolo26Detector()
            assert det.detect_persons(np.zeros((480, 640, 3), dtype=np.uint8)) == []

    def test_lazy_load_called_once(self, monkeypatch):
        calls = []

        def _loader(self):
            calls.append(1)
            return make_fake_model([([100, 200, 140, 360], 0.9, 0)])

        monkeypatch.setattr("app.detector._load_yolo", _loader)
        det = Yolo26Detector()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        det.detect_persons(img)
        det.detect_persons(img)
        assert calls == [1]

    def test_missing_import_raises_helpful(self, monkeypatch):
        """ultralytics 未安装时给出可操作的错误信息。"""
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name.startswith("ultralytics"):
                raise ImportError("no ultralytics")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        det = Yolo26Detector()
        with pytest.raises(ImportError, match=r"\[detector\]"):
            det.detect_persons(np.zeros((480, 640, 3), dtype=np.uint8))
