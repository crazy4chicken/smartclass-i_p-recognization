"""人员检测：Detector 协议 + YOLO26 适配器。

ultralytics 惰性导入（AGPL 依赖隔离在本文件），协议层可替换为
其他检测器（RT-DETR 等）而不影响上层。

COCO 类 0 = person；定位点 = 检测框底边中点（spec §2）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class Detection:
    """一个人员检测结果。"""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2（原图坐标）
    confidence: float
    foot_point: tuple[float, float]          # 底边中点


@runtime_checkable
class Detector(Protocol):
    def detect_persons(self, image_bgr: np.ndarray) -> list[Detection]: ...


def _load_yolo(self: "Yolo26Detector"):
    """惰性加载 ultralytics YOLO。分离为模块级函数便于测试替换。"""
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover - 环境相关
        raise ImportError(
            "[detector] ultralytics 未安装；运行 uv pip install -e \".[detector\"]"
        ) from e
    return YOLO(self.weights)


class Yolo26Detector:
    """YOLO26 人员检测适配器（默认 yolo26s，COCO 预训练，开箱即用）。"""

    def __init__(self, weights: str = "yolo26s.pt", conf: float = 0.25,
                 device: str | None = None):
        if not 0.0 < conf < 1.0:
            raise ValueError("conf 必须在 (0, 1) 区间")
        self.weights = weights
        self.conf = conf
        self.device = device
        self._model = None

    def detect_persons(self, image_bgr: np.ndarray) -> list[Detection]:
        if self._model is None:
            self._model = _load_yolo(self)
        kwargs = {"conf": self.conf, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self._model(image_bgr, **kwargs)
        out: list[Detection] = []
        for r in results:
            names = getattr(r, "names", {}) or {}
            boxes = getattr(r, "boxes", None)
            if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
                continue
            for xyxy, cf, cl in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                    boxes.cls.tolist()):
                if int(cl) != PERSON_CLASS_ID:
                    continue
                if cf < self.conf:
                    continue  # 防御性过滤：不依赖模型侧 conf 生效
                if names and names.get(int(cl)) not in (None, "person"):
                    continue  # 类名异常（非 COCO 约定）时保守过滤
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                out.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(cf),
                    foot_point=((x1 + x2) / 2.0, y2),
                ))
        return out
