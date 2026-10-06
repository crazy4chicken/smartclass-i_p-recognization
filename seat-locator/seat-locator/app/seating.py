"""座位网格核心：布局加载与"定位点 → (排, 列)"映射。

纯几何模块，零 CV / 零 IO 依赖，可完全单元测试。

约定（spec §2）：
- 第 1 排最靠近讲台（画面上方），排号向下递增；
- 列号从画面左 → 右连续编号，走道不断列；
- 定位点 = 人员检测框底边中点；
- 定位点未落入任何座位的局部自适应阈值 → non_seated。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


class LayoutError(ValueError):
    """布局数据不合法（结构/取值/一致性校验失败）。"""


@dataclass(frozen=True)
class Anchor:
    """一个座位的标定锚点：座位中心像素坐标。"""
    row: int
    col: int
    x: float
    y: float


@dataclass(frozen=True)
class Layout:
    classroom_id: str
    image_width: int
    image_height: int
    rows: int
    cols: int
    anchors: tuple[Anchor, ...]


@dataclass(frozen=True)
class SeatAssignment:
    """单条定位点的落座判定结果。"""
    row: int | None
    col: int | None
    status: str  # "seated" | "non_seated"


def load_layout(data: dict[str, Any]) -> Layout:
    """从 dict（JSON 反序列化产物）构造 Layout，做全部结构校验。"""
    try:
        classroom_id = data["classroom_id"]
        image_width = data["image_width"]
        image_height = data["image_height"]
        rows = data["rows"]
        cols = data["cols"]
        raw_anchors = data["anchors"]
    except KeyError as e:
        raise LayoutError(f"missing field: {e}") from e

    if not isinstance(classroom_id, str) or not classroom_id.strip():
        raise LayoutError("classroom_id must be a non-empty string")
    if not isinstance(image_width, int) or image_width <= 0:
        raise LayoutError("image_width must be a positive int")
    if not isinstance(image_height, int) or image_height <= 0:
        raise LayoutError("image_height must be a positive int")
    if not isinstance(rows, int) or rows <= 0:
        raise LayoutError("rows must be a positive int")
    if not isinstance(cols, int) or cols <= 0:
        raise LayoutError("cols must be a positive int")
    if not isinstance(raw_anchors, list) or not raw_anchors:
        raise LayoutError("anchors must be a non-empty list")

    if len(raw_anchors) != rows * cols:
        raise LayoutError(
            f"anchors count {len(raw_anchors)} != rows*cols {rows * cols}")

    anchors: list[Anchor] = []
    seen_seats: set[tuple[int, int]] = set()
    for raw in raw_anchors:
        try:
            row, col = int(raw["row"]), int(raw["col"])
            x, y = float(raw["x"]), float(raw["y"])
        except (KeyError, TypeError, ValueError) as e:
            raise LayoutError(f"bad anchor {raw!r}: {e}") from e
        if not (1 <= row <= rows and 1 <= col <= cols):
            raise LayoutError(f"anchor seat ({row}, {col}) out of range "
                              f"1..{rows} x 1..{cols}")
        if (row, col) in seen_seats:
            raise LayoutError(f"duplicate seat ({row}, {col})")
        if not (0 <= x <= image_width and 0 <= y <= image_height):
            raise LayoutError(
                f"anchor ({row}, {col}) pixel ({x}, {y}) outside image "
                f"{image_width}x{image_height}")
        if not (math.isfinite(x) and math.isfinite(y)):
            raise LayoutError(f"anchor ({row}, {col}) non-finite coordinates")
        seen_seats.add((row, col))
        anchors.append(Anchor(row=row, col=col, x=x, y=y))

    return Layout(classroom_id=classroom_id, image_width=image_width,
                  image_height=image_height, rows=rows, cols=cols,
                  anchors=tuple(anchors))


def _nearest_neighbor_distances(layout: Layout) -> dict[tuple[int, int], float]:
    """每个锚点到最近邻锚点的欧氏距离（像素）。

    单锚点布局无邻居，用整幅画面对角线长度代替（阈值退化为全图尺度）。
    """
    anchors = layout.anchors
    if len(anchors) == 1:
        diag = math.hypot(layout.image_width, layout.image_height)
        return {(anchors[0].row, anchors[0].col): diag}
    result: dict[tuple[int, int], float] = {}
    for a in anchors:
        best = min(math.hypot(a.x - b.x, a.y - b.y)
                   for b in anchors if b is not a)
        result[(a.row, a.col)] = best
    return result


def assign_seats(
    foot_points: Iterable[tuple[float, float]],
    layout: Layout,
    scale_factor: float = 0.45,
) -> list[SeatAssignment]:
    """把定位点序列映射为座位判定序列（顺序一一对应）。

    规则：
    1. 每个定位点找最近锚点（欧氏距离）；
    2. 局部自适应阈值 = scale_factor × 该锚点最近邻距离——后排锚点密
       → 阈值小、前排疏 → 阈值大，吸收透视畸变；
    3. 超过阈值（或与画面外坐标）→ non_seated；
    4. 一座一人：同一座位多个定位点时距离最近者胜，其余 non_seated。
    """
    if not (0 < scale_factor < 1):
        raise ValueError("scale_factor must be in (0, 1)")

    nnd = _nearest_neighbor_distances(layout)
    points = [(float(x), float(y)) for x, y in foot_points]

    # 第一遍：每个定位点 -> 候选座位（锚点索引, 距离, 是否过阈值）
    candidates: list[tuple[int, float] | None] = []
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            candidates.append(None)
            continue
        best_idx: int | None = None
        best_dist = math.inf
        for i, a in enumerate(layout.anchors):
            d = math.hypot(x - a.x, y - a.y)
            if d < best_dist:
                best_dist, best_idx = d, i
        if best_idx is None:
            candidates.append(None)
            continue
        anchor = layout.anchors[best_idx]
        threshold = scale_factor * nnd[(anchor.row, anchor.col)]
        candidates.append(best_idx if best_dist <= threshold else None)

    # 第二遍：一座一人冲突消解——同座位取距离最近者
    winners: dict[int, int] = {}  # anchor_idx -> point_idx
    for point_idx, cand in enumerate(candidates):
        if cand is None:
            continue
        if cand not in winners:
            winners[cand] = point_idx
        else:
            cur = winners[cand]
            d_new = math.hypot(points[point_idx][0] - layout.anchors[cand].x,
                               points[point_idx][1] - layout.anchors[cand].y)
            d_cur = math.hypot(points[cur][0] - layout.anchors[cand].x,
                               points[cur][1] - layout.anchors[cand].y)
            if d_new < d_cur:
                winners[cand] = point_idx

    # 第三遍：组装输出
    result: list[SeatAssignment] = []
    for point_idx, cand in enumerate(candidates):
        if cand is not None and winners.get(cand) == point_idx:
            a = layout.anchors[cand]
            result.append(SeatAssignment(row=a.row, col=a.col,
                                         status="seated"))
        else:
            result.append(SeatAssignment(row=None, col=None,
                                         status="non_seated"))
    return result
