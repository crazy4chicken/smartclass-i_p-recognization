"""T1：seating.py 纯几何核心——布局加载与落座映射。

覆盖计划规定的 5 类边界输入：走道中点、站立者、一座多人、损坏布局、空教室。
"""
from __future__ import annotations

import pytest

from app.seating import (
    Anchor,
    Layout,
    LayoutError,
    SeatAssignment,
    assign_seats,
    load_layout,
)


def make_layout_data(rows: int = 2, cols: int = 3, spacing_x: float = 100.0,
                     spacing_y: float = 80.0) -> dict:
    """规则网格布局：row 1 在画面上方（近讲台），y 随排号增大。"""
    anchors = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            anchors.append({"row": r, "col": c,
                            "x": c * spacing_x, "y": r * spacing_y})
    return {"classroom_id": "test-room", "image_width": 640, "image_height": 480,
            "rows": rows, "cols": cols, "anchors": anchors}


class TestLoadLayout:
    def test_valid(self):
        layout = load_layout(make_layout_data())
        assert layout.classroom_id == "test-room"
        assert layout.rows == 2 and layout.cols == 3
        assert len(layout.anchors) == 6

    def test_missing_classroom_id(self):
        data = make_layout_data()
        del data["classroom_id"]
        with pytest.raises(LayoutError):
            load_layout(data)

    def test_anchor_count_mismatch(self):
        data = make_layout_data()
        data["anchors"] = data["anchors"][:4]  # 6 -> 4
        with pytest.raises(LayoutError):
            load_layout(data)

    def test_duplicate_seat(self):
        data = make_layout_data()
        data["anchors"][1] = dict(data["anchors"][0])  # 重复 (row, col)
        with pytest.raises(LayoutError):
            load_layout(data)

    def test_out_of_range_row(self):
        data = make_layout_data(rows=1, cols=1)
        data["anchors"].append({"row": 5, "col": 1, "x": 10, "y": 10})
        with pytest.raises(LayoutError):
            load_layout(data)

    def test_out_of_image_coordinates(self):
        data = make_layout_data(rows=1, cols=1)
        data["anchors"][0] = {"row": 1, "col": 1, "x": 9999, "y": 50}
        with pytest.raises(LayoutError):
            load_layout(data)

    def test_non_positive_rows(self):
        data = make_layout_data(rows=1, cols=1)
        data["rows"] = 0
        data["anchors"] = []
        with pytest.raises(LayoutError):
            load_layout(data)


class TestAssignSeats:
    def test_exact_anchor_hit(self):
        layout = load_layout(make_layout_data())
        result = assign_seats([(100.0, 80.0)], layout)
        assert result == [SeatAssignment(row=1, col=1, status="seated")]

    def test_near_anchor_within_threshold(self):
        """定位点落在锚点附近（阈值内）→ 就座。"""
        layout = load_layout(make_layout_data())
        result = assign_seats([(112.0, 86.0)], layout)
        assert result[0].row == 1 and result[0].col == 1
        assert result[0].status == "seated"

    def test_aisle_midpoint_is_non_seated(self):
        """走道中点：距两侧锚点均超过阈值 → non_seated。"""
        layout = load_layout(make_layout_data())  # 锚点 x=100,200,300
        result = assign_seats([(150.0, 80.0)], layout)
        assert result[0].row is None and result[0].col is None
        assert result[0].status == "non_seated"

    def test_standing_person_is_non_seated(self):
        """站立者：y 明显偏离锚点行 → non_seated。"""
        layout = load_layout(make_layout_data())
        result = assign_seats([(100.0, 300.0)], layout)  # 行在 y=80/160
        assert result[0].status == "non_seated"

    def test_conflict_one_seat_one_person(self):
        """一座多人：距离最近者胜，其余 non_seated。"""
        layout = load_layout(make_layout_data())
        result = assign_seats([(101.0, 80.0), (105.0, 81.0)], layout)
        seated = [a for a in result if a.status == "seated"]
        assert len(seated) == 1
        assert seated[0].row == 1 and seated[0].col == 1

    def test_empty_input(self):
        layout = load_layout(make_layout_data())
        assert assign_seats([], layout) == []

    def test_empty_classroom(self):
        """空教室（1x1 布局，无人员）→ 空结果。"""
        layout = load_layout(make_layout_data(rows=1, cols=1))
        assert assign_seats([], layout) == []

    def test_adaptive_threshold_absorbs_perspective(self):
        """透视：后排锚点更密（间距小→阈值小），前排更宽（阈值大）。
        构造前排间距 150、后排间距 50 的布局，验证同一偏移量
        在前排算就座、在后排算 non_seated。
        """
        anchors = [
            {"row": 1, "col": 1, "x": 0, "y": 0},
            {"row": 1, "col": 2, "x": 150, "y": 0},      # 前排间距 150
            {"row": 2, "col": 1, "x": 0, "y": 200},
            {"row": 2, "col": 2, "x": 50, "y": 200},     # 后排间距 50
        ]
        data = {"classroom_id": "p", "image_width": 640, "image_height": 480,
                "rows": 2, "cols": 2, "anchors": anchors}
        layout = load_layout(data)
        # 偏移 25px：前排（阈值≈0.45*150=67.5）在范围内；后排（阈值≈0.45*50=22.5）在范围外
        result = assign_seats([(25.0, 0.0), (25.0, 200.0)], layout)
        assert result[0].status == "seated"
        assert result[1].status == "non_seated"

    def test_foot_points_order_preserved(self):
        """输出顺序与输入定位点顺序一一对应。"""
        layout = load_layout(make_layout_data())
        result = assign_seats([(300.0, 160.0), (100.0, 80.0)], layout)
        assert result[0] == SeatAssignment(row=2, col=3, status="seated")
        assert result[1] == SeatAssignment(row=1, col=1, status="seated")

    def test_scale_factor_bounds(self):
        layout = load_layout(make_layout_data())
        with pytest.raises(ValueError):
            assign_seats([(100.0, 80.0)], layout, scale_factor=0.0)
        with pytest.raises(ValueError):
            assign_seats([(100.0, 80.0)], layout, scale_factor=1.5)
