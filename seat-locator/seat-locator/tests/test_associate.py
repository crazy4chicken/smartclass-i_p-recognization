"""T5：associate.py——face↔person bbox 空间关联。"""
from __future__ import annotations

from app.associate import associate_faces
from app.faceclient import FaceResult


def face(x1, y1, x2, y2, uid="u-1", sim=0.7, matched=True, det=0.9):
    return FaceResult(bbox=(x1, y1, x2, y2), matched=matched,
                      user_id=uid, similarity=sim, det_score=det)


class TestAssociateFaces:
    def test_det_score_breaks_iou_tie(self):
        """多脸同人且 IoU 完全相等 → det_score 高者胜（计划 Task5 承诺）。"""
        persons = [(100, 0, 200, 300)]
        # 两张脸与 person 框 IoU 相等（对称位置同尺寸），det_score 不同
        low = face(120, 30, 160, 70, uid="u-low", det=0.35)
        high = face(140, 30, 180, 70, uid="u-high", det=0.93)
        for order in ([low, high], [high, low]):  # 顺序无关
            out = associate_faces(persons, order)
            assert out[0].user_id == "u-high", \
                "IoU 平局应由 det_score 决断"

    def test_center_containment(self):
        """脸中心落在 person 框内 → 关联。"""
        persons = [(100, 0, 200, 300), (300, 0, 400, 300)]
        faces = [face(130, 30, 170, 90)]  # 中心(150,60) 在第一框内
        out = associate_faces(persons, faces)
        assert out[0].user_id == "u-1"
        assert out[1].user_id is None

    def test_one_face_two_persons_containment_pick_best_iou(self):
        """脸中心同时落在两个重叠 person 框内 → IoU 更高者胜。"""
        persons = [(100, 0, 180, 300), (150, 0, 250, 300)]
        faces = [face(150, 30, 200, 90)]  # 中心(175,60) 同时在两框内
        out = associate_faces(persons, faces)
        # 与第二框 IoU(0.1) 高于第一框(~0.071)
        assert out[1].user_id == "u-1"
        assert out[0].user_id is None

    def test_iou_fallback_when_no_containment(self):
        """中心都不包含（侧脸贴边）→ IoU>0 兜底。"""
        persons = [(100, 0, 200, 300)]
        faces = [face(180, 20, 260, 100)]  # 中心(220,60) 在框外，但有重叠
        out = associate_faces(persons, faces)
        assert out[0].user_id == "u-1"

    def test_no_overlap_no_association(self):
        persons = [(100, 0, 200, 300)]
        faces = [face(400, 400, 500, 500)]
        out = associate_faces(persons, faces)
        assert out[0].user_id is None
        assert out[0].face_bbox is None

    def test_two_faces_one_person_best_wins(self):
        """两条脸都想关联同一 person → IoU 高者胜，另一条悬空。"""
        persons = [(100, 0, 200, 300)]
        f1 = face(110, 10, 190, 110, uid="u-a")   # IoU 大
        f2 = face(180, 20, 260, 100, uid="u-b")   # IoU 小
        out = associate_faces(persons, [f1, f2])
        assert out[0].user_id == "u-a"

    def test_unmatched_face_not_claimed_twice(self):
        """person 数多于脸数：每张脸只进一个 person。"""
        persons = [(100, 0, 200, 300), (250, 0, 350, 300)]
        faces = [face(130, 30, 170, 90)]
        out = associate_faces(persons, faces)
        assert out[0].user_id == "u-1"
        assert out[1].user_id is None

    def test_unmatched_face_keeps_bbox(self):
        """关联失败的 person 其 face 字段为 None；matched=False。"""
        persons = [(100, 0, 200, 300)]
        out = associate_faces(persons, [])
        assert out[0].user_id is None
        assert out[0].matched is False
        assert out[0].similarity == 0.0

    def test_matched_false_face_still_associates(self):
        """识别未命中（user_id=None, matched=False）的脸仍参与关联
        （face_bbox 要出现在输出里，spec §4.1 调试字段）。"""
        persons = [(100, 0, 200, 300)]
        faces = [face(130, 30, 170, 90, uid=None, sim=0.2, matched=False)]
        out = associate_faces(persons, faces)
        assert out[0].user_id is None
        assert out[0].matched is False
        assert out[0].face_bbox == (130.0, 30.0, 170.0, 90.0)
        assert out[0].similarity == pytest_approx(0.2)


def pytest_approx(v):
    import pytest
    return pytest.approx(v)


# ---------------- P0/P1：行一致性校验 + assoc_iou + 异常告警 ----------------

from app.associate import associate_with_layout, warn_anomalies  # noqa: E402
from app.seating import assign_seats, load_layout  # noqa: E402


def make_layout_rows(spacing_y=150.0):
    """3 排 3 列：row1 y=400（前/画面下），row2 y=250，row3 y=100（后/画面上）。"""
    anchors = []
    for r, y in ((1, 400.0), (2, 250.0), (3, 100.0)):
        for c, x in ((1, 100.0), (2, 300.0), (3, 500.0)):
            anchors.append({"row": r, "col": c, "x": x, "y": y})
    return load_layout({
        "classroom_id": "p0p1", "image_width": 640, "image_height": 480,
        "rows": 3, "cols": 3, "anchors": anchors})


class TestAssocIou:
    def test_iou_exposed_on_association(self):
        persons = [(100, 0, 200, 300)]
        f = face(110, 10, 190, 110)  # 交叠 80x100，并集 100x300
        out = associate_faces(persons, [f])
        assert out[0].iou == pytest_approx(80 * 100 / (100 * 300))

    def test_no_face_iou_zero(self):
        out = associate_faces([(100, 0, 200, 300)], [])
        assert out[0].iou == 0.0


class TestRowConsistency:
    def test_backrow_face_in_front_box_rejected(self):
        """缺陷1核心场景：后排学生的脸落进前排大框（透视+遮挡）→ 拒配。"""
        layout = make_layout_rows()
        # P1=前排(row1, 脚点 y=400)，框被透视拉高到覆盖后排头部区
        p1_bbox = (50, 100, 250, 400)
        seats = assign_seats([(150.0, 400.0)], layout)
        assert seats[0].row == 1
        # 后排学生(row3 区域, 脸中心 y=130)的脸 contained 于 P1 框
        f_back = face(135, 115, 165, 145, uid="u-back")
        assoc, rejected = associate_with_layout(
            [p1_bbox], [f_back], seats, layout)
        assert rejected == 1
        assert assoc[0].user_id is None
        assert assoc[0].matched is False
        assert assoc[0].face_bbox == (135.0, 115.0, 165.0, 145.0)  # 调试保留
        assert assoc[0].rejected is True
        assert assoc[0].iou > 0  # 几何证据保留

    def test_own_face_slack_one_row(self):
        """本人脸在脚点上方一排内 → 容忍不拒（正常头身几何）。"""
        layout = make_layout_rows()
        p1_bbox = (50, 100, 250, 400)   # row1, 脚 y=400
        seats = assign_seats([(150.0, 400.0)], layout)
        # 脸中心 y=320 → 隐含 row2（|320-250|=70 < |320-400|=80），差 1 → 容忍
        f_own = face(135, 305, 165, 335, uid="u-front")
        assoc, rejected = associate_with_layout(
            [p1_bbox], [f_own], seats, layout)
        assert rejected == 0
        assert assoc[0].user_id == "u-front"
        assert assoc[0].rejected is False

    def test_non_seated_person_not_rejected(self):
        """non_seated（row=None）无从交叉验证 → 跳过校验。"""
        layout = make_layout_rows()
        p_bbox = (50, 100, 250, 470)      # 走道上（脚点 y=470 在所有排之外）
        seats = assign_seats([(150.0, 470.0)], layout)
        assert seats[0].row is None
        f = face(135, 115, 165, 145, uid="u-x")  # 隐含 row3
        assoc, rejected = associate_with_layout(
            [p_bbox], [f], seats, layout)
        assert rejected == 0
        assert assoc[0].user_id == "u-x"

    def test_honest_face_kept(self):
        """正常场景：前排人自己的脸（隐含排=脚点排）→ 保留。"""
        layout = make_layout_rows()
        p1_bbox = (50, 300, 250, 400)
        seats = assign_seats([(150.0, 400.0)], layout)
        f_own = face(135, 310, 165, 340, uid="u-front")  # cy=325 → row1
        assoc, rejected = associate_with_layout(
            [p1_bbox], [f_own], seats, layout)
        assert rejected == 0
        assert assoc[0].user_id == "u-front"


class TestWarnAnomalies:
    def test_faces_more_than_persons(self):
        msgs = warn_anomalies(faces_count=5, persons_count=3,
                              unmatched_count=2)
        assert any("多于" in m for m in msgs)

    def test_high_unmatched_ratio(self):
        msgs = warn_anomalies(faces_count=10, persons_count=10,
                              unmatched_count=6)
        assert any("坐标" in m or "遮挡" in m for m in msgs)

    def test_faces_without_persons(self):
        msgs = warn_anomalies(faces_count=4, persons_count=0,
                              unmatched_count=4)
        assert len(msgs) >= 1

    def test_normal_no_warning(self):
        assert warn_anomalies(3, 3, 0) == []
