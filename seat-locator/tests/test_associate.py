"""T5：associate.py——face↔person bbox 空间关联。"""
from __future__ import annotations

from app.associate import associate_faces
from app.faceclient import FaceResult


def face(x1, y1, x2, y2, uid="u-1", sim=0.7, matched=True):
    return FaceResult(bbox=(x1, y1, x2, y2), matched=matched,
                      user_id=uid, similarity=sim)


class TestAssociateFaces:
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
