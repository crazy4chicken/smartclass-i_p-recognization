"""face↔person 空间关联：把 face-backend 的逐脸结果挂到 YOLO person 框上。

策略（spec §4.1）：
1) 人脸框中心落在 person 框内的候选集合中取 IoU 最高者（包含优先）；
2) 无人满足包含时，回退到与任意 person 框的最大 IoU（>0）；
3) 一张脸最多关联一个 person，一个 person 最多收一张脸——
   冲突时按 IoU 降序贪心分配。
输出与 person 输入顺序一一对应；未关联的 face 只计入 faces_unmatched。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.faceclient import FaceResult

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class AssociatedFace:
    """一个 person 关联到的人脸信息（可为空）。"""
    user_id: str | None
    matched: bool
    similarity: float
    face_bbox: BBox | None
    det_score: float = 0.0


EMPTY_FACE = AssociatedFace(user_id=None, matched=False, similarity=0.0,
                            face_bbox=None, det_score=0.0)


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _center(f: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = f
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _contains(p: BBox, point: tuple[float, float]) -> bool:
    x, y = point
    return p[0] <= x <= p[2] and p[1] <= y <= p[3]


def associate_faces(person_bboxes: list[BBox],
                    faces: list[FaceResult]) -> list[AssociatedFace]:
    """person_bboxes 顺序 = Detection 顺序；输出一一对应。"""
    n = len(person_bboxes)
    if n == 0:
        return []

    # 候选：(face_idx, person_idx, iou, contained)
    candidates = []
    for fi, fr in enumerate(faces):
        cx, cy = _center(fr.bbox)
        for pi, pb in enumerate(person_bboxes):
            contained = _contains(pb, (cx, cy))
            iou = _iou(fr.bbox, pb)
            if contained or iou > 0:
                candidates.append((fi, pi, iou, contained))

    # 包含优先，其次 IoU；稳定排序保证确定性
    candidates.sort(key=lambda c: (not c[3], -c[2]))

    face_taken: set[int] = set()
    person_taken: set[int] = set()
    assignment: dict[int, FaceResult] = {}
    for f_idx, p_idx, _ov, _ct in candidates:
        if f_idx in face_taken or p_idx in person_taken:
            continue
        face_taken.add(f_idx)
        person_taken.add(p_idx)
        assignment[p_idx] = faces[f_idx]

    out: list[AssociatedFace] = []
    for pi in range(n):
        fr = assignment.get(pi)
        if fr is None:
            out.append(EMPTY_FACE)
        else:
            out.append(AssociatedFace(
                user_id=fr.user_id,
                matched=fr.matched,
                similarity=fr.similarity,
                face_bbox=fr.bbox,
                det_score=fr.det_score,
            ))
    return out
