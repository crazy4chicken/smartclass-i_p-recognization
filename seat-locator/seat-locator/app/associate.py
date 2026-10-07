"""face↔person 空间关联：把 face-backend 的逐脸结果挂到 YOLO person 框上。

策略（spec §4.1）：
1) 人脸框中心落在 person 框内的候选集合中取 IoU 最高者（包含优先）；
2) 无人满足包含时，回退到与任意 person 框的最大 IoU（>0）；
3) 一张脸最多关联一个 person，一个 person 最多收一张脸——
   冲突按 (包含, IoU 降序, det_score 降序) 贪心分配：
   几何证据（包含/IoU）分不出高下时，取人脸检测置信度更高者
   （误检脸 det_score 通常显著偏低）；
4) 完全零重叠的 face 不分配，只计入 faces_unmatched。

det_score 语义：SCRFD 人脸检测器对"该区域是一张脸"的置信度
（face-backend 计算），与 similarity（ArcFace 身份相似度）正交。
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
    iou: float = 0.0
    rejected: bool = False


EMPTY_FACE = AssociatedFace(user_id=None, matched=False, similarity=0.0,
                            face_bbox=None, det_score=0.0, iou=0.0,
                            rejected=False)

# 行一致性校验容忍的排差：脸在脚点上方，受透视影响可"漂"进相邻排
ROW_SLACK = 1


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

    # 决断优先级：包含 > IoU > det_score（检测置信度平局决断，
    # 兼作确定性保证——同分时稳定排序保持输入顺序）
    candidates.sort(
        key=lambda c: (not c[3], -c[2], -faces[c[0]].det_score))

    face_taken: set[int] = set()
    person_taken: set[int] = set()
    assignment: dict[int, tuple[int, float]] = {}  # p_idx -> (f_idx, iou)
    for f_idx, p_idx, _ov, _ct in candidates:
        if f_idx in face_taken or p_idx in person_taken:
            continue
        face_taken.add(f_idx)
        person_taken.add(p_idx)
        assignment[p_idx] = (f_idx, _ov)

    out: list[AssociatedFace] = []
    for pi in range(n):
        hit = assignment.get(pi)
        if hit is None:
            out.append(EMPTY_FACE)
        else:
            f_idx, iou = hit
            fr = faces[f_idx]
            out.append(AssociatedFace(
                user_id=fr.user_id,
                matched=fr.matched,
                similarity=fr.similarity,
                face_bbox=fr.bbox,
                det_score=fr.det_score,
                iou=iou,
            ))
    return out


def _row_centers(layout) -> dict[int, float]:
    """布局每排的锚点 y 均值 → 排中心线。"""
    from collections import defaultdict
    ys: dict[int, list[float]] = defaultdict(list)
    for a in layout.anchors:
        ys[a.row].append(a.y)
    return {row: sum(v) / len(v) for row, v in ys.items()}


def associate_with_layout(
    person_bboxes: list[BBox],
    faces: list,
    seats: list,
    layout,
) -> tuple[list[AssociatedFace], int]:
    """关联 + P0 行一致性校验。

    seats 必须与 person_bboxes 等长对齐（seating.assign_seats 的输出）。
    规则：被关联脸的中心 y 按布局排中心线推得"隐含排"，与该人
    脚点判定的排相差 > ROW_SLACK（默认 1）→ 拒配（防透视遮挡串座）：
    user_id/matched/similarity 清空，face_bbox/det_score/iou 保留供调试。
    non_seated（row=None）跳过校验。

    返回 (关联结果, 拒配计数)。
    """
    out = associate_faces(person_bboxes, faces)
    if not out or layout is None:
        return out, 0
    centers = _row_centers(layout)
    if not centers:
        return out, 0

    def _implied_row(cy: float) -> int:
        return min(centers, key=lambda r: abs(cy - centers[r]))

    rejected = 0
    checked: list[AssociatedFace] = []
    for assoc, seat in zip(out, seats):
        if (assoc.rejected or assoc.face_bbox is None
                or seat is None or seat.row is None):
            checked.append(assoc)
            continue
        cy = (assoc.face_bbox[1] + assoc.face_bbox[3]) / 2.0
        if abs(_implied_row(cy) - seat.row) > ROW_SLACK:
            checked.append(AssociatedFace(
                user_id=None, matched=False, similarity=0.0,
                face_bbox=assoc.face_bbox, det_score=assoc.det_score,
                iou=assoc.iou, rejected=True))
            rejected += 1
        else:
            checked.append(assoc)
    return checked, rejected


def warn_anomalies(faces_count: int, persons_count: int,
                   unmatched_count: int,
                   ratio_threshold: float = 0.5) -> list[str]:
    """P1b 关联异常自检（纯函数，供 api 层 log.warning）。

    ① 脸多于人：疑似 YOLO 漏检人体或重复/误检脸；
    ② 有人无脸（P=0 且 F>0）：人脸检出但人体全漏——严重；
    ③ 脸零重叠占比超阈值：疑似坐标系漂移（隐性契约）或重度遮挡。
    """
    msgs: list[str] = []
    if persons_count == 0 and faces_count > 0:
        msgs.append(f"检出 {faces_count} 张人脸但 0 个人体框："
                    "人体检测疑似整体失效")
        return msgs
    if faces_count > persons_count:
        msgs.append(f"人脸数 ({faces_count}) 多于人体数 ({persons_count})："
                    "疑似人体漏检或重复人脸")
    if faces_count > 0 and unmatched_count / faces_count > ratio_threshold:
        msgs.append(f"人脸未关联比例 {unmatched_count}/{faces_count} "
                    f"超过 {ratio_threshold:.0%}：疑似坐标系漂移或重度遮挡")
    return msgs
