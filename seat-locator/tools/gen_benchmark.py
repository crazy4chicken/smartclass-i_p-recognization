"""合成教室基准生成器：透视投影座位网格 + 人形剪影 + 精确 GT。

用途（spec §7 / Q9b）：无真实摄像头数据时的可复现评测集。
每张图：
- 透视教室：排距向画面下方递增（近大远小），列距同理；
- 随机占座（occupancy 60~95%），人形剪影（头+躯干）画在座位锚点上，
  脚点 = 座位锚点（GT 定位点，无噪声）或加可选抖动；
- 可选：站立者（non_seated GT）。

输出：
  benchmarks/synthetic/layout.json   共享布局（按第一张图生成）
  benchmarks/synthetic/img_XXX.png   图像
  benchmarks/synthetic/gt.json       [{file, persons: [{row, col, foot:[x,y]}],
                                       standing: [[x,y], ...]}]
"""
from __future__ import annotations

import argparse
import json
import os
import random

import cv2
import numpy as np


def project(row: int, col: int, rows: int, cols: int,
            w: int, h: int) -> tuple[float, float]:
    """排/列 → 像素坐标（透视：后排更密更高）。"""
    # 排：0（最后排，画面上方）→ 1（第一排，画面下方）
    t_row = (rows - row) / rows          # row=1 → 接近 1（底部）
    y = h * (0.26 + 0.66 * t_row)        # 顶部留 26% 黑板区
    # 列：中间列稀、两侧密（水平透视）
    t_col = (col - 1) / max(1, cols - 1)  # 0..1
    center = 0.5 + 0.04 * (t_row - 0.5)  # 消失点略偏
    spread = 0.72 - 0.30 * t_row         # 近排更宽
    x = w * (center + (t_col - 0.5) * spread)
    return x, y


def _shirt_color(rng: random.Random) -> tuple[int, int, int]:
    """随机上衣颜色（饱和但不极端，模拟真实着装）。"""
    import colorsys
    hue = rng.random()
    sat = 0.35 + 0.35 * rng.random()
    val = 0.45 + 0.35 * rng.random()
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(b * 255), int(g * 255), int(r * 255))


SKIN = (150, 175, 205)   # BGR 肤色
HAIR = (40, 45, 60)      # 深发色


def _load_sprites() -> tuple:
    here = os.path.dirname(os.path.abspath(__file__))
    seated = cv2.imread(os.path.join(
        here, "..", "benchmarks", "assets", "sprite_a.png"))
    standing = cv2.imread(os.path.join(
        here, "..", "benchmarks", "assets", "sprite_stand.png"))
    if seated is None or standing is None:
        raise SystemExit("[gen] 缺少精灵图，先运行 z-ai image 生成 "
                         "benchmarks/assets/sprite_{a,stand}.png")
    return seated, standing


def _rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
    for cx, cy in ((radius, radius), (w - radius, radius),
                   (radius, h - radius), (w - radius, h - radius)):
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    cv2.GaussianBlur(mask, (5, 5), 0, dst=mask)
    return mask


def paste_sprite(img, sprite, x: float, y: float,
                 target_h: int, flip: bool) -> None:
    """把精灵贴到 (x, y)=底部中心，高度 target_h，圆角羽化边缘。"""
    if target_h < 8:
        return
    sh, sw = sprite.shape[:2]
    tw = max(4, int(target_h * sw / sh))
    sp = cv2.resize(sprite, (tw, target_h),
                    interpolation=cv2.INTER_AREA)
    if flip:
        sp = cv2.flip(sp, 1)
    x0, y0 = int(x - tw / 2), int(y - target_h)
    x1, y1 = x0 + tw, y0 + target_h
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = sp[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    mask = _rounded_mask(tw, target_h, min(tw, target_h) // 8)
    msub = mask[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    roi = img[cy0:cy1, cx0:cx1]
    img[cy0:cy1, cx0:cx1] = np.where(
        msub[:, :, None] > 0,
        (roi.astype(np.int16) * (255 - msub[:, :, None]) // 255
         + sub.astype(np.int16) * msub[:, :, None] // 255).astype(np.uint8),
        roi)


def draw_person(img, x: float, y: float, scale: float,
                rng: random.Random | None = None,
                sprite: np.ndarray | None = None) -> None:
    """坐姿学生：贴精灵图，底部中心 = (x, y)（座位锚点）。"""
    rng = rng or random.Random()
    if sprite is not None:
        target_h = int(120 * scale)
        paste_sprite(img, sprite, x, y, target_h,
                     flip=rng.random() < 0.5)
        return
    _draw_person_fallback(img, x, y, scale, rng)


def draw_standing(img, x: float, y: float, scale: float,
                  rng: random.Random | None = None,
                  sprite: np.ndarray | None = None) -> None:
    """站立者：贴全身精灵，底部中心 = (x, y)。"""
    rng = rng or random.Random()
    if sprite is not None:
        paste_sprite(img, sprite, x, y, int(200 * scale),
                     flip=rng.random() < 0.5)
        return
    _draw_standing_fallback(img, x, y, scale, rng)


def _draw_person_fallback(img, x, y, scale, rng):
    """无精灵时的矢量兜底（检测召回可能为 0，仅作占位）。"""
    head_r = max(3, int(10 * scale))
    torso_w, torso_h = int(28 * scale), int(40 * scale)
    cx, cy = int(x), int(y)
    shirt = _shirt_color(rng)
    cv2.ellipse(img, (cx, cy - torso_h // 2),
                (torso_w, torso_h), 0, 0, 360, shirt, -1)
    head_cy = cy - torso_h - head_r + 5
    cv2.circle(img, (cx, head_cy), head_r, SKIN, -1)
    cv2.ellipse(img, (cx, head_cy), (head_r + 1, head_r + 1),
                0, 180, 360, HAIR, -1)
    cv2.rectangle(img, (cx - torso_w - 8, cy - 12),
                  (cx + torso_w + 8, cy - 2), (110, 95, 75), -1)


def _draw_standing_fallback(img, x, y, scale, rng):
    head_r = max(3, int(9 * scale))
    torso_w, torso_h = int(24 * scale), int(70 * scale)
    cx, cy = int(x), int(y)
    cv2.rectangle(img, (cx - torso_w // 2 - 2, cy - torso_h - 20),
                  (cx + torso_w // 2 + 2, cy), (55, 50, 45), -1)
    shirt = _shirt_color(rng)
    cv2.ellipse(img, (cx, cy - torso_h // 2 - 10),
                (torso_w, torso_h // 2 + 10), 0, 0, 360, shirt, -1)
    head_cy = cy - torso_h - 30 - head_r
    cv2.circle(img, (cx, head_cy), head_r, SKIN, -1)
    cv2.ellipse(img, (cx, head_cy), (head_r + 1, head_r + 1),
                0, 180, 360, HAIR, -1)


def gen_image(idx: int, rows: int, cols: int, w: int, h: int,
              rng: random.Random, jitter: float = 0.0,
              sprites: tuple | None = None):
    img = np.full((h, w, 3), 235, dtype=np.uint8)
    # 黑板 + 墙面 + 地板
    cv2.rectangle(img, (0, 0), (w, int(h * 0.22)), (28, 42, 72), -1)
    cv2.rectangle(img, (0, int(h * 0.22)), (w, int(h * 0.32)),
                  (200, 195, 188), -1)
    cv2.rectangle(img, (0, int(h * 0.32)), (w, h), (168, 160, 150), -1)
    cv2.putText(img, "BLACKBOARD", (w // 2 - 90, int(h * 0.14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    anchors = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            anchors.append((r, c, *project(r, c, rows, cols, w, h)))

    seated_sprite = sprites[0] if sprites else None
    stand_sprite = sprites[1] if sprites else None

    persons, standing = [], []
    occupancy = rng.uniform(0.6, 0.95)
    for (r, c, x, y) in anchors:
        if rng.random() < occupancy:
            jx = rng.uniform(-jitter, jitter) if jitter else 0.0
            jy = rng.uniform(-jitter, jitter) if jitter else 0.0
            persons.append({"row": r, "col": c,
                            "foot": [x + jx, y + jy]})
    # 按 row 降序绘制（后排先画，前排自然遮挡后排）
    for p in sorted(persons, key=lambda p: -p["row"]):
        fx, fy = p["foot"]
        t_row = (rows - p["row"]) / rows
        draw_person(img, fx, fy, 0.55 + 0.6 * (1 - t_row), rng,
                    sprite=seated_sprite)
    # 站立者 0~2 个（走道）
    for _ in range(rng.randint(0, 2)):
        sx = rng.uniform(w * 0.42, w * 0.58)
        sy = rng.uniform(h * 0.55, h * 0.9)
        draw_standing(img, sx, sy, 1.0, rng, sprite=stand_sprite)
        standing.append([sx, sy])

    layout_anchors = [{"row": r, "col": c, "x": x, "y": y}
                      for (r, c, x, y) in anchors]
    return img, layout_anchors, persons, standing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/synthetic")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="脚点随机抖动像素（模拟坐姿噪声）")
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sprites = _load_sprites()
    rng = random.Random(args.seed)
    gt = []
    layout_anchors = None
    for i in range(args.n):
        img, anchors, persons, standing = gen_image(
            i, args.rows, args.cols, args.width, args.height, rng,
            jitter=args.jitter, sprites=sprites)
        if layout_anchors is None:
            layout_anchors = anchors
        f = os.path.join(args.out, f"img_{i:03d}.png")
        cv2.imwrite(f, img)
        gt.append({"file": f"img_{i:03d}.png", "persons": persons,
                   "standing": standing})
        print(f"[gen] {f}: {len(persons)} seated, "
              f"{len(standing)} standing")

    layout = {
        "classroom_id": "synthetic-001",
        "image_width": args.width, "image_height": args.height,
        "rows": args.rows, "cols": args.cols,
        "anchors": layout_anchors,
    }
    with open(os.path.join(args.out, "layout.json"), "w") as f:
        json.dump(layout, f, indent=1)
    with open(os.path.join(args.out, "gt.json"), "w") as f:
        json.dump(gt, f, indent=1)
    print(f"[gen] layout.json + gt.json → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
