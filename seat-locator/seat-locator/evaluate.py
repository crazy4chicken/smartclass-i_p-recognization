"""座位级精度评测：mapping（几何）与 full（含检测）两种模式。

用法：
  python evaluate.py benchmarks/synthetic                 # mapping + full
  python evaluate.py benchmarks/synthetic --mode mapping  # 仅几何（无需模型）

指标（spec §8 / NFR1）：
- seat_accuracy：GT 就座人员中被判对 (row, col) 的比例
- non_seated_errors：GT 就座却被判 non_seated（或判错座位且不在任何座）
- standing_as_seated：GT 站立者被分配了座位（应为 0）
- detection_recall（full 模式）：GT 人员被检测到的比例

<95% 时按报告归因（漏检 vs 映射错误），不阻塞交付（Q9b 修订语义）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from app.detector import Yolo26Detector  # noqa: E402
from app.seating import assign_seats, load_layout  # noqa: E402


def evaluate_mapping(dataset_dir: str) -> dict:
    """模式 A：GT 定位点直接进映射（测几何，不测检测）。"""
    with open(os.path.join(dataset_dir, "gt.json")) as f:
        gt = json.load(f)
    with open(os.path.join(dataset_dir, "layout.json")) as f:
        layout = load_layout(json.load(f))

    correct = total = non_seated_errors = 0
    per_image = []
    for entry in gt:
        pts = [tuple(p["foot"]) for p in entry["persons"]]
        results = assign_seats(pts, layout)
        img_ok = img_total = 0
        for p, r in zip(entry["persons"], results):
            total += 1
            if r.status == "seated" and (r.row, r.col) == (p["row"],
                                                            p["col"]):
                correct += 1
                img_ok += 1
            elif r.status == "non_seated":
                non_seated_errors += 1
        per_image.append({"file": entry["file"], "ok": img_ok,
                          "total": len(pts)})
    return {"mode": "mapping", "seat_accuracy": correct / max(1, total),
            "correct": correct, "total": total,
            "non_seated_errors": non_seated_errors,
            "per_image": per_image}


def evaluate_full(dataset_dir: str, weights: str = "yolo26s.pt",
                  conf: float = 0.25) -> dict:
    """模式 B：真实检测 → 映射 → 与 GT 匹配。"""
    with open(os.path.join(dataset_dir, "gt.json")) as f:
        gt = json.load(f)
    with open(os.path.join(dataset_dir, "layout.json")) as f:
        layout = load_layout(json.load(f))

    det = Yolo26Detector(weights=weights, conf=conf)
    correct = total = detected = non_seated_errors = 0
    standing_as_seated = 0
    per_image = []
    for entry in gt:
        img = cv2.imread(os.path.join(dataset_dir, entry["file"]))
        detections = det.detect_persons(img)
        results = assign_seats([d.foot_point for d in detections], layout)

        gt_pts = [(p["row"], p["col"], p["foot"]) for p in entry["persons"]]
        standing_pts = [tuple(s) for s in entry.get("standing", [])]
        matched_gt = set()
        img_ok = img_det = 0
        for r, d in zip(results, detections):
            # 找最近的 GT 脚点（含站立者）
            best_i, best_d2 = None, 1e18
            for i, (gr, gc, gf) in enumerate(gt_pts):
                dd = (d.foot_point[0] - gf[0]) ** 2 + \
                     (d.foot_point[1] - gf[1]) ** 2
                if dd < best_d2:
                    best_i, best_d2 = i, dd
            for j, sp in enumerate(standing_pts):
                dd = (d.foot_point[0] - sp[0]) ** 2 + \
                     (d.foot_point[1] - sp[1]) ** 2
                if dd < best_d2:
                    best_i, best_d2 = None, dd  # 匹配到站立者

            if best_i is None or best_d2 > 60 * 60:  # 60px 匹配半径
                if best_i is None and best_d2 <= 60 * 60 \
                        and r.status == "seated":
                    standing_as_seated += 1
                continue
            if best_i in matched_gt:
                continue  # 一对一
            matched_gt.add(best_i)
            detected += 1
            img_det += 1
            gr, gc, _ = gt_pts[best_i]
            total += 1
            if r.status == "seated" and (r.row, r.col) == (gr, gc):
                correct += 1
                img_ok += 1
            elif r.status == "non_seated":
                non_seated_errors += 1
        per_image.append({"file": entry["file"], "ok": img_ok,
                          "detected": img_det,
                          "gt": len(gt_pts)})
    gt_all = sum(len(e["persons"]) for e in gt)
    return {"mode": "full", "seat_accuracy": correct / max(1, total),
            "detection_recall": detected / max(1, gt_all),
            "correct": correct, "detected": detected, "gt_total": gt_all,
            "non_seated_errors": non_seated_errors,
            "standing_as_seated": standing_as_seated,
            "per_image": per_image}


def render_report(results: list[dict], dataset_dir: str) -> str:
    lines = [f"# 座位级精度报告（{dataset_dir}）", ""]
    for r in results:
        lines.append(f"## 模式：{r['mode']}")
        acc = r["seat_accuracy"]
        lines.append(f"- seat_accuracy: **{acc:.1%}** "
                     f"({r['correct']}/{r.get('total', r.get('detected', 0))})")
        if r["mode"] == "full":
            lines.append(f"- detection_recall: {r['detection_recall']:.1%} "
                         f"({r['detected']}/{r['gt_total']})")
            lines.append(f"- standing_as_seated: {r['standing_as_seated']}")
        lines.append(f"- non_seated_errors: {r['non_seated_errors']}")
        if acc < 0.95:
            lines.append(f"- ⚠️ 未达 95% 目标：归因见 "
                         f"detection_recall（漏检）与 mapping 模式（映射）")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="数据集目录（含 gt.json/layout.json）")
    ap.add_argument("--mode", choices=["mapping", "full", "all"],
                    default="all")
    ap.add_argument("--weights", default="yolo26s.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--report", default=None, help="报告输出路径 (.md)")
    args = ap.parse_args()

    results = []
    if args.mode in ("mapping", "all"):
        results.append(evaluate_mapping(args.dataset))
        print(f"[mapping] accuracy="
              f"{results[-1]['seat_accuracy']:.1%}")
    if args.mode in ("full", "all"):
        results.append(evaluate_full(args.dataset, args.weights, args.conf))
        r = results[-1]
        print(f"[full] seat_accuracy={r['seat_accuracy']:.1%} "
              f"detection_recall={r['detection_recall']:.1%}")

    report = render_report(results, args.dataset)
    out = args.report or os.path.join(args.dataset, "report.md")
    with open(out, "w") as f:
        f.write(report)
    print(f"[report] → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
