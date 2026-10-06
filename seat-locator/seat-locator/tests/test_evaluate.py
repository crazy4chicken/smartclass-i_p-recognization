"""T10：evaluate.py 评测脚本（mapping 模式确定性可测；full 模式重模型走 E2E）。"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


def _tiny_dataset(tmp_path, n=2):
    """用生成器造 2 图小数据集。"""
    cmd = [sys.executable, "tools/gen_benchmark.py",
           "--out", str(tmp_path), "--n", str(n),
           "--rows", "4", "--cols", "5"]
    subprocess.run(cmd, check=True, capture_output=True,
                   cwd=os.path.dirname(os.path.dirname(
                       os.path.abspath(__file__))))
    assert (tmp_path / "gt.json").exists()
    assert (tmp_path / "layout.json").exists()
    return tmp_path


class TestEvaluateMapping:
    def test_perfect_geometry(self, tmp_path):
        ds = _tiny_dataset(tmp_path)
        sys.path.insert(0, str(tmp_path.parent))
        from evaluate import evaluate_mapping, render_report
        r = evaluate_mapping(str(ds))
        assert r["seat_accuracy"] == 1.0
        assert r["non_seated_errors"] == 0
        assert len(r["per_image"]) == 2
        # 报告渲染
        report = render_report([r], str(ds))
        assert "100.0%" in report
        assert "mapping" in report

    def test_cli_runs(self, tmp_path):
        ds = _tiny_dataset(tmp_path)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = tmp_path / "r.md"
        proc = subprocess.run(
            [sys.executable, "evaluate.py", str(ds),
             "--mode", "mapping", "--report", str(out)],
            cwd=root, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        assert "seat_accuracy" in out.read_text()


class TestGTIntegrity:
    def test_gt_footpoints_inside_image(self, tmp_path):
        ds = _tiny_dataset(tmp_path)
        gt = json.loads((ds / "gt.json").read_text())
        layout = json.loads((ds / "layout.json").read_text())
        w, h = layout["image_width"], layout["image_height"]
        for e in gt:
            for p in e["persons"]:
                x, y = p["foot"]
                assert 0 < x < w and 0 < y < h
