"""交互式座位标定工具：抓帧点选锚点 → 布局 JSON → PUT API。

用法：
  python tools/calibrate.py \
      --classroom rm-101 --rows 6 --cols 8 \
      --image classroom_101.png \
      --api http://127.0.0.1:18001 --token <bearer>

流程（spec FR3）：
1. 打开图片窗口，按提示依次点击每个座位中心：
   第 1 排（最近讲台）从左到右，再第 2 排……共 rows*cols 次点击；
2. 组装布局 JSON（image 尺寸取自图片本身）；
3. PUT /api/v1/classrooms/{id}/layout（需 seat:manage:any）。

提示：正式标定建议用"有人上课"的抓帧（定位点特征对齐），
空教室可用桌面中心粗标（Q11 决策）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# 允许 `python tools/calibrate.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import httpx  # noqa: E402


def build_layout(clicks: list[tuple[int, int]], *, rows: int, cols: int,
                 classroom_id: str, image_path: str) -> dict:
    """点击点序列 → 布局 dict；数量不符即退出。"""
    if len(clicks) != rows * cols:
        print(f"[calibrate] 点击数 {len(clicks)} != 排×列 {rows * cols}",
              file=sys.stderr)
        raise SystemExit(2)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[calibrate] 无法读取图片: {image_path}", file=sys.stderr)
        raise SystemExit(2)
    h, w = img.shape[:2]
    anchors = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            x, y = clicks[(r - 1) * cols + (c - 1)]
            anchors.append({"row": r, "col": c, "x": float(x), "y": float(y)})
    return {
        "classroom_id": classroom_id,
        "image_width": int(w), "image_height": int(h),
        "rows": rows, "cols": cols,
        "anchors": anchors,
    }


def collect_clicks(image_path: str, rows: int, cols: int
                   ) -> list[tuple[int, int]]:
    """cv2 窗口点选 rows*cols 个锚点（按排、列顺序）。"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[calibrate] 无法读取图片: {image_path}", file=sys.stderr)
        raise SystemExit(2)
    total = rows * cols
    clicks: list[tuple[int, int]] = []
    state = {"img": img.copy()}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < total:
            clicks.append((x, y))
            r = len(clicks) // cols + 1
            c = len(clicks) % cols if len(clicks) % cols else cols
            cv2.circle(state["img"], (x, y), 6, (0, 0, 255), -1)
            cv2.putText(state["img"], f"{r}-{c}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("calibrate", state["img"])

    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate", on_mouse)
    cv2.imshow("calibrate", img)
    print(f"[calibrate] 请按顺序点击 {total} 个座位中心"
          f"（第 1 排最靠近讲台，每排从左到右）")
    while len(clicks) < total:
        remaining = total - len(clicks)
        cv2.setWindowTitle("calibrate",
                           f"calibrate - 剩余 {remaining} 个点")
        if cv2.waitKey(100) == 27:  # ESC
            print("[calibrate] 已取消", file=sys.stderr)
            raise SystemExit(130)
    cv2.destroyAllWindows()
    return clicks


async def upload_layout(api_base: str, token: str, layout: dict, *,
                        client: httpx.AsyncClient | None = None) -> bool:
    """PUT 布局到服务端；失败即退出。"""
    base = api_base.rstrip("/")
    url = f"{base}/api/v1/classrooms/{layout['classroom_id']}/layout"
    own = client is None
    if own:
        client = httpx.AsyncClient(timeout=15)
    try:
        resp = await client.put(url, json=layout,
                                headers={"Authorization": f"Bearer {token}"})
    finally:
        if own:
            await client.aclose()
    if resp.status_code not in (200, 201):
        print(f"[calibrate] 上传失败 HTTP {resp.status_code}: "
              f"{resp.text[:300]}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[calibrate] 布局已上传: {url}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calibrate", description="教室座位标定工具（API 客户端）")
    parser.add_argument("--classroom", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--image", required=True,
                        help="用于点选的抓帧图片路径")
    parser.add_argument("--api", default="http://127.0.0.1:18001")
    parser.add_argument("--token", required=True, help="Bearer 令牌")
    parser.add_argument("--save-json",
                        help="同时把布局 JSON 保存到本地（调试用）")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.image):
        print(f"[calibrate] 图片不存在: {args.image}", file=sys.stderr)
        raise SystemExit(2)
    if args.rows < 1 or args.cols < 1:
        print("[calibrate] rows/cols 必须 >= 1", file=sys.stderr)
        raise SystemExit(2)

    clicks = collect_clicks(args.image, args.rows, args.cols)
    layout = build_layout(clicks, rows=args.rows, cols=args.cols,
                          classroom_id=args.classroom,
                          image_path=args.image)
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(layout, f, ensure_ascii=False, indent=2)
        print(f"[calibrate] 已保存本地副本: {args.save_json}")
    asyncio.run(upload_layout(args.api, args.token, layout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
