"""真实并发冒烟验证：yolo26s 实模型 × 两间教室 × 并发 locate。

对比 detect_workers=1（串行基线）与 detect_workers=2（并行）在
4 个并发 locate（2 教室 × 各 2 请求）下的吞吐与正确性。

用法：.venv/bin/python tools/verify_concurrency.py
"""
from __future__ import annotations

import asyncio
import functools
import json
import os
import random
import sys
import tempfile
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import httpx  # noqa: E402
import pgserver  # noqa: E402

from app.api import create_app  # noqa: E402
from app.faceclient import StubFaceClient  # noqa: E402
from tools.gen_benchmark import _load_sprites, gen_image  # noqa: E402

CLASSROOMS = [
    {"id": "verify-a", "rows": 4, "cols": 6, "seed": 101},
    {"id": "verify-b", "rows": 5, "cols": 7, "seed": 202},
]
N_REQ_PER_ROOM = 2


def build_fixtures(workdir: str):
    """生成两间教室的图片 + 布局 + GT，并起本地图片服务。"""
    sprites = _load_sprites()
    rooms, gts = {}, {}
    for spec in CLASSROOMS:
        rng = random.Random(spec["seed"])
        img, anchors, persons, _ = gen_image(
            0, spec["rows"], spec["cols"], 1280, 720, rng, sprites=sprites)
        name = f"{spec['id']}.png"
        cv2.imwrite(os.path.join(workdir, name), img)
        rooms[spec["id"]] = {
            "layout": {
                "classroom_id": spec["id"],
                "image_width": 1280, "image_height": 720,
                "rows": spec["rows"], "cols": spec["cols"],
                "anchors": anchors,
            },
            "file": name,
        }
        gts[spec["id"]] = {(p["row"], p["col"]) for p in persons}

    handler = functools.partial(SimpleHTTPRequestHandler, directory=workdir)
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return rooms, gts, f"http://127.0.0.1:{port}"


async def run_round(pg_uri: str, rooms, gts, base_url, workers: int,
                    workdir: str):
    """一次测量轮：建 app → 预热(在 lifespan) → 4 并发 locate → 计时+校验。"""
    app = create_app(
        dsn=pg_uri, dev=True,
        face_client=StubFaceClient([]),
        allow_private_image_hosts=True,
        detect_workers=workers,
    )  # detector=None → 真实 yolo26s（lifespan 内预热加载）

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        # 布局注册
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://t") as c:
            for room in rooms.values():
                r = await c.put(
                    f"/api/v1/classrooms/{room['layout']['classroom_id']}"
                    "/layout", json=room["layout"])
                assert r.status_code in (200, 201), r.text

        # 预热推理（除模型加载外的首轮开销：线程创建/图优化）
        await asyncio.get_running_loop().run_in_executor(
            app.state.detect_executor,
            app.state.detector.detect_persons,
            cv2.imread(os.path.join(workdir,
                                    rooms[CLASSROOMS[0]["id"]]["file"])))

        # 并发请求
        t0 = time.monotonic()
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://t",
                                     timeout=120) as c:
            tasks = []
            for spec in CLASSROOMS:
                for _ in range(N_REQ_PER_ROOM):
                    tasks.append(c.post("/api/v1/locate", json={
                        "image_url": f"{base_url}/"
                                     f"{rooms[spec['id']]['file']}",
                        "classroom_id": spec["id"]}))
            responses = await asyncio.gather(*tasks)
        wall = time.monotonic() - t0

    # 正确性：每响应座位集合与 GT 交集 ≥ 70%
    for resp, spec in zip(
            responses,
            [s for s in CLASSROOMS for _ in range(N_REQ_PER_ROOM)]):
        assert resp.status_code == 200, resp.text
        body = resp.json()
        got = {(p["row"], p["col"]) for p in body["persons"]
               if p["status"] == "seated"}
        overlap = len(got & gts[spec["id"]]) / len(gts[spec["id"]])
        assert overlap >= 0.7, f"{spec['id']}: overlap {overlap:.0%} < 70%"
    return wall, len(responses)


async def main_async() -> int:
    workdir = tempfile.mkdtemp(prefix="verify_conc_")
    pg_dir = tempfile.mkdtemp(prefix="verify_pg_")
    pg = pgserver.get_server(pg_dir)
    rooms, gts, base_url = build_fixtures(workdir)

    sizes = [f"{c['id']}({c['rows']}x{c['cols']}={c['rows']*c['cols']}座)"
             for c in CLASSROOMS]
    print(f"[verify] 教室: {', '.join(sizes)}")
    print(f"[verify] 每轮 {len(CLASSROOMS) * N_REQ_PER_ROOM} 个并发 locate"
          f"（{len(CLASSROOMS)} 教室 × {N_REQ_PER_ROOM}）\n")

    # 串行基线（workers=1）
    w1, n1 = await run_round(pg.get_uri(), rooms, gts, base_url, workers=1,
                             workdir=workdir)
    # 并行（workers=2）
    w2, n2 = await run_round(pg.get_uri(), rooms, gts, base_url, workers=2,
                             workdir=workdir)

    speedup = w1 / w2
    print(f"detect_workers=1（串行）: {w1:6.2f}s  ({n1/w1:.2f} req/s)")
    print(f"detect_workers=2（并行）: {w2:6.2f}s  ({n2/w2:.2f} req/s)")
    print(f"加速比: {speedup:.2f}x")

    await verify_responsiveness(pg.get_uri(), rooms, gts, base_url,
                                workdir=workdir)
    print("\n[verify] 正确性：全部响应座位 overlap ≥ 70% ✅")
    return 0


async def verify_responsiveness(pg_uri, rooms, gts, base_url, workdir):
    """核心保障：真实 locate 进行中，事件循环（/healthz）不被阻塞。"""
    app = create_app(
        dsn=pg_uri, dev=True,
        face_client=StubFaceClient([]),
        allow_private_image_hosts=True)
    transport = httpx.ASGITransport(app=app)
    first = rooms[CLASSROOMS[0]["id"]]
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://t") as c:
            await c.put(
                f"/api/v1/classrooms/{first['layout']['classroom_id']}"
                f"/layout", json=first["layout"])
            # 预热模型（lifespan 已含 64x64 预热，这里不再重复计时）
            lat: list[float] = []

            async def poll():
                t = time.monotonic()
                await c.get("/healthz")
                lat.append((time.monotonic() - t) * 1000)

            async def locate():
                return (await c.post("/api/v1/locate", json={
                    "image_url": f"{base_url}/{first['file']}",
                    "classroom_id": first["layout"]["classroom_id"],
                })).status_code

            print("\n[verify] 事件循环响应性（真实推理期间 /healthz 探测）：")
            t0 = time.monotonic()
            task = asyncio.create_task(locate())
            while not task.done():
                await poll()
            dt = (time.monotonic() - t0) * 1000
            idle = lat[-5:]
            during = lat[:-5] or lat
            print(f"  locate 总耗时 {dt:.0f}ms，期间探测 {len(during)} 次："
                  f"avg {sum(during)/len(during):.1f}ms / "
                  f"max {max(during):.1f}ms")
            print(f"  空闲基线 avg {sum(idle)/len(idle):.1f}ms / "
                  f"max {max(idle):.1f}ms")
            assert task.result() == 200
            assert max(during) < 100, "推理期间 healthz 被显著阻塞"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
