"""进程入口：SEAT_* 环境变量 → create_app → uvicorn。

用法（Nekostick 托管）：
  SEAT_DSN=postgres://... SEAT_TEAMUSERS_URL=http://teamusers:8080 \
  SEAT_TEAMUSERS_SERVICE_TOKEN=... SEAT_FACE_BACKEND_URL=http://face:18000 \
  SEAT_HOST=0.0.0.0 SEAT_PORT=18001 python -m app.main

本地开发（旁路鉴权 + 无 DSN 时用 pgserver）：
  SEAT_DEV=true SEAT_DSN=... python -m app.main
"""
from __future__ import annotations

import logging
import sys


def main() -> int:
    from app.config import from_env
    cfg = from_env()

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    from app.api import create_app
    app = create_app(
        dsn=cfg.dsn,
        schema=cfg.schema,
        dev=cfg.dev,
        scale_factor=cfg.scale_factor,
        image_max_bytes=cfg.image_max_bytes,
        image_max_pixels=cfg.image_max_pixels,
        detector_weights=cfg.detector_weights,
        detector_conf=cfg.detector_conf,
        face_backend_url=cfg.face_backend_url,
        face_backend_timeout=cfg.face_backend_timeout_s,
    )

    import uvicorn
    host, _, port = cfg.listen_addr.partition(":")
    uvicorn.run(app, host=host or "0.0.0.0",
                port=int(port or 18001), log_level=cfg.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
