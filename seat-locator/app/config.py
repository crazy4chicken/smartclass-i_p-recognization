"""配置：SEAT_* 环境变量（对齐 dispatchub 的 DISPATCH_* 风格）。

优先级：环境变量 > 默认值。SEAT_DEV=true 时放行 teamusers/DSN 必填项
（开发旁路，与 dispatchub 的 DISPATCH_DEV 同语义，仅限本地开发）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(v: str | None, default: int) -> int:
    try:
        return int(v) if v not in (None, "") else default
    except ValueError:
        raise SystemExit(f"[config] 非法整数环境变量值: {v!r}")


@dataclass(frozen=True)
class Config:
    listen_addr: str
    dsn: str
    schema: str
    teamusers_url: str
    teamusers_issuer: str
    teamusers_audience: str
    teamusers_service_token: str
    teamusers_timeout_s: float
    face_backend_url: str
    face_backend_timeout_s: float
    image_max_bytes: int
    image_max_pixels: int
    allow_private_image_hosts: bool
    scale_factor: float
    detector_weights: str
    detector_conf: float
    dev: bool
    log_level: str


def from_env(env: dict[str, str] | None = None) -> Config:
    e = env if env is not None else dict(os.environ)
    dev = _bool(e.get("SEAT_DEV"))
    missing = [k for k in ("SEAT_DSN", "SEAT_TEAMUSERS_URL",
                           "SEAT_TEAMUSERS_SERVICE_TOKEN",
                           "SEAT_FACE_BACKEND_URL")
               if not e.get(k)]
    if missing and not dev:
        raise SystemExit(
            f"[config] 缺少必填环境变量: {', '.join(missing)}"
            "（本地开发可设 SEAT_DEV=true 旁路）")
    return Config(
        listen_addr=e.get("SEAT_LISTEN_ADDR", ":18001"),
        dsn=e.get("SEAT_DSN", ""),
        schema=e.get("SEAT_SCHEMA", "smartclass_seatlocator"),
        teamusers_url=e.get("SEAT_TEAMUSERS_URL", ""),
        teamusers_issuer=e.get("SEAT_TEAMUSERS_ISSUER", "teamusers"),
        teamusers_audience=e.get("SEAT_TEAMUSERS_AUDIENCE", "nekostick"),
        teamusers_service_token=e.get("SEAT_TEAMUSERS_SERVICE_TOKEN", ""),
        teamusers_timeout_s=float(e.get("SEAT_TEAMUSERS_TIMEOUT", "5") or 5),
        face_backend_url=e.get("SEAT_FACE_BACKEND_URL", ""),
        face_backend_timeout_s=float(
            e.get("SEAT_FACE_BACKEND_TIMEOUT", "10") or 10),
        image_max_bytes=_int(e.get("SEAT_IMAGE_MAX_BYTES"), 10 * 1024 * 1024),
        image_max_pixels=_int(e.get("SEAT_IMAGE_MAX_PIXELS"), 30_000_000),
        allow_private_image_hosts=_bool(
            e.get("SEAT_ALLOW_PRIVATE_IMAGE_HOSTS")),
        scale_factor=float(e.get("SEAT_SCALE_FACTOR", "0.45") or 0.45),
        detector_weights=e.get("SEAT_DETECTOR_WEIGHTS", "yolo26s.pt"),
        detector_conf=float(e.get("SEAT_DETECTOR_CONF", "0.25") or 0.25),
        dev=dev,
        log_level=e.get("SEAT_LOG_LEVEL", "info"),
    )
