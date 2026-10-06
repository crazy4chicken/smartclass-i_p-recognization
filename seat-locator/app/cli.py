"""CLI：register-permissions 子命令。

契约（对齐 smartclass-dispatchub internal/iamauth/register.go）：
POST {TEAMUSERS_URL}/permissions/  (admin bearer)
{"key": "...", "description": "...", "registered_by": "nsc-seatlocator"}
→ 201 新建 / 200 未变化；逐键 upsert，部分失败汇总报错。

用法：
  SEAT_TEAMUSERS_URL=... python -m app.cli register-permissions -token <admin>
"""
from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from app import auth


async def register_permissions(url: str, token: str,
                               keys: dict[str, str] | None = None,
                               timeout: float = 60.0) -> tuple[int, int]:
    """注册权限目录，返回 (成功数, 失败数)。"""
    catalog = keys or auth.PERMISSION_CATALOG
    base = url.rstrip("/")
    failures = 0
    async with httpx.AsyncClient(timeout=timeout) as client:
        for key, description in catalog.items():
            try:
                resp = await client.post(
                    f"{base}/permissions/",
                    json={"key": key, "description": description,
                          "registered_by": auth.REGISTERED_BY},
                    headers={"Authorization": f"Bearer {token}",
                             "Accept": "application/json"})
                if not (200 <= resp.status_code < 300):
                    failures += 1
                    print(f"  ✗ {key}: HTTP {resp.status_code}",
                          file=sys.stderr)
            except httpx.HTTPError as e:
                failures += 1
                print(f"  ✗ {key}: {e}", file=sys.stderr)
    return len(catalog) - failures, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nsc-seatlocator")
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser(
        "register-permissions",
        help="向 teamusers 注册 seat:* 权限目录（幂等 upsert）")
    reg.add_argument("-url", "--teamusers-url",
                     default=None,
                     help="teamusers base URL（默认取 SEAT_TEAMUSERS_URL）")
    reg.add_argument("-token", "--admin-token", default=None,
                     help="teamusers admin bearer token"
                          "（默认取 SEAT_TEAMUSERS_ADMIN_TOKEN）")
    args = parser.parse_args(argv)

    if args.command == "register-permissions":
        import os
        url = args.teamusers_url or os.environ.get("SEAT_TEAMUSERS_URL", "")
        token = args.admin_token or os.environ.get(
            "SEAT_TEAMUSERS_ADMIN_TOKEN", "")
        if not url:
            print("需要 -url 或 SEAT_TEAMUSERS_URL", file=sys.stderr)
            return 2
        if not token:
            print("需要 -token 或 SEAT_TEAMUSERS_ADMIN_TOKEN",
                  file=sys.stderr)
            return 2
        ok, fail = asyncio.run(register_permissions(url, token))
        print(f"权限目录注册完成: 成功 {ok}, 失败 {fail}")
        return 0 if fail == 0 else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
