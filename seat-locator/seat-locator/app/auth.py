"""teamusers 鉴权：校验访问者 EdDSA 令牌并按权限点放行。

模式复刻 face-backend/app/auth.py（调研 2026-10-06）：
- 验签走 teamusers JWKS（EdDSA），本服务不自签令牌；
- 权限判定走 teamusers_sdk.PermissionsClient（服务令牌）；
- 未认证 → 401，已认证无权限 → 403（fail-closed）；
- SEAT_DEV=true 时返回合成 claims（仅本地开发）。

权限点（spec §5）：
- seat:check:any   locate 分析
- seat:read:any    布局读取
- seat:manage:any  布局写入/删除
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import Request

from app.problems import ProblemError

PERM_CHECK = "seat:check:any"
PERM_READ = "seat:read:any"
PERM_MANAGE = "seat:manage:any"

PERMISSION_CATALOG: dict[str, str] = {
    PERM_CHECK: "Run classroom seat location analysis on a photo URL.",
    PERM_READ: "Read classroom seat layouts.",
    PERM_MANAGE: "Create, replace and delete classroom seat layouts.",
}
REGISTERED_BY = "nsc-seatlocator"

_lock = threading.Lock()
_client: Any = None
_dev_mode: bool | None = None


def build_client(teamusers_url: str, audience: str, issuer: str,
                 service_token: str):
    """构造 teamusers SDK Client（Verifier + PermissionsClient）。"""
    from teamusers_sdk import Client, PermissionsClient, Verifier
    verifier = Verifier(teamusers_url, audience=audience, issuer=issuer)
    permissions = PermissionsClient(teamusers_url,
                                    service_token=service_token)
    return Client(verifier=verifier, permissions=permissions)


def set_client(client: Any, dev: bool | None = None) -> None:
    """注入客户端（应用启动 / 测试）。"""
    global _client, _dev_mode
    with _lock:
        _client = client
        if dev is not None:
            _dev_mode = dev


def synthetic_claims():
    """SEAT_DEV 模式的合成身份。"""
    from teamusers_sdk import Claims
    return Claims(
        subject="dev-local",
        team="",
        kind="service",
        perm_ver=0,
        expiry=datetime.now(timezone.utc).replace(year=9999),
        audience="nekostick",
    )


def _get_client() -> Any:
    with _lock:
        return _client


def _is_dev() -> bool:
    with _lock:
        if _dev_mode is not None:
            return _dev_mode
    from app import config
    return config.from_env().dev


def require(permission: str):
    """FastAPI 依赖工厂：验证令牌 + 判定权限，返回 Claims。"""

    async def _dependency(request: Request):
        if _is_dev():
            return synthetic_claims()
        client = _get_client()
        if client is None:
            raise ProblemError(
                503, "config_unavailable",
                "teamusers 客户端未初始化（SEAT_DEV=false 时必须配置）")
        from teamusers_sdk import Authenticate, UnauthorizedError
        try:
            # 注意：传 {"headers": dict(...)} 而非原始 Request——
            # starlette Headers 不满足 SDK 的 Mapping 检查会被判为无令牌；
            # face-backend 的 auth.py 正是这么调用的。
            claims = Authenticate(
                {"headers": dict(request.headers)}, client.verifier)
        except UnauthorizedError as e:
            raise ProblemError(401, "unauthorized", str(e) or
                               "缺少或无效的 Bearer 令牌") from e
        result = client.allow(claims, permission, None)
        if not getattr(result, "allow", False):
            reason = getattr(result, "reason", "") or "permission denied"
            raise ProblemError(403, "permission_denied",
                               f"需要权限 {permission}: {reason}")
        return claims

    return _dependency


require_check = lambda: require(PERM_CHECK)      # noqa: E731
require_read = lambda: require(PERM_READ)        # noqa: E731
require_manage = lambda: require(PERM_MANAGE)    # noqa: E731
