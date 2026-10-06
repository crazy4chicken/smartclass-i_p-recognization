"""RFC 9457 problem details：集群统一失败响应体。

对齐 dispatchub 的失败语义：
- 401 unauthorized：缺少/无效 Bearer 令牌
- 403 permission_denied：已认证但无权限（fail-closed）
- 503 config_unavailable：依赖不可用（teamusers / PG）
其余 4xx/5xx 由各模块通过 ProblemError 抛出。
"""
from __future__ import annotations

from typing import Any

CONTENT_TYPE = "application/problem+json"


class ProblemError(Exception):
    """携带 RFC 9457 字段的业务异常，由 api 层统一转成响应。"""

    def __init__(self, status: int, code: str, detail: str,
                 **extra: Any):
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.extra = extra

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": f"https://seatlocator.problems/{self.code}",
            "title": self.code,
            "status": self.status,
            "detail": self.detail,
        }
        out.update(self.extra)
        return out


def install_problem_handler(app) -> None:
    """在 FastAPI 应用上注册 ProblemError → RFC 9457 响应。"""
    from fastapi.responses import JSONResponse

    @app.exception_handler(ProblemError)
    async def _problem(_request, exc: ProblemError):
        return JSONResponse(status_code=exc.status,
                            content=exc.to_dict(),
                            media_type=CONTENT_TYPE)
