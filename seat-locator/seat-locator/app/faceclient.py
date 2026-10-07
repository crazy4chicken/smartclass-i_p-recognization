"""face-backend 客户端：FaceClient 协议 + HTTP 实现 + 测试桩。

契约（调研 face-backend app/main.py /recognize，2026-10-06）：
- POST {base}/recognize，multipart 字段 image=原图字节；
- Bearer 令牌透传（Q15：调用方令牌需含 face:check:any）；
- 200 → {"faces_found": N, "faces": [{bbox(原图坐标), det_score,
  matched, similarity, user_id|null}]}；无脸 = faces_found 0（200）。
失败映射：上游 5xx / 网络错误 / 超时 → 502 face_backend_error。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx


class FaceBackendError(Exception):
    """face-backend 调用失败（网关语义 502）。"""

    def __init__(self, detail: str, status: int = 502,
                 code: str = "face_backend_error"):
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class FaceResult:
    """一张人脸的识别结果。"""
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2（原图坐标）
    matched: bool
    user_id: str | None
    similarity: float
    det_score: float = 0.0


@runtime_checkable
class FaceClient(Protocol):
    async def recognize(self, image_bytes: bytes, *, bearer: str) \
            -> list[FaceResult]: ...


class HttpFaceClient:
    """真实实现：multipart 转发原图 + Bearer 透传。"""

    def __init__(self, base_url: str, *, client: httpx.AsyncClient,
                 timeout: float | None = None):
        self._base = base_url.rstrip("/")
        self._client = client
        self._timeout = timeout

    async def recognize(self, image_bytes: bytes, *, bearer: str) \
            -> list[FaceResult]:
        files = {"image": ("photo.jpg", image_bytes, "application/octet-stream")}
        headers = {"Authorization": f"Bearer {bearer}"}
        try:
            resp = await self._client.post(
                f"{self._base}/recognize", files=files, headers=headers,
                timeout=self._timeout)
        except httpx.HTTPError as e:
            raise FaceBackendError(f"face-backend 请求失败: {e}") from e
        if resp.status_code != 200:
            raise FaceBackendError(
                f"face-backend 返回 {resp.status_code}: "
                f"{resp.text[:200]}")
        try:
            payload = resp.json()
            faces = payload["faces"]
        except (ValueError, KeyError) as e:
            raise FaceBackendError("face-backend 响应格式异常") from e

        out: list[FaceResult] = []
        for f in faces:
            x1, y1, x2, y2 = (float(v) for v in f["bbox"])
            out.append(FaceResult(
                bbox=(x1, y1, x2, y2),
                matched=bool(f.get("matched")),
                user_id=f.get("user_id"),
                similarity=float(f.get("similarity") or 0.0),
                det_score=float(f.get("det_score") or 0.0),
            ))
        return out


class StubFaceClient:
    """测试桩：返回预配置结果，忽略输入。"""

    def __init__(self, results: list[FaceResult]):
        self.results = list(results)
        self.calls: list[bytes] = []

    async def recognize(self, image_bytes: bytes, *, bearer: str) \
            -> list[FaceResult]:
        self.calls.append(image_bytes)
        return self.results
