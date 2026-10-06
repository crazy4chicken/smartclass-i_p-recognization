"""T3：imagefetch.py——URL 下载与防御（大小/像素/SSRF/解码）。"""
from __future__ import annotations

import numpy as np
import pytest

from app.imagefetch import (
    FetchedImage,
    ImageFetchError,
    fetch_image,
)


def _png_bytes(width: int, height: int) -> bytes:
    """生成合法 PNG 字节。"""
    import cv2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _client(handler):
    import httpx
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _resolve_public(host, port=0):
    """测试用解析器：一律返回公网地址。"""
    import socket
    return [(socket.AF_INET, None, None, "", ("93.184.216.34", port))]


async def _resolve_private(host, port=0):
    import socket
    return [(socket.AF_INET, None, None, "", ("192.168.1.10", port))]


class TestFetchImage:
    async def test_ok(self):
        png = _png_bytes(320, 240)

        def handler(request):
            return httpx.Response(200, content=png,
                                  headers={"Content-Type": "image/png"})

        import httpx
        async with _client(handler) as client:
            img = await fetch_image("https://files.example.com/p/1",
                                    client=client, resolver=_resolve_public,
                                    max_bytes=10_000_000, max_pixels=5_000_000)
        assert isinstance(img, FetchedImage)
        assert img.width == 320 and img.height == 240
        assert img.bgr.shape == (240, 320, 3)
        assert img.raw[:8] == b"\x89PNG\r\n\x1a\n"

    async def test_content_length_reject(self):
        import httpx

        def handler(request):
            return httpx.Response(200, content=b"x" * 100,
                                  headers={"Content-Length": "999999"})

        async with _client(handler) as client:
            with pytest.raises(ImageFetchError) as ei:
                await fetch_image("https://files.example.com/p/1",
                                  client=client, resolver=_resolve_public,
                                  max_bytes=1000, max_pixels=5_000_000)
        assert ei.value.status == 413
        assert ei.value.code == "image_too_large"

    async def test_body_too_large_no_content_length(self):
        import httpx
        # 模拟流式响应：MockTransport 无 CL 头
        def handler(request):
            return httpx.Response(200, content=b"x" * 5000,
                                  headers={"Content-Type": "image/png"})

        async with _client(handler) as client:
            with pytest.raises(ImageFetchError) as ei:
                await fetch_image("https://files.example.com/p/1",
                                  client=client, resolver=_resolve_public,
                                  max_bytes=1000, max_pixels=5_000_000)
        assert ei.value.status == 413

    async def test_pixel_bomb(self):
        # 声称很大：用真实编码的小图但 max_pixels 设小
        png = _png_bytes(2000, 2000)

        def handler(request):
            return httpx.Response(200, content=png)

        import httpx
        async with _client(handler) as client:
            with pytest.raises(ImageFetchError) as ei:
                await fetch_image("https://files.example.com/p/1",
                                  client=client, resolver=_resolve_public,
                                  max_bytes=50_000_000, max_pixels=1_000_000)
        assert ei.value.status == 422
        assert ei.value.code == "invalid_image"

    async def test_not_an_image(self):
        def handler(request):
            return httpx.Response(200, content=b"hello, not an image")

        import httpx
        async with _client(handler) as client:
            with pytest.raises(ImageFetchError) as ei:
                await fetch_image("https://files.example.com/p/1",
                                  client=client, resolver=_resolve_public,
                                  max_bytes=10_000, max_pixels=5_000_000)
        assert ei.value.status == 422
        assert ei.value.code == "invalid_image"

    async def test_upstream_404_maps_502(self):
        def handler(request):
            return httpx.Response(404)

        import httpx
        async with _client(handler) as client:
            with pytest.raises(ImageFetchError) as ei:
                await fetch_image("https://files.example.com/p/1",
                                  client=client, resolver=_resolve_public,
                                  max_bytes=10_000, max_pixels=5_000_000)
        assert ei.value.status == 502
        assert ei.value.code == "image_fetch_failed"

    async def test_ssrf_private_rejected(self):
        def handler(request):  # 不应被触达
            raise AssertionError("private host must be rejected before request")

        import httpx
        async with _client(handler) as client:
            with pytest.raises(ImageFetchError) as ei:
                await fetch_image("https://intranet.internal/p/1",
                                  client=client, resolver=_resolve_private,
                                  max_bytes=10_000, max_pixels=5_000_000)
        assert ei.value.status == 403
        assert ei.value.code == "image_host_forbidden"

    async def test_ssrf_private_allowed_when_flagged(self):
        """白名单模式（放行 webcam-server/files 内网域）通过。"""
        png = _png_bytes(64, 48)

        def handler(request):
            return httpx.Response(200, content=png)

        import httpx
        async with _client(handler) as client:
            img = await fetch_image("https://intranet.internal/p/1",
                                    client=client, resolver=_resolve_private,
                                    max_bytes=10_000, max_pixels=5_000_000,
                                    allow_private_hosts=True)
        assert img.width == 64

    async def test_non_http_scheme_rejected(self):
        import httpx
        async with _client(lambda r: None) as client:
            with pytest.raises(ImageFetchError) as ei:
                await fetch_image("ftp://files.example.com/p/1",
                                  client=client, resolver=_resolve_public,
                                  max_bytes=10_000, max_pixels=5_000_000)
        assert ei.value.status == 422
        assert ei.value.code == "invalid_url"

    async def test_grayscale_decoded_to_3ch(self):
        """灰度图自动扩展为 3 通道，保证下游模型输入形状稳定。"""
        import cv2
        g = np.full((100, 80), 128, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", g)
        assert ok

        def handler(request):
            return httpx.Response(200, content=buf.tobytes())

        import httpx
        async with _client(handler) as client:
            img = await fetch_image("https://files.example.com/p/1",
                                    client=client, resolver=_resolve_public,
                                    max_bytes=100_000, max_pixels=5_000_000)
        assert img.bgr.shape == (100, 80, 3)
