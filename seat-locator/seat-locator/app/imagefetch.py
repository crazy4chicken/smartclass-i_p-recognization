"""图片获取：下载 image_url 并做防御性校验。

对齐 face-backend 的防御姿势（Content-Length 预拒 + 像素上限）
并叠加 SSRF 防护（私网默认拒绝，白名单显式放行内网文件域）。

失败语义（spec §6）：
- 422 invalid_url / invalid_image
- 413 image_too_large
- 403 image_host_forbidden（SSRF 拒绝）
- 502 image_fetch_failed（上游非 2xx / 网络错误 / 超时）
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np


class ImageFetchError(Exception):
    """携带 HTTP 语义的图片获取失败。"""

    def __init__(self, status: int, code: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class FetchedImage:
    """下载并解码后的图片：raw 供 face-backend 原样转发，bgr 供检测。"""
    raw: bytes
    width: int
    height: int
    bgr: np.ndarray


class _Resolver(Protocol):
    def __call__(self, host: str, port: int = 0) -> Awaitable: ...


async def _default_resolver(host: str, port: int = 0):
    return socket.getaddrinfo(host, port or 0, proto=socket.IPPROTO_TCP)


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # 无法解析的地址一律视为不可信
    return addr.is_private or addr.is_loopback or addr.is_link_local \
        or addr.is_reserved or addr.is_multicast or addr.is_unspecified


async def fetch_image(
    url: str,
    *,
    client: httpx.AsyncClient,
    max_bytes: int,
    max_pixels: int,
    resolver: _Resolver = _default_resolver,
    allow_private_hosts: bool = False,
    timeout: float | None = None,
) -> FetchedImage:
    """下载并解码图片；全部失败路径抛 ImageFetchError。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ImageFetchError(422, "invalid_url",
                              f"仅支持 http/https 图片 URL: {url!r}")

    # --- SSRF 防护：请求前解析并校验目标地址 ---
    if not allow_private_hosts:
        try:
            infos = await resolver(parsed.hostname, parsed.port or 0)
        except (socket.gaierror, OSError) as e:
            raise ImageFetchError(502, "image_fetch_failed",
                                  f"域名解析失败: {parsed.hostname}") from e
        for info in infos:
            ip = info[4][0]
            if _is_private_ip(ip):
                raise ImageFetchError(
                    403, "image_host_forbidden",
                    f"目标主机 {parsed.hostname}({ip}) 位于私有网段，"
                    "默认拒绝；如需放行请配置 SEAT_ALLOW_PRIVATE_HOSTS")

    # --- Content-Length 预拒（face-backend 同款防御）---
    try:
        head = await client.send(client.build_request("HEAD", url))
        declared = head.headers.get("Content-Length")
        if declared is not None and declared.isdigit() \
                and int(declared) > max_bytes:
            raise ImageFetchError(
                413, "image_too_large",
                f"图片超过大小上限 {max_bytes} 字节")
    except httpx.HTTPError:
        pass  # HEAD 不被支持时由 GET 阶段兜底

    # --- GET 下载 ---
    try:
        resp = await client.get(url, timeout=timeout)
    except httpx.HTTPError as e:
        raise ImageFetchError(502, "image_fetch_failed",
                              f"下载失败: {e}") from e
    if resp.status_code != 200:
        raise ImageFetchError(502, "image_fetch_failed",
                              f"上游返回 {resp.status_code}")
    data = resp.content
    if len(data) > max_bytes:
        raise ImageFetchError(413, "image_too_large",
                              f"图片超过大小上限 {max_bytes} 字节")
    if not data:
        raise ImageFetchError(422, "invalid_image", "图片内容为空")

    # --- 解码 + 像素上限 ---
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # None → 非法图片
    if bgr is None:
        raise ImageFetchError(422, "invalid_image",
                              "无法解码为图片（格式不支持或已损坏）")
    h, w = bgr.shape[:2]
    if w * h > max_pixels:
        raise ImageFetchError(422, "invalid_image",
                              f"图片像素 {w}x{h} 超过上限 {max_pixels}")
    return FetchedImage(raw=data, width=w, height=h, bgr=bgr)
