"""外部 URL 的 SSRF 防护 —— 用于任何"服务端(或其下游)会去 fetch 用户提供的 URL"的入口。

背景(安全 review P2):`/api/v1/understand/image` 把用户给的 `image_url` 原样透传给
vLLM,vLLM 服务端会去 fetch → 可探测内网(`http://127.0.0.1:xxxx`、云 metadata
`169.254.169.254`)。这里对 URL 做 scheme + 解析后 IP 的白名单校验。
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class UnsafeURLError(ValueError):
    """URL 未通过 SSRF 白名单校验。"""


def _all_resolved_ips_public(host: str) -> bool:
    """host 解析出的**每个** IP 都是公网地址才算安全(有一个私网就拒)。"""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        # is_global 已排除 private / loopback / link-local / reserved / multicast。
        if not ip.is_global:
            return False
    return True


def validate_external_image_url(url: str) -> None:
    """校验可安全交给下游 fetch 的 image URL。不安全则抛 UnsafeURLError。

    放行:
    - `data:image/...` 内联(base64,根本不发起网络请求 → 天然安全)
    - 无 scheme 的本地路径/相对引用(如 `/path/to/img.png`)—— 不是网络 fetch,SSRF 不适用
    - `https://<公网主机>`(解析出的所有 IP 均为公网)
    拒绝:http(常用于打内网)、file:// 等其它 scheme、非公网主机、无法解析的主机。
    """
    if url.startswith("data:image/"):
        return
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme == "":
        # 本地路径/相对引用,不发起网络请求 → 不是 SSRF 面。
        return
    if scheme != "https":
        # http(打内网常用)、file://(LFI)、gopher:// 等一律拒。
        raise UnsafeURLError(
            "network image_url must use https:// (or inline data:image/); "
            f"got scheme {scheme!r}"
        )
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("image_url has no host")
    if not _all_resolved_ips_public(host):
        raise UnsafeURLError(
            f"image_url host {host!r} does not resolve to a public address "
            "(private/loopback/metadata targets are blocked)"
        )


async def validate_chat_image_urls(messages: list | None) -> None:
    """遍历 OpenAI chat messages 的 content 数组,对每个 image_url part 做 SSRF 校验。

    vLLM 服务端会 fetch messages[].content[].image_url.url → 任何下游 API-key 持有者
    可用 http://169.254.169.254/... 或 http://127.0.0.1:<port> 探测内网/云 metadata。
    与 understand/webhook 口径一致(放行 data:/公网 https,拒私网/http/非法主机)。
    getaddrinfo 阻塞 → 逐个 to_thread。命中即抛 UnsafeURLError,调用方转 400。
    """
    import asyncio
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if isinstance(url, str) and url:
                    await asyncio.to_thread(validate_external_image_url, url)
