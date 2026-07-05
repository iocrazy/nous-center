"""SSRF URL 白名单(安全 review P2)。"""
import pytest

from src.utils.url_security import UnsafeURLError, validate_external_image_url


def test_data_image_url_allowed():
    # 内联 base64,不发起网络请求 → 放行。
    validate_external_image_url("data:image/png;base64,iVBORw0KGgo=")


def test_local_path_allowed():
    # 无 scheme 的本地路径不是网络 fetch → 放行(保留端点既有契约)。
    validate_external_image_url("/path/to/img.png")


def test_http_scheme_rejected():
    with pytest.raises(UnsafeURLError):
        validate_external_image_url("http://example.com/x.png")


def test_file_scheme_rejected():
    with pytest.raises(UnsafeURLError):
        validate_external_image_url("file:///etc/passwd")


def test_loopback_rejected(monkeypatch):
    import src.utils.url_security as us
    monkeypatch.setattr(us.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 0))])
    with pytest.raises(UnsafeURLError):
        validate_external_image_url("https://sneaky.internal/x.png")


def test_metadata_ip_rejected(monkeypatch):
    import src.utils.url_security as us
    monkeypatch.setattr(us.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("169.254.169.254", 0))])
    with pytest.raises(UnsafeURLError):
        validate_external_image_url("https://metadata/x.png")


def test_private_ip_rejected(monkeypatch):
    import src.utils.url_security as us
    monkeypatch.setattr(us.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("10.0.0.10", 0))])
    with pytest.raises(UnsafeURLError):
        validate_external_image_url("https://vpn-host/x.png")


def test_public_https_allowed(monkeypatch):
    import src.utils.url_security as us
    monkeypatch.setattr(us.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    validate_external_image_url("https://example.com/x.png")


def test_mixed_resolution_rejected(monkeypatch):
    # 一个主机同时解析到公网 + 私网 IP → 拒(DNS rebinding 面)。
    import src.utils.url_security as us
    monkeypatch.setattr(us.socket, "getaddrinfo",
                        lambda *a, **k: [
                            (2, 1, 6, "", ("93.184.216.34", 0)),
                            (2, 1, 6, "", ("127.0.0.1", 0)),
                        ])
    with pytest.raises(UnsafeURLError):
        validate_external_image_url("https://rebind.example/x.png")
