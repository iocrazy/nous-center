"""SSRF 防护:fire_webhook 的 URL 校验(安全 P1)。用 IP 字面量,不依赖 DNS。"""
import pytest

from src.services.prediction_service import validate_webhook_url


class TestValidateWebhookUrl:
    @pytest.mark.parametrize("url", [
        "https://8.8.8.8/hook",
        "http://93.184.216.34/cb",       # 公网 IP
        "https://api.example.com/x",      # 公网域名(getaddrinfo 解析)
    ])
    def test_allows_public(self, url):
        # 公网域名可能因离线解析失败;IP 字面量必过
        err = validate_webhook_url(url)
        if url.replace("https://", "").replace("http://", "")[0].isdigit():
            assert err is None, err

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1/x",            # loopback
        "http://[::1]/x",                # ipv6 loopback
        "http://169.254.169.254/latest", # 云元数据
        "http://10.0.0.5/x",             # 私网 A
        "http://192.168.1.1/x",          # 私网 C
        "http://172.16.5.5/x",           # 私网 B
        "http://0.0.0.0/x",              # 未指定
    ])
    def test_blocks_private_and_metadata(self, url):
        assert validate_webhook_url(url) is not None

    @pytest.mark.parametrize("url", [
        "ftp://8.8.8.8/x", "file:///etc/passwd", "gopher://x", "", "not-a-url",
        "http://", "https:///nohostpath",
    ])
    def test_blocks_bad_scheme_or_malformed(self, url):
        assert validate_webhook_url(url) is not None
