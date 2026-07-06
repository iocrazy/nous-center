"""二轮安全:SSRF 残留 —— chat messages image_url + responses input_image + verify_token 加固。"""
import pytest

from src.utils.url_security import UnsafeURLError, validate_chat_image_urls


class TestChatImageUrls:
    @pytest.mark.asyncio
    async def test_blocks_private_image_url(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest"}},
        ]}]
        with pytest.raises(UnsafeURLError):
            await validate_chat_image_urls(messages)

    @pytest.mark.asyncio
    async def test_allows_data_uri_and_public(self):
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "image_url", "image_url": {"url": "https://8.8.8.8/x.png"}},
        ]}]
        await validate_chat_image_urls(messages)  # 不抛

    @pytest.mark.asyncio
    async def test_ignores_text_only_and_none(self):
        await validate_chat_image_urls(None)
        await validate_chat_image_urls([{"role": "user", "content": "just text"}])


class TestVerifyTokenRobustness:
    def test_verify_token_no_secret_returns_false_not_raise(self, monkeypatch):
        """token-only 部署(ADMIN_SESSION_SECRET 空):含 '.' 的伪造 cookie → False,不抛 500。"""
        import src.api.admin_session as sess
        def _boom():
            raise RuntimeError("ADMIN_SESSION_SECRET must be set")
        monkeypatch.setattr(sess, "_secret", _boom)
        # future expiry 以越过前面的过期检查,逼到 _secret() 分支
        forged = f"{int(__import__('time').time()) + 9999}.deadbeef"
        assert sess.verify_token(forged) is False
