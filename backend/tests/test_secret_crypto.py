"""API key 复看值静态加密(安全 review P1)。"""
import importlib

import pytest


@pytest.fixture
def crypto(monkeypatch):
    """重载 secret_crypto 并把 ADMIN_SESSION_SECRET 指到一个固定测试值。"""
    from src.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "ADMIN_SESSION_SECRET", "test-secret-0123456789abcdef", raising=False)
    import src.utils.secret_crypto as sc
    importlib.reload(sc)
    return sc


def test_roundtrip_encrypts_then_decrypts(crypto):
    plain = "sk-abcd-0123456789abcdef0123456789abcdef"
    stored = crypto.encrypt_secret(plain)
    assert stored != plain  # 不是裸明文
    assert stored.startswith("enc:v1:")
    assert plain not in stored  # 明文不出现在密文里
    assert crypto.decrypt_secret(stored) == plain


def test_legacy_plaintext_passes_through(crypto):
    # 历史裸明文行(无 enc: 前缀)原样返回,保证老 key 仍可复看。
    assert crypto.decrypt_secret("sk-old-plaintextkey") == "sk-old-plaintextkey"


def test_ciphertext_fits_column(crypto):
    # instance_api_keys.secret_plaintext 是 String(200);密文必须放得下。
    stored = crypto.encrypt_secret("sk-abcd-" + "0" * 32)
    assert len(stored) <= 200


def test_dev_mode_no_secret_stores_plaintext(monkeypatch):
    # ADMIN_SESSION_SECRET 空(dev,admin gate 关)→ 不加密,原样存/取。
    from src.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "ADMIN_SESSION_SECRET", "", raising=False)
    import src.utils.secret_crypto as sc
    importlib.reload(sc)
    plain = "sk-dev-key"
    assert sc.encrypt_secret(plain) == plain
    assert sc.decrypt_secret(plain) == plain


def test_wrong_secret_cannot_decrypt(monkeypatch):
    # 密文用 secret A 加密,换成 secret B 解密应失败返回 None(不泄露、不崩)。
    from src.config import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "ADMIN_SESSION_SECRET", "secret-A-aaaaaaaaaaaaaaaa", raising=False)
    import src.utils.secret_crypto as sc
    importlib.reload(sc)
    stored = sc.encrypt_secret("sk-abcd-secret")

    monkeypatch.setattr(settings, "ADMIN_SESSION_SECRET", "secret-B-bbbbbbbbbbbbbbbb", raising=False)
    importlib.reload(sc)
    assert sc.decrypt_secret(stored) is None
