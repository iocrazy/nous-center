"""对称加密下游 API key 明文,用于"随时复看"功能的静态存储保护。

背景(安全 review P1):`instance_api_keys.secret_plaintext` 原先裸存完整明文 key
(`sk-xxx-...`),DB 一旦泄露(备份落 NAS 异机、误 dump)所有下游 key 尽失,bcrypt
形同虚设。这里用 `ADMIN_SESSION_SECRET`(已是 HMAC 会话密钥的单一来源)HKDF 派生一把
Fernet 对称密钥,把明文加密后再落库,复看时解密。DB 泄露但 .env 未泄露 → 密文无用。

- 密文带 `enc:v1:` 前缀标记 → 与历史裸明文行区分:`decrypt_secret` 见到无前缀的值
  当作 legacy 明文原样返回(平滑迁移,老 key 仍可复看,下次 reset 自动升级为密文)。
- dev 模式(`ADMIN_SESSION_SECRET` 空、admin gate 关)无安全边界 → 不加密,原样存,
  `encrypt_secret` 直接返回明文。
"""
from __future__ import annotations

import base64

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from src.config import get_settings

_MARKER = "enc:v1:"
_HKDF_INFO = b"nous-center api-key secret-at-rest v1"


def _fernet() -> Fernet | None:
    """从 ADMIN_SESSION_SECRET HKDF 派生 Fernet 密钥;secret 空(dev)→ None。"""
    secret = get_settings().ADMIN_SESSION_SECRET
    if not secret:
        return None
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=_HKDF_INFO,
    ).derive(secret.encode("utf-8"))
    return Fernet(base64.urlsafe_b64encode(key))


def encrypt_secret(plaintext: str) -> str:
    """明文 → 落库值。有 ADMIN_SESSION_SECRET 则返回 `enc:v1:<token>`,否则原样返回。"""
    f = _fernet()
    if f is None:
        return plaintext
    token = f.encrypt(plaintext.encode("utf-8")).decode("ascii")
    return _MARKER + token


def decrypt_secret(stored: str | None) -> str | None:
    """落库值 → 明文。无前缀 = legacy 裸明文,原样返回;密文解密失败返回 None(不炸)。"""
    if stored is None:
        return None
    if not stored.startswith(_MARKER):
        return stored  # legacy plaintext row
    f = _fernet()
    if f is None:
        # 有密文却没 secret → 无法解密(secret 丢了),别把密文当明文返回给 UI。
        return None
    try:
        return f.decrypt(stored[len(_MARKER):].encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError):
        return None
