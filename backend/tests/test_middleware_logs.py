from src.api.middleware import _audit_detail, derive_audit_action


def test_derive_audit_action():
    assert derive_audit_action("POST", "/api/v1/engines/cosyvoice2/load") == "load_engine"
    assert derive_audit_action("POST", "/api/v1/engines/cosyvoice2/unload") == "unload_engine"
    assert derive_audit_action("POST", "/api/v1/engines/reload") == "reload_registry"
    assert derive_audit_action("POST", "/api/v1/workflows") == "create_workflow"
    assert derive_audit_action("PATCH", "/api/v1/workflows/123") == "update_workflow"
    assert derive_audit_action("DELETE", "/api/v1/workflows/123") == "delete_workflow"
    assert derive_audit_action("POST", "/api/v1/workflows/123/publish-app") == "publish_app"
    assert derive_audit_action("DELETE", "/api/v1/apps/my-app") == "unpublish_app"
    # Fallback
    assert derive_audit_action("POST", "/api/v1/unknown/thing") == "post_thing"


def test_audit_detail_binary_and_nul():
    # multipart / 二进制上传 → 占位符,不记二进制乱码
    assert _audit_detail(b"\x00\x01bin", "multipart/form-data; boundary=x") == (
        "<multipart/form-data body, 5 bytes>"
    )
    assert _audit_detail(b"\xff\xfb", "audio/mpeg") == "<audio/mpeg body, 2 bytes>"
    # 文本 body 含 NUL(0x00)→ 必须剔除(PG text 存不下,否则 flush 失败丢审计行)
    out = _audit_detail(b'{"a":"b\x00c"}', "application/json")
    assert "\x00" not in out
    assert out == '{"a":"bc"}'
    # 空 body
    assert _audit_detail(b"", "application/json") == ""
    # 普通 JSON 原样(截断到 2000)
    assert _audit_detail(b'{"k":"v"}', "application/json") == '{"k":"v"}'
