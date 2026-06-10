from src.api.middleware import derive_audit_action


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
