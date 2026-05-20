"""PR-6 L2 cache smoke (no GPU): real write_image → cache anchor → serve/re-sign
→ reaped-miss. Verifies the anchor/re-sign path on real disk + real HMAC signing.

Run: cd backend && set -a && source .env && set +a && PYTHONPATH=. .venv/bin/python scripts/smoke_pr6_l2.py
"""
from src.services.image_output_storage import write_image, verify_token
from src.services.inference.image_l2_cache import ImageOutputCache, serve_image_l2

rec = write_image(b"\x89PNG\r\n\x1a\n", ext="png", ttl_seconds=3600)
print(f"[1] wrote uuid={rec['uuid']} date={rec['date']} signed_url={'yes' if rec['url'] else 'NO (no ADMIN_SESSION_SECRET)'}")

cache = ImageOutputCache()
entry = {"image_uuid": rec["uuid"], "date": rec["date"], "ext": "png", "meta": {"seed": 42}, "width": 512, "height": 512}
cache.put("k", entry)

hit = serve_image_l2(cache.get("k"), ttl=3600)
ok_url = False
if hit and hit["image_url"]:
    tok = hit["image_url"].split("token=")[1].split("&")[0]
    exp = int(hit["image_url"].split("expires=")[1])
    ok_url = verify_token(rec["uuid"], exp, tok)
print(f"[2] L2 hit: cached={hit and hit['cached']} re-signed_url_valid={ok_url}")

rec["path"].unlink()
miss = serve_image_l2(cache.get("k"), ttl=3600)
print(f"[3] after PNG reaped: serve={miss} (expect None = miss → recompute)")

result = "PASS" if (hit and hit["cached"] and miss is None) else "FAIL"
print(f"\nRESULT: {result}")
