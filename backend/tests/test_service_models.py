"""Unit tests for service_models.extract_service_models (display-only ref
enumeration for the service overview)."""
from src.services.service_models import extract_service_models


def _published(nodes: dict) -> dict:
    return {"schema": "comfy/api-1", "nodes": nodes}


def test_flux2_components_extracted_by_file():
    snap = _published({
        "n1": {"class_type": "flux2_load_diffusion_model",
               "inputs": {"file": "/m/flux2/transformer/x.safetensors", "device": "cuda:1"}},
        "n2": {"class_type": "flux2_load_clip",
               "inputs": {"clips": [{"file": "/m/clip/qwen.safetensors", "weight_dtype": "bf16"}]}},
        "n3": {"class_type": "flux2_load_vae",
               "inputs": {"file": "/m/vae/flux2-vae.safetensors"}},
    })
    refs = extract_service_models(snap)
    by_role = {r["role"]: r for r in refs}
    assert by_role["diffusion_models"]["file"] == "/m/flux2/transformer/x.safetensors"
    assert by_role["diffusion_models"]["label"] == "x.safetensors"
    assert by_role["clip"]["file"] == "/m/clip/qwen.safetensors"
    assert by_role["vae"]["label"] == "flux2-vae.safetensors"
    assert all(r["kind"] == "component" for r in refs)


def test_clip_backcompat_single_file():
    snap = _published({
        "n1": {"class_type": "flux2_load_clip", "inputs": {"file": "/m/clip/old.safetensors"}},
    })
    refs = extract_service_models(snap)
    assert len(refs) == 1
    assert refs[0]["role"] == "clip"
    assert refs[0]["file"] == "/m/clip/old.safetensors"


def test_llm_and_tts_engines():
    snap = _published({
        "n1": {"class_type": "llm", "inputs": {"model_key": "qwen3-8b"}},
        "n2": {"class_type": "tts_engine", "inputs": {"engine": "cosyvoice2"}},
    })
    refs = extract_service_models(snap)
    by_key = {r["engine_key"]: r for r in refs}
    assert by_key["qwen3-8b"]["kind"] == "engine"
    assert by_key["qwen3-8b"]["role"] == "llm"
    assert by_key["cosyvoice2"]["role"] == "tts"


def test_trivial_quick_provision_engine():
    # quick-provision uses class_type "LLMEngine"/"TTSEngine" + inputs.engine
    snap = _published({
        "engine_1": {"class_type": "LLMEngine", "inputs": {"engine": "qwen3-5"}},
    })
    refs = extract_service_models(snap)
    assert len(refs) == 1
    assert refs[0]["kind"] == "engine"
    assert refs[0]["engine_key"] == "qwen3-5"


def test_dedup_same_file_across_nodes():
    snap = _published({
        "a": {"class_type": "flux2_load_vae", "inputs": {"file": "/m/vae/v.safetensors"}},
        "b": {"class_type": "flux2_load_vae", "inputs": {"file": "/m/vae/v.safetensors"}},
    })
    assert len(extract_service_models(snap)) == 1


def test_empty_and_malformed_snapshots():
    assert extract_service_models(None) == []
    assert extract_service_models({}) == []
    assert extract_service_models({"nodes": {}}) == []
    assert extract_service_models({"nodes": [{"id": "x"}]}) == []


def test_editor_shape_list_with_type_data():
    # tolerate the editor shape (list of {id,type,data}) too
    snap = {"nodes": [
        {"id": "n1", "type": "flux2_load_vae", "data": {"file": "/m/vae/v.safetensors"}},
    ]}
    refs = extract_service_models(snap)
    assert len(refs) == 1 and refs[0]["role"] == "vae"
