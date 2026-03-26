"""Qwen3-TTS node executors -- ported from ComfyUI-Qwen-TTS."""

import base64
import io
import logging
import os
import wave

import numpy as np

logger = logging.getLogger(__name__)

# Model cache (same pattern as ComfyUI version)
_MODEL_CACHE = {}


def _numpy_to_wav_base64(samples: np.ndarray, sample_rate: int) -> str:
    """Convert numpy float32 audio array to WAV base64 string."""
    # Ensure float32 mono
    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)
    samples = samples.astype(np.float32)

    # Convert to int16 PCM
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return base64.b64encode(buf.getvalue()).decode()


def _get_qwen_model(model_type: str, model_choice: str, device: str = "cuda", precision: str = "bf16"):
    """Load or get cached Qwen3TTSModel."""
    cache_key = (model_type, model_choice, device, precision)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError:
        raise RuntimeError("qwen_tts package not installed. Run: pip install qwen-tts")

    import torch
    from src.config import get_settings

    settings = get_settings()

    # Model path mapping
    HF_MAP = {
        ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ("CustomVoice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        ("CustomVoice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    }

    source = HF_MAP.get((model_type, model_choice), "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    # Check local path first
    local_dir = os.path.join(settings.LOCAL_MODELS_PATH, "tts", source.split("/")[-1].lower())
    if os.path.exists(local_dir):
        source = local_dir

    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    model = Qwen3TTSModel.from_pretrained(source, device_map=device, dtype=dtype)

    _MODEL_CACHE[cache_key] = model
    return model


def _parse_model_cache_key(inputs: dict, default_model_type: str) -> tuple:
    """Parse model cache key from inputs, with fallback defaults."""
    model_cache_key = inputs.get("model", "")
    if model_cache_key and "|" in str(model_cache_key):
        parts = str(model_cache_key).split("|")
        return parts[0], parts[1], parts[2], parts[3]
    # Fallback: backward compatibility
    return default_model_type, "1.7B", "auto", "bf16"


async def exec_qwen3_model_loader(data: dict, inputs: dict) -> dict:
    """Load Qwen3-TTS model and return cache key."""
    import asyncio

    model_name = data.get("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    device = data.get("device", "auto")
    precision = data.get("precision", "bf16")

    # Determine model type from name
    if "VoiceDesign" in model_name:
        model_type = "VoiceDesign"
    elif "CustomVoice" in model_name:
        model_type = "CustomVoice"
    else:
        model_type = "Base"

    # Determine model choice (0.6B or 1.7B)
    model_choice = "0.6B" if "0.6B" in model_name else "1.7B"

    # Load model (cached)
    await asyncio.to_thread(_get_qwen_model, model_type, model_choice, device, precision)

    # Return cache key so downstream nodes can retrieve the model
    cache_key = f"{model_type}|{model_choice}|{device}|{precision}"
    return {"model": cache_key}


async def exec_qwen3_voice_clone(data: dict, inputs: dict) -> dict:
    """Voice clone from reference audio."""
    import asyncio

    import torch

    text = inputs.get("text", data.get("text", ""))
    if not text:
        raise RuntimeError("文本不能为空")

    # Get model from inputs (loaded by model_loader node)
    model_type, model_choice, device, precision = _parse_model_cache_key(inputs, "Base")

    model = await asyncio.to_thread(_get_qwen_model, model_type, model_choice, device, precision)

    seed = int(data.get("seed", 0))
    torch.manual_seed(seed)

    wavs, sr = await asyncio.to_thread(
        model.generate_voice_clone,
        text=text,
        language=data.get("language", "auto"),
        ref_audio=inputs.get("ref_audio"),
        ref_text=data.get("ref_text"),
        max_new_tokens=int(data.get("max_new_tokens", 2048)),
        top_p=float(data.get("top_p", 0.8)),
        temperature=float(data.get("temperature", 1.0)),
    )

    if not wavs:
        raise RuntimeError("语音生成失败：无输出")
    audio_b64 = _numpy_to_wav_base64(np.array(wavs[0]), sr)
    return {"audio": audio_b64, "sample_rate": sr}


async def exec_qwen3_custom_voice(data: dict, inputs: dict) -> dict:
    """Custom voice TTS with preset speakers."""
    import asyncio

    text = inputs.get("text", data.get("text", ""))
    if not text:
        raise RuntimeError("文本不能为空")

    # Get model from inputs (loaded by model_loader node)
    model_type, model_choice, device, precision = _parse_model_cache_key(inputs, "CustomVoice")

    model = await asyncio.to_thread(_get_qwen_model, model_type, model_choice, device, precision)

    wavs, sr = await asyncio.to_thread(
        model.generate_custom_voice,
        text=text,
        language=data.get("language", "auto"),
        speaker=data.get("speaker", "ryan"),
        instruct=data.get("instruct") or None,
        max_new_tokens=int(data.get("max_new_tokens", 2048)),
        top_p=float(data.get("top_p", 0.8)),
        temperature=float(data.get("temperature", 1.0)),
    )

    if not wavs:
        raise RuntimeError("语音生成失败：无输出")
    audio_b64 = _numpy_to_wav_base64(np.array(wavs[0]), sr)
    return {"audio": audio_b64, "sample_rate": sr}


async def exec_qwen3_voice_design(data: dict, inputs: dict) -> dict:
    """Voice design from text description."""
    import asyncio

    text = inputs.get("text", data.get("text", ""))
    instruct = data.get("instruct", "")
    if not text or not instruct:
        raise RuntimeError("文本和音色描述都不能为空")

    # Get model from inputs (loaded by model_loader node)
    model_type, model_choice, device, precision = _parse_model_cache_key(inputs, "VoiceDesign")

    model = await asyncio.to_thread(_get_qwen_model, model_type, model_choice, device, precision)

    wavs, sr = await asyncio.to_thread(
        model.generate_voice_design,
        text=text,
        language=data.get("language", "auto"),
        instruct=instruct,
        max_new_tokens=int(data.get("max_new_tokens", 2048)),
        top_p=float(data.get("top_p", 0.8)),
        temperature=float(data.get("temperature", 1.0)),
    )

    if not wavs:
        raise RuntimeError("语音生成失败：无输出")
    audio_b64 = _numpy_to_wav_base64(np.array(wavs[0]), sr)
    return {"audio": audio_b64, "sample_rate": sr}


# Register executors -- this dict is read by the node package scanner
EXECUTORS = {
    "qwen3_model_loader": exec_qwen3_model_loader,
    "qwen3_voice_clone": exec_qwen3_voice_clone,
    "qwen3_custom_voice": exec_qwen3_custom_voice,
    "qwen3_voice_design": exec_qwen3_voice_design,
}
