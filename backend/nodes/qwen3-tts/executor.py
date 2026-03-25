"""Qwen3-TTS node executors -- ported from ComfyUI-Qwen-TTS."""

import base64
import io
import logging
import os
import struct
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


async def exec_qwen3_voice_clone(data: dict, inputs: dict) -> dict:
    """Voice clone from reference audio."""
    import asyncio

    import torch

    text = inputs.get("text", data.get("text", ""))
    if not text:
        raise RuntimeError("文本不能为空")

    model = await asyncio.to_thread(
        _get_qwen_model, "Base", data.get("model_choice", "1.7B")
    )

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

    model = await asyncio.to_thread(
        _get_qwen_model, "CustomVoice", data.get("model_choice", "1.7B")
    )

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

    model = await asyncio.to_thread(
        _get_qwen_model, "VoiceDesign", data.get("model_choice", "1.7B")
    )

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
    "qwen3_voice_clone": exec_qwen3_voice_clone,
    "qwen3_custom_voice": exec_qwen3_custom_voice,
    "qwen3_voice_design": exec_qwen3_voice_design,
}
