"""VoxCPM2 node executors — 30-language TTS with voice design & cloning."""

import base64
import io
import logging
import wave
import tempfile
import os

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}


def _numpy_to_wav_base64(samples: np.ndarray, sample_rate: int) -> str:
    """Convert numpy float32 audio to WAV base64 data URL."""
    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)
    samples = samples.astype(np.float32)
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"


def _base64_to_tempfile(data_url: str) -> str:
    """Save base64 audio data URL to a temp file, return path."""
    header, b64data = data_url.split(",", 1)
    audio_bytes = base64.b64decode(b64data)
    ext = ".wav"
    if "mp3" in header:
        ext = ".mp3"
    elif "flac" in header:
        ext = ".flac"
    elif "ogg" in header:
        ext = ".ogg"
    elif "m4a" in header:
        ext = ".m4a"
    fd, path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, "wb") as f:
        f.write(audio_bytes)
    return path


def _get_model(model_name: str, device: str = "auto", load_denoiser: bool = False):
    """Load or get cached VoxCPM model."""
    cache_key = (model_name, device, load_denoiser)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        from voxcpm import VoxCPM
    except ImportError:
        raise RuntimeError("voxcpm package not installed. Run: pip install voxcpm")

    from src.config import get_settings
    settings = get_settings()
    model_path = os.path.join(settings.LOCAL_MODELS_PATH, "tts", model_name)

    if not os.path.exists(model_path):
        # Try loading from HuggingFace/ModelScope ID
        model_path = f"openbmb/{model_name}"

    logger.info("Loading VoxCPM model from %s (device=%s, denoiser=%s)", model_path, device, load_denoiser)
    model = VoxCPM.from_pretrained(model_path, load_denoiser=load_denoiser)
    _MODEL_CACHE[cache_key] = model
    logger.info("VoxCPM model loaded")
    return model


# --- Node executors ---

async def exec_voxcpm2_load_model(data: dict, inputs: dict) -> dict:
    """Load VoxCPM2 model and pass as output."""
    model_name = data.get("model_name", "VoxCPM2")
    device = data.get("device", "auto")
    load_denoiser = data.get("load_denoiser", False)

    model = _get_model(model_name, device, load_denoiser)
    return {"model": {"_type": "voxcpm2", "model_name": model_name, "device": device, "load_denoiser": load_denoiser}}


async def exec_voxcpm2_generate(data: dict, inputs: dict) -> dict:
    """Generate speech with VoxCPM2."""
    import torch

    # Get model from input
    model_info = inputs.get("model", {})
    if not model_info or not isinstance(model_info, dict) or model_info.get("_type") != "voxcpm2":
        raise RuntimeError("VoxCPM2 模型未连接。请连接 VoxCPM2 加载模型节点。")

    model = _get_model(
        model_info.get("model_name", "VoxCPM2"),
        model_info.get("device", "auto"),
        model_info.get("load_denoiser", False),
    )

    text = inputs.get("text", "") or data.get("text", "")
    if not text:
        raise RuntimeError("缺少输入文本")

    mode = data.get("mode", "design")
    cfg_value = float(data.get("cfg_value", 2.0))
    inference_timesteps = int(data.get("inference_timesteps", 10))
    seed = int(data.get("seed", -1))

    if seed >= 0:
        torch.manual_seed(seed)

    # Reference audio handling
    ref_audio = inputs.get("reference_audio") or inputs.get("audio") or ""
    ref_path = None
    if ref_audio and ref_audio.startswith("data:"):
        ref_path = _base64_to_tempfile(ref_audio)

    try:
        if mode == "design":
            # Voice design mode — describe voice in text
            voice_desc = data.get("voice_description", "")
            if voice_desc:
                text = f"({voice_desc}){text}"
            wav = model.generate(
                text=text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )
        elif mode == "clone":
            # Controllable cloning
            if not ref_path:
                raise RuntimeError("可控克隆模式需要参考音频")
            wav = model.generate(
                text=text,
                reference_wav_path=ref_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
            )
        elif mode == "ultimate":
            # Ultimate cloning (audio continuation)
            if not ref_path:
                raise RuntimeError("终极克隆模式需要参考音频")
            prompt_text = data.get("prompt_text", "")
            wav = model.generate(
                text=text,
                prompt_wav_path=ref_path,
                prompt_text=prompt_text if prompt_text else None,
                reference_wav_path=ref_path,
            )
        else:
            raise RuntimeError(f"未知模式: {mode}")
    finally:
        # Clean up temp file
        if ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)

    # Convert to base64 audio
    sample_rate = model.tts_model.sample_rate
    audio_b64 = _numpy_to_wav_base64(wav, sample_rate)

    return {"audio": audio_b64}


# Register executors
EXECUTORS = {
    "voxcpm2_load_model": exec_voxcpm2_load_model,
    "voxcpm2_generate": exec_voxcpm2_generate,
}
