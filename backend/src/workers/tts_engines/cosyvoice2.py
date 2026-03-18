import io
import logging
import struct
import sys
from pathlib import Path

import torch

from src.config import get_settings
from src.workers.tts_engines.base import TTSEngine, TTSResult
from src.workers.tts_engines.registry import register_engine

logger = logging.getLogger(__name__)

# Default reference audio for zero-shot when no reference is provided
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # backend/
_DEFAULT_VOICES = {
    "default": {
        "wav": str(_PROJECT_ROOT.parent / "assets" / "voices" / "default_zh_female.wav"),
        "prompt_text": "希望你以后能够做的比我还好呦。",
    },
}


def _ensure_cosyvoice_path():
    """Add CosyVoice repo to sys.path if not already present."""
    settings = get_settings()
    repo_path = settings.COSYVOICE_REPO_PATH
    matcha_path = str(Path(repo_path) / "third_party" / "Matcha-TTS")
    for p in [repo_path, matcha_path]:
        if p not in sys.path:
            sys.path.insert(0, p)


@register_engine
class CosyVoice2Engine(TTSEngine):
    ENGINE_NAME = "cosyvoice2"

    def load(self) -> None:
        logger.info("Loading CosyVoice2-0.5B from %s (device=%s)", self.model_path, self.device)
        _ensure_cosyvoice_path()

        # CosyVoice2Model.__init__ hardcodes self.device = torch.device('cuda')
        # which always resolves to cuda:0. We patch __init__ to override
        # self.device to our target AFTER the original runs but BEFORE .load()
        # moves weights. We also rebuild the CUDA stream on the correct device.
        target_device = torch.device(self.device)
        from cosyvoice.cli.model import CosyVoice2Model
        _orig_init = CosyVoice2Model.__init__

        def _patched_init(self_model, llm, flow, hift, fp16):
            _orig_init(self_model, llm, flow, hift, fp16)
            # Override device and rebuild CUDA stream on target GPU
            self_model.device = target_device
            if torch.cuda.is_available():
                self_model.llm_context = torch.cuda.stream(
                    torch.cuda.Stream(target_device)
                )

        CosyVoice2Model.__init__ = _patched_init
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
            self._model = CosyVoice2(
                str(self.model_path),
                load_jit=False,
                load_trt=False,
                fp16=torch.cuda.is_available(),
            )
        finally:
            CosyVoice2Model.__init__ = _orig_init
        logger.info(
            "CosyVoice2-0.5B loaded, sample_rate=%d", self._model.sample_rate
        )

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        sample_rate: int = 24000,
        reference_audio: str | None = None,
        reference_text: str | None = None,
        emotion: str | None = None,
    ) -> TTSResult:
        if not self.is_loaded:
            raise RuntimeError("CosyVoice2 model not loaded. Call load() first.")

        logger.info("Synthesizing: %s (voice=%s, speed=%.1f)", text[:50], voice, speed)

        # Resolve reference audio and prompt text
        if reference_audio:
            prompt_wav = reference_audio
            prompt_text = ""  # User-provided audio, no transcript needed for cross-lingual
        elif voice in _DEFAULT_VOICES:
            prompt_wav = _DEFAULT_VOICES[voice]["wav"]
            prompt_text = _DEFAULT_VOICES[voice]["prompt_text"]
        else:
            prompt_wav = _DEFAULT_VOICES["default"]["wav"]
            prompt_text = _DEFAULT_VOICES["default"]["prompt_text"]

        # Collect all chunks from the generator
        chunks = []
        if reference_audio and not prompt_text:
            # Cross-lingual mode when we have audio but no transcript
            for chunk in self._model.inference_cross_lingual(
                text, prompt_wav, stream=False, speed=speed
            ):
                chunks.append(chunk["tts_speech"])
        else:
            # Zero-shot mode with prompt text + audio
            for chunk in self._model.inference_zero_shot(
                text, prompt_text, prompt_wav, stream=False, speed=speed
            ):
                chunks.append(chunk["tts_speech"])

        if not chunks:
            return TTSResult(audio_bytes=b"", sample_rate=sample_rate, duration_seconds=0.0)

        # Concatenate all speech chunks
        speech = torch.cat(chunks, dim=1)
        native_sr = self._model.sample_rate

        # Resample if requested sample rate differs from native
        if sample_rate != native_sr:
            import torchaudio
            speech = torchaudio.functional.resample(speech, native_sr, sample_rate)

        # Convert tensor to WAV bytes using soundfile (avoids torchcodec issues)
        import soundfile as sf
        import numpy as np
        audio_np = speech.squeeze(0).cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        duration = speech.shape[1] / sample_rate

        return TTSResult(
            audio_bytes=wav_bytes,
            sample_rate=sample_rate,
            duration_seconds=round(duration, 2),
            format="wav",
        )

    @property
    def engine_name(self) -> str:
        return self.ENGINE_NAME

    @property
    def supported_voices(self) -> list[str]:
        return list(_DEFAULT_VOICES.keys())
