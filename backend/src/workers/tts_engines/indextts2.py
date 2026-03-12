import io
import logging
import sys
from pathlib import Path

import torch

from src.config import get_settings
from src.workers.tts_engines.base import TTSEngine, TTSResult
from src.workers.tts_engines.registry import register_engine

logger = logging.getLogger(__name__)

# Default reference audio (same as CosyVoice2 default)
_DEFAULT_VOICE_WAV = "assets/voices/default_zh_female.wav"


def _ensure_indextts_path():
    """Add IndexTTS repo to sys.path if not already present."""
    settings = get_settings()
    repo_path = settings.INDEXTTS_REPO_PATH
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


@register_engine
class IndexTTS2Engine(TTSEngine):
    ENGINE_NAME = "indextts2"

    def load(self) -> None:
        logger.info("Loading IndexTTS-2 from %s", self.model_path)
        _ensure_indextts_path()
        from indextts.infer_v2 import IndexTTS2

        cfg_path = str(self.model_path / "config.yaml")
        self._model = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=str(self.model_path),
            use_fp16=torch.cuda.is_available(),
        )
        logger.info("IndexTTS-2 loaded")

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        sample_rate: int = 24000,
        reference_audio: str | None = None,
    ) -> TTSResult:
        if not self.is_loaded:
            raise RuntimeError("IndexTTS-2 model not loaded. Call load() first.")

        logger.info("Synthesizing with IndexTTS-2: %s", text[:50])

        # Resolve reference audio
        ref_audio = reference_audio or _DEFAULT_VOICE_WAV

        # IndexTTS2.infer with output_path=None returns (sample_rate, np.ndarray)
        sr, audio_np = self._model.infer(
            spk_audio_prompt=ref_audio,
            text=text,
            output_path=None,
            verbose=False,
        )

        # Convert numpy int16 to WAV bytes using soundfile
        import soundfile as sf
        import numpy as np

        # audio_np from IndexTTS is int16 format, convert to float for soundfile
        if audio_np.dtype == np.int16:
            audio_float = audio_np.astype(np.float32) / 32768.0
        else:
            audio_float = audio_np.astype(np.float32)

        # Handle shape: (samples, channels) or (samples,)
        if audio_float.ndim > 1:
            audio_float = audio_float[:, 0]  # take first channel

        # Resample if needed
        native_sr = sr
        if sample_rate != native_sr:
            import torchaudio
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, native_sr, sample_rate)
            audio_float = audio_tensor.squeeze(0).numpy()

        buf = io.BytesIO()
        sf.write(buf, audio_float, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        duration = round(len(audio_float) / sample_rate, 2)

        return TTSResult(
            audio_bytes=wav_bytes,
            sample_rate=sample_rate,
            duration_seconds=duration,
            format="wav",
        )

    @property
    def engine_name(self) -> str:
        return self.ENGINE_NAME
