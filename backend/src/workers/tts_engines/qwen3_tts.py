import io
import logging
from pathlib import Path

import torch

from src.workers.tts_engines.base import TTSEngine, TTSResult
from src.workers.tts_engines.registry import register_engine

logger = logging.getLogger(__name__)

# Default reference audio for voice clone (Base model)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_VOICE_WAV = str(_PROJECT_ROOT.parent / "assets" / "voices" / "default_zh_female.wav")
_DEFAULT_VOICE_TEXT = "希望你以后能够做的比我还好呦。"


class Qwen3TTSBase(TTSEngine):
    """Shared loading logic for all Qwen3-TTS variants.

    Requires the `qwen-tts` package (pip install qwen-tts).
    NOTE: qwen-tts requires transformers >= 4.57.3, which conflicts
    with CosyVoice2/IndexTTS (transformers 4.52.x). Run in a separate
    worker environment if needed.
    """

    def _to_result(self, audio_np, native_sr: int, target_sr: int) -> TTSResult:
        """Convert numpy audio array to TTSResult with optional resampling."""
        import numpy as np
        import soundfile as sf

        audio = audio_np.astype(np.float32) if audio_np.dtype != np.float32 else audio_np

        if audio.ndim > 1:
            audio = audio[:, 0]

        if target_sr != native_sr:
            import torchaudio
            tensor = torch.from_numpy(audio).unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, native_sr, target_sr)
            audio = tensor.squeeze(0).numpy()

        buf = io.BytesIO()
        sf.write(buf, audio, target_sr, format="WAV", subtype="PCM_16")

        duration = round(len(audio) / target_sr, 2)
        return TTSResult(
            audio_bytes=buf.getvalue(),
            sample_rate=target_sr,
            duration_seconds=duration,
            format="wav",
        )

    def _load_model(self) -> None:
        logger.info("Loading %s from %s", self.ENGINE_NAME, self.model_path)
        from qwen_tts import Qwen3TTSModel

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = Qwen3TTSModel.from_pretrained(
            str(self.model_path),
            device_map=self.device,
            dtype=dtype,
        )
        logger.info("%s loaded", self.ENGINE_NAME)


@register_engine
class Qwen3TTSBaseEngine(Qwen3TTSBase):
    """Qwen3-TTS Base: 3-second rapid voice clone from reference audio."""

    ENGINE_NAME = "qwen3_tts_base"

    def load_sync(self) -> None:
        self._load_model()

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
            raise RuntimeError(f"{self.ENGINE_NAME} not loaded. Call load() first.")

        logger.info("Synthesizing with %s (voice clone): %s", self.ENGINE_NAME, text[:50])

        ref_audio = reference_audio or _DEFAULT_VOICE_WAV
        ref_text = _DEFAULT_VOICE_TEXT if not reference_audio else ""

        # generate_voice_clone returns (wavs: list[np.ndarray], sr: int)
        if ref_text:
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language="Auto",
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
        else:
            # No transcript: use x_vector_only_mode
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language="Auto",
                ref_audio=ref_audio,
                ref_text="",
                x_vector_only_mode=True,
            )

        return self._to_result(wavs[0], sr, sample_rate)

    @property
    def engine_name(self) -> str:
        return self.ENGINE_NAME


@register_engine
class Qwen3TTSCustomVoiceEngine(Qwen3TTSBase):
    """Qwen3-TTS CustomVoice: 9 built-in speakers with instruction control."""

    ENGINE_NAME = "qwen3_tts_customvoice"

    _SPEAKERS = [
        "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
        "Ryan", "Aiden", "Ono_Anna", "Sohee",
    ]

    def load_sync(self) -> None:
        self._load_model()

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
            raise RuntimeError(f"{self.ENGINE_NAME} not loaded. Call load() first.")

        # Map "default" to first speaker
        speaker = voice if voice in self._SPEAKERS else "Vivian"

        logger.info(
            "Synthesizing with %s (speaker=%s): %s",
            self.ENGINE_NAME, speaker, text[:50],
        )

        # generate_custom_voice returns (wavs: list[np.ndarray], sr: int)
        wavs, sr = self._model.generate_custom_voice(
            text=text,
            language="Auto",
            speaker=speaker,
        )

        return self._to_result(wavs[0], sr, sample_rate)

    @property
    def engine_name(self) -> str:
        return self.ENGINE_NAME

    @property
    def supported_voices(self) -> list[str]:
        return self._SPEAKERS


@register_engine
class Qwen3TTSVoiceDesignEngine(Qwen3TTSBase):
    """Qwen3-TTS VoiceDesign: design voice from natural language description."""

    ENGINE_NAME = "qwen3_tts_voicedesign"

    def load_sync(self) -> None:
        self._load_model()

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
            raise RuntimeError(f"{self.ENGINE_NAME} not loaded. Call load() first.")

        # voice param is used as the voice description (instruct)
        instruct = voice if voice != "default" else ""

        logger.info(
            "Synthesizing with %s (instruct=%s): %s",
            self.ENGINE_NAME, instruct[:30], text[:50],
        )

        # generate_voice_design returns (wavs: list[np.ndarray], sr: int)
        wavs, sr = self._model.generate_voice_design(
            text=text,
            language="Auto",
            instruct=instruct,
        )

        return self._to_result(wavs[0], sr, sample_rate)

    @property
    def engine_name(self) -> str:
        return self.ENGINE_NAME
