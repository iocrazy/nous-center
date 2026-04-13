"""VoxCPM2 TTS Engine — 30-language, voice design & cloning, 48kHz output."""

import io
import logging
import wave

import numpy as np

from src.workers.tts_engines.base import TTSEngine, TTSResult

logger = logging.getLogger(__name__)


class VoxCPM2Engine(TTSEngine):
    model_type = "tts"
    estimated_vram_mb = 8000

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        super().__init__(model_path=model_path, device=device)
        self._voxcpm = None

    @property
    def engine_name(self) -> str:
        return "voxcpm2"

    def load_sync(self) -> None:
        from voxcpm import VoxCPM
        from src.config import get_settings
        import os

        settings = get_settings()
        full_path = os.path.join(settings.LOCAL_MODELS_PATH, self.model_path)
        if not os.path.exists(full_path):
            full_path = "openbmb/VoxCPM2"

        logger.info("Loading VoxCPM2 from %s", full_path)
        self._voxcpm = VoxCPM.from_pretrained(full_path, load_denoiser=False)
        self._model = True
        logger.info("VoxCPM2 loaded (sample_rate=%d)", self._voxcpm.tts_model.sample_rate)

    def unload(self) -> None:
        self._voxcpm = None
        self._model = None

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        sample_rate: int = 48000,
        reference_audio: str | None = None,
        reference_text: str | None = None,
        emotion: str | None = None,
    ) -> TTSResult:
        if self._voxcpm is None:
            raise RuntimeError("VoxCPM2 model not loaded")

        import time
        start = time.monotonic()

        kwargs = {
            "text": text,
            "cfg_value": 2.0,
            "inference_timesteps": 10,
        }

        # Voice design: use emotion as voice description
        if emotion:
            kwargs["text"] = f"({emotion}){text}"

        # Voice cloning: use reference audio
        if reference_audio:
            import tempfile, base64, os
            # Save reference audio to temp file
            if reference_audio.startswith("data:"):
                _, b64 = reference_audio.split(",", 1)
                audio_bytes = base64.b64decode(b64)
            else:
                with open(reference_audio, "rb") as f:
                    audio_bytes = f.read()

            fd, ref_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(audio_bytes)

            try:
                if reference_text:
                    # Ultimate cloning
                    kwargs["prompt_wav_path"] = ref_path
                    kwargs["prompt_text"] = reference_text
                    kwargs["reference_wav_path"] = ref_path
                else:
                    # Controllable cloning
                    kwargs["reference_wav_path"] = ref_path

                wav = self._voxcpm.generate(**kwargs)
            finally:
                os.unlink(ref_path)
        else:
            wav = self._voxcpm.generate(**kwargs)

        sr = self._voxcpm.tts_model.sample_rate
        elapsed = time.monotonic() - start

        # Convert to WAV bytes
        if wav.ndim > 1:
            wav = np.mean(wav, axis=0)
        wav = wav.astype(np.float32)
        pcm = np.clip(wav * 32767, -32768, 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

        duration = len(wav) / sr

        return TTSResult(
            audio_bytes=buf.getvalue(),
            sample_rate=sr,
            duration_seconds=duration,
            format="wav",
        )

    @property
    def supported_voices(self) -> list[str]:
        return ["default"]
