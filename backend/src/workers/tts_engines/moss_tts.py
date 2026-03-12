import io
import logging

import torch

from src.workers.tts_engines.base import TTSEngine, TTSResult
from src.workers.tts_engines.registry import register_engine

logger = logging.getLogger(__name__)

# Default reference audio for voice cloning
_DEFAULT_VOICE_WAV = "assets/voices/default_zh_female.wav"


@register_engine
class MOSSTTSEngine(TTSEngine):
    """MOSS-TTS (MossTTSDelay-8B): production-grade zero-shot voice cloning.

    Uses transformers AutoModel/AutoProcessor with trust_remote_code=True.
    NOTE: Requires transformers >= 5.0.0 and the moss-tts package, which
    conflicts with CosyVoice2/IndexTTS (transformers 4.52.x). Run in a
    separate worker environment if needed.
    """

    ENGINE_NAME = "moss_tts"

    def load(self) -> None:
        logger.info("Loading MOSS-TTS from %s", self.model_path)
        from transformers import AutoModel, AutoProcessor

        # Disable broken cuDNN SDPA backend
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Resolve attention implementation
        attn_impl = "sdpa"
        try:
            import importlib.util
            if (
                self.device.startswith("cuda")
                and importlib.util.find_spec("flash_attn") is not None
                and dtype in {torch.float16, torch.bfloat16}
            ):
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    attn_impl = "flash_attention_2"
        except Exception:
            pass

        self._processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        self._processor.audio_tokenizer = self._processor.audio_tokenizer.to(self.device)

        self._model = AutoModel.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
        ).to(self.device)
        self._model.eval()

        logger.info(
            "MOSS-TTS loaded (attn=%s, sr=%d)",
            attn_impl, self._processor.model_config.sampling_rate,
        )

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        sample_rate: int = 24000,
        reference_audio: str | None = None,
    ) -> TTSResult:
        if not self.is_loaded:
            raise RuntimeError("MOSS-TTS not loaded. Call load() first.")

        logger.info("Synthesizing with MOSS-TTS: %s", text[:50])

        # Build conversation with optional voice cloning reference
        ref_audio = reference_audio or _DEFAULT_VOICE_WAV
        if reference_audio or voice != "default":
            conversation = [self._processor.build_user_message(
                text=text,
                reference=[ref_audio],
            )]
        else:
            conversation = [self._processor.build_user_message(text=text)]

        batch = self._processor([conversation], mode="generation")
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4096,
                # Recommended hyperparameters for MossTTSDelay-8B
                audio_temperature=1.7,
                audio_top_p=0.8,
                audio_top_k=25,
                audio_repetition_penalty=1.0,
            )

        messages = self._processor.decode(outputs)
        audio_tensor = messages[0].audio_codes_list[0]  # 1-D torch tensor

        native_sr = self._processor.model_config.sampling_rate

        # Convert to numpy
        audio_np = audio_tensor.cpu().float().numpy()

        # Resample if needed
        if sample_rate != native_sr:
            import torchaudio
            tensor = torch.from_numpy(audio_np).unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, native_sr, sample_rate)
            audio_np = tensor.squeeze(0).numpy()

        # Write WAV bytes
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")

        duration = round(len(audio_np) / sample_rate, 2)

        return TTSResult(
            audio_bytes=buf.getvalue(),
            sample_rate=sample_rate,
            duration_seconds=duration,
            format="wav",
        )

    @property
    def engine_name(self) -> str:
        return self.ENGINE_NAME
