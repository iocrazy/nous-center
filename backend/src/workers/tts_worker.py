import io
import logging
import struct
from pathlib import Path

from src.config import get_settings, load_model_configs
from src.storage.nas import StorageService
from src.workers.celery_app import celery_app

# Import engine modules to trigger registration
import src.workers.tts_engines.cosyvoice2  # noqa: F401
import src.workers.tts_engines.indextts2  # noqa: F401
import src.workers.tts_engines.qwen3_tts  # noqa: F401
import src.workers.tts_engines.moss_tts  # noqa: F401
from src.workers.tts_engines import get_engine

logger = logging.getLogger(__name__)


def _resolve_model_path(engine_name: str) -> str:
    """Resolve the full local path for a TTS model."""
    settings = get_settings()
    configs = load_model_configs()
    config = configs.get(engine_name)
    if config is None:
        raise ValueError(f"No model config found for engine: {engine_name}")

    local_path = config.get("local_path")
    if local_path is None:
        raise ValueError(f"No local_path configured for engine: {engine_name}")

    return str(Path(settings.LOCAL_MODELS_PATH) / local_path)


def _encode_wav(audio_bytes: bytes, sample_rate: int, num_channels: int = 1,
                sample_width: int = 2) -> bytes:
    """Encode raw PCM audio bytes into WAV format."""
    buf = io.BytesIO()
    data_size = len(audio_bytes)
    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * num_channels * sample_width))
    buf.write(struct.pack("<H", num_channels * sample_width))
    buf.write(struct.pack("<H", sample_width * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio_bytes)
    return buf.getvalue()


@celery_app.task(bind=True, name="src.workers.tts_worker.generate_tts_task")
def generate_tts_task(self, task_id: str, params: dict):
    """Generate speech using the specified TTS engine."""
    self.update_state(state="RUNNING", meta={"progress": 0})

    engine_name = params.get("engine", "cosyvoice2")
    text = params["text"]
    voice = params.get("voice", "default")
    speed = params.get("speed", 1.0)
    sample_rate = params.get("sample_rate", 24000)
    reference_audio = params.get("reference_audio")

    try:
        # 1. Get or create engine, load model if needed
        self.update_state(state="RUNNING", meta={"progress": 10, "step": "loading_model"})
        model_path = _resolve_model_path(engine_name)
        engine = get_engine(engine_name, model_path=model_path, device="cuda")

        if not engine.is_loaded:
            logger.info("Loading TTS engine: %s", engine_name)
            engine.load()

        # 2. Run inference
        self.update_state(state="RUNNING", meta={"progress": 30, "step": "synthesizing"})
        result = engine.synthesize(
            text=text,
            voice=voice,
            speed=speed,
            sample_rate=sample_rate,
            reference_audio=reference_audio,
        )

        # 3. Save audio file
        self.update_state(state="RUNNING", meta={"progress": 80, "step": "saving"})
        storage = StorageService()

        if result.audio_bytes:
            # Wrap in WAV if raw PCM
            if result.format == "wav" and not result.audio_bytes[:4] == b"RIFF":
                wav_data = _encode_wav(result.audio_bytes, result.sample_rate)
            else:
                wav_data = result.audio_bytes

            file_path = storage.save(wav_data, task_id, "output.wav")
        else:
            file_path = ""

        self.update_state(state="RUNNING", meta={"progress": 100, "step": "done"})

        return {
            "task_id": task_id,
            "status": "completed",
            "file": file_path,
            "engine": engine_name,
            "duration_seconds": result.duration_seconds,
            "sample_rate": result.sample_rate,
        }

    except Exception as e:
        logger.exception("TTS generation failed for task %s", task_id)
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "engine": engine_name,
        }
