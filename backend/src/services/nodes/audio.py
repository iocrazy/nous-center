"""Audio nodes: ref audio + TTS engine.

TTSEngineNode calls the v2 InferenceAdapter (TTSEngine subclass) via
adapter.infer(AudioRequest). adapter.infer wraps the per-engine sync
synthesize() in asyncio.to_thread and returns a unified InferenceResult.
"""

from __future__ import annotations

import base64
import uuid

from src.services.inference.base import AudioRequest
from src.services.nodes.registry import register


@register("ref_audio")
class RefAudioNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        return {
            "audio_path": data.get("path", ""),
            "audio": data.get("audio_data", ""),  # base64 data URL for LLM audio input
            "ref_text": data.get("ref_text", ""),
        }


@register("tts_engine")
class TTSEngineNode:
    async def invoke(self, data: dict, inputs: dict) -> dict:
        from src.services import workflow_executor as we

        text = inputs.get("text", "")
        if not text:
            raise we.ExecutionError("TTS 节点缺少文本输入")

        engine_name = data.get("engine", "cosyvoice2")

        if we._model_manager is None:
            raise we.ExecutionError("ModelManager 未初始化")

        adapter = await we._model_manager.get_loaded_adapter(engine_name)

        req = AudioRequest(
            request_id=str(uuid.uuid4()),
            text=text,
            voice=data.get("voice", "default"),
            speed=float(data.get("speed", 1.0)),
            sample_rate=int(data.get("sample_rate", 24000)),
        )
        result = await adapter.infer(req)
        meta = result.metadata
        return {
            "audio": base64.b64encode(result.data).decode(),
            "sample_rate": meta.get("sample_rate"),
            "duration_seconds": meta.get("duration_seconds"),
            "format": meta.get("format"),
        }
