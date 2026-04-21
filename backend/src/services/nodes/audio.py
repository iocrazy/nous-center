"""Audio-related invokable nodes: ref audio + TTS engine.

Migrated from workflow_executor._exec_ref_audio / _exec_tts_engine as part of
Wave 1 Task 4.4. Bodies are copied verbatim — no refactor.
"""

from __future__ import annotations

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
        """Call TTS engine via ModelManager."""
        import asyncio
        import base64

        # Import lazily to avoid circular imports and to read the current
        # _model_manager global at call time (matches legacy _exec_tts_engine).
        from src.services import workflow_executor as we

        text = inputs.get("text", "")
        if not text:
            raise we.ExecutionError("TTS 节点缺少文本输入")

        engine_name = data.get("engine", "cosyvoice2")

        if we._model_manager is None:
            raise we.ExecutionError("ModelManager 未初始化")

        adapter = we._model_manager.get_adapter(engine_name)
        if adapter is None or not adapter.is_loaded:
            raise we.ExecutionError(
                f"引擎 {engine_name} 未加载，请先通过管理 API 加载"
            )

        kwargs = {
            "text": text,
            "voice": data.get("voice", "default"),
            "speed": data.get("speed", 1.0),
            "sample_rate": data.get("sample_rate", 24000),
        }

        result = await asyncio.to_thread(adapter.synthesize, **kwargs)
        audio_b64 = base64.b64encode(result.audio_bytes).decode()
        return {
            "audio": audio_b64,
            "sample_rate": result.sample_rate,
            "duration_seconds": result.duration_seconds,
            "format": result.format,
        }
