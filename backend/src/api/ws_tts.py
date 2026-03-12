"""WebSocket TTS session handler with connection-level reuse."""

import base64
import json
import time

from fastapi import WebSocket, WebSocketDisconnect

from src.api.routes.tts import _get_loaded_engine


async def handle_tts_websocket(websocket: WebSocket):
    await websocket.accept()
    active_session: str | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")
            session_id = msg.get("session_id", "")

            if msg_type == "start_session":
                active_session = session_id
                engine_name = msg.get("engine", "")
                engine = _get_loaded_engine(engine_name)
                if engine is None:
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "ENGINE_NOT_LOADED",
                        "message": f"Engine {engine_name} not loaded",
                    })
                    active_session = None
                else:
                    # Store engine ref for this session
                    websocket.state.engine = engine
                    websocket.state.session_config = msg
                    await websocket.send_json({
                        "type": "session_started",
                        "session_id": session_id,
                    })

            elif msg_type == "synthesize":
                if active_session != session_id:
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "NO_ACTIVE_SESSION",
                        "message": "No active session for this session_id",
                    })
                    continue

                engine = getattr(websocket.state, "engine", None)
                if engine is None:
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "ENGINE_NOT_LOADED",
                        "message": "No engine loaded for session",
                    })
                    continue

                try:
                    text = msg.get("text", "")
                    config = getattr(websocket.state, "session_config", {})
                    start = time.monotonic()

                    result = engine.synthesize(
                        text=text,
                        voice=config.get("voice", "default"),
                        speed=config.get("speed", 1.0),
                        sample_rate=config.get("sample_rate", 24000),
                        reference_audio=config.get("reference_audio"),
                        reference_text=config.get("reference_text"),
                        emotion=config.get("emotion") or msg.get("emotion"),
                    )
                    elapsed = time.monotonic() - start

                    audio_b64 = base64.b64encode(result.audio_bytes).decode()
                    seq = getattr(websocket.state, "seq", 0) + 1
                    websocket.state.seq = seq

                    await websocket.send_json({
                        "type": "audio",
                        "session_id": session_id,
                        "seq": seq,
                        "audio": audio_b64,
                        "format": result.format,
                        "duration_ms": int(result.duration_seconds * 1000),
                        "rtf": round(elapsed / max(result.duration_seconds, 0.01), 4),
                    })
                except Exception as exc:
                    await websocket.send_json({
                        "type": "error",
                        "session_id": session_id,
                        "code": "SYNTHESIS_FAILED",
                        "message": str(exc),
                    })

            elif msg_type == "end_session":
                active_session = None
                websocket.state.engine = None
                websocket.state.seq = 0
                await websocket.send_json({
                    "type": "session_ended",
                    "session_id": session_id,
                })

    except WebSocketDisconnect:
        pass
