from starlette.testclient import TestClient

from src.api.main import create_app


def test_ws_tts_endpoint_accepts_connection():
    """WS /ws/tts should accept connections and handle start_session."""
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "start_session", "session_id": "s1", "engine": "cosyvoice2"})
        resp = ws.receive_json()
        # Engine not loaded → error
        assert resp["type"] == "error"
        assert resp["session_id"] == "s1"
