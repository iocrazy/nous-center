import json

from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    def __init__(self):
        # {task_id: [websocket, ...]}
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)

    def disconnect(self, task_id: str, websocket: WebSocket):
        if task_id in self.active_connections:
            self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]

    async def send_update(self, task_id: str, data: dict):
        if task_id in self.active_connections:
            message = json.dumps(data)
            for ws in self.active_connections[task_id]:
                await ws.send_text(message)


ws_manager = ConnectionManager()
