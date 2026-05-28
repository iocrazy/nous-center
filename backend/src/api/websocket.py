import json

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        # {task_id: [websocket, ...]}
        self.active_connections: dict[str, list[WebSocket]] = {}
        # Global task list subscribers
        self._global_subscribers: list[WebSocket] = []
        # Model status subscribers
        self._model_subscribers: list[WebSocket] = []

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

    # --- Global task list ---

    async def subscribe_global(self, websocket: WebSocket):
        await websocket.accept()
        self._global_subscribers.append(websocket)

    def unsubscribe_global(self, websocket: WebSocket):
        if websocket in self._global_subscribers:
            self._global_subscribers.remove(websocket)

    async def broadcast_task_update(self, event: str, task_data: dict):
        """Broadcast task list change to all global subscribers."""
        if not self._global_subscribers:
            return
        message = json.dumps({"event": event, "task": task_data})
        dead: list[WebSocket] = []
        for ws in self._global_subscribers:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._global_subscribers.remove(ws)

    async def broadcast_task_progress(self, task_id: int, payload: dict) -> None:
        """PR-6(2026-05-28 任务面板重置全局 L3 progress):widget 在 ActiveTaskRow 显示
        「⚡ dit step 27/50 · 240ms · ETA 5.5s」需要的 node_progress payload,带 task_id
        路由让前端按 task 区分。WS 单一连接收所有 task 的 progress —— 多任务并发场景每个
        ActiveTaskRow 都能拿到自己的 L3 数据,不需要 per-task WS。
        Payload 形态:{event: "progress", task_id, ...progress_fields}。
        progress_fields = node_progress event payload(stage/step/total_steps/eta_ms/...)。"""
        if not self._global_subscribers:
            return
        message = json.dumps({
            "event": "progress",
            "task_id": str(task_id),
            **payload,
        })
        dead: list[WebSocket] = []
        for ws in self._global_subscribers:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._global_subscribers.remove(ws)

    # --- Model status ---

    async def subscribe_models(self, websocket: WebSocket):
        await websocket.accept()
        self._model_subscribers.append(websocket)

    def unsubscribe_models(self, websocket: WebSocket):
        if websocket in self._model_subscribers:
            self._model_subscribers.remove(websocket)

    async def broadcast_model_status(self, model_id: str, status: str, detail: str = ""):
        """Broadcast model loading status to all model subscribers."""
        if not self._model_subscribers:
            return
        message = json.dumps({
            "event": "model_status",
            "model": model_id,
            "status": status,
            "detail": detail,
        })
        dead: list[WebSocket] = []
        for ws in self._model_subscribers:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._model_subscribers.remove(ws)

    async def broadcast_component_state(self, component_key: str, state: str, error: str | None = None) -> None:
        """Push a component load-state change to /ws/models subscribers (spec §6.3)."""
        if not self._model_subscribers:
            return
        message = json.dumps({
            "event": "component_state_changed",
            "component_key": component_key,
            "state": state,
            "error": error,
        })
        dead: list[WebSocket] = []
        for ws in self._model_subscribers:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._model_subscribers.remove(ws)


ws_manager = ConnectionManager()
