"""WS 广播基础设施已下沉到 src/services/ws_hub.py(打破 services→api 反向依赖)。

本模块保留为 re-export shim:大量路由/测试仍 `from src.api.websocket import
ws_manager`,一并兼容,零改动。新代码请直接从 src.services.ws_hub import。
"""
from src.services.ws_hub import (  # noqa: F401
    ConnectionManager,
    _ws_connections,
    ws_manager,
)
