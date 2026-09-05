import asyncio
import datetime
import logging
import time
from typing import Any, Callable, Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()


class ProxyEventLogger:
    """代理模式下 AI Agent 请求的中央事件记录器与终端可视化器。"""

    def __init__(self):
        self.active_subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self._history: List[Dict[str, Any]] = []
        self.is_proxy_mode_active: bool = False

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        if callback not in self.active_subscribers:
            self.active_subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        if callback in self.active_subscribers:
            self.active_subscribers.remove(callback)

    def log_request_start(
        self,
        protocol: str,
        endpoint: str,
        model: str,
        provider_name: str,
        messages_count: int,
        estimated_tokens: int,
        tools_names: List[str],
        user_agent: str = "",
        client_ip: str = "127.0.0.1",
    ) -> str:
        req_id = f"req_{int(time.time() * 1000) % 1000000}"
        event = {
            "id": req_id,
            "type": "request_start",
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "protocol": protocol,
            "endpoint": endpoint,
            "model": model,
            "provider": provider_name,
            "messages_count": messages_count,
            "tokens": estimated_tokens,
            "tools": tools_names,
            "user_agent": user_agent or "Coding Agent",
            "client_ip": client_ip,
            "start_time": time.time(),
        }
        self._dispatch(event)
        return req_id

    def log_thinking_chunk(self, req_id: str, text: str):
        self._dispatch({"id": req_id, "type": "thinking_chunk", "text": text})

    def log_content_chunk(self, req_id: str, text: str):
        self._dispatch({"id": req_id, "type": "content_chunk", "text": text})

    def log_tool_call(self, req_id: str, tool_name: str, arguments_str: str):
        self._dispatch({
            "id": req_id,
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments_str,
        })

    def log_request_end(
        self,
        req_id: str,
        status_code: int = 200,
        tokens_out: int = 0,
        error: Optional[str] = None,
    ):
        event = {
            "id": req_id,
            "type": "request_end",
            "status_code": status_code,
            "tokens_out": tokens_out,
            "error": error,
            "end_time": time.time(),
        }
        self._dispatch(event)

    def _dispatch(self, event: Dict[str, Any]):
        for sub in self.active_subscribers:
            try:
                sub(event)
            except Exception as e:
                logger.debug(f"Proxy 订阅回调异常: {e}")


proxy_logger = ProxyEventLogger()
