from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional, Dict, Any
import httpx
from app.schemas.chat import (
    DeepSeekChatRequest,
    DeepSeekChatResponse,
    ModelInfo,
    StreamChunk,
)


class BaseLLMProvider(ABC):
    """所有 LLM 提供商 (DeepSeek, Qwen, GLM) 的抽象基类。"""

    def __init__(self, provider_id: str, display_name: str, http_client: httpx.AsyncClient):
        self.provider_id = provider_id
        self.display_name = display_name
        self.http_client = http_client

    @abstractmethod
    def get_models(self) -> List[ModelInfo]:
        """返回提供商支持的模型列表。"""
        pass

    @abstractmethod
    def is_authenticated(self) -> bool:
        """检查该提供商的认证凭证是否存在。"""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        request: DeepSeekChatRequest,
    ) -> AsyncGenerator[StreamChunk, None]:
        """流式调用接口 (生成 StreamChunk 对象)。"""
        pass

    @abstractmethod
    async def send_message(
        self,
        request: DeepSeekChatRequest,
    ) -> DeepSeekChatResponse:
        """同步非流式调用接口。"""
        pass

    @abstractmethod
    def get_current_session_id(self) -> Optional[str]:
        """获取当前活跃会话 ID。"""
        pass

    @abstractmethod
    def set_session_id(self, session_id: str) -> None:
        """设置当前活跃会话 ID。"""
        pass

    @abstractmethod
    def reset_session(self) -> None:
        """重置当前会话上下文。"""
        pass

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """获取服务端历史会话列表。"""
        return []
