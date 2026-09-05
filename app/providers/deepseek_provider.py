from typing import AsyncGenerator, List, Optional
import httpx
from app.core.credentials import credentials_manager
from app.providers.base import BaseLLMProvider
from app.schemas.chat import (
    DeepSeekChatRequest,
    DeepSeekChatResponse,
    ModelInfo,
    StreamChunk,
)
from app.services.deepseek_client import AVAILABLE_MODELS, DeepSeekClient


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek Web API (chat.deepseek.com) 提供商。"""

    def __init__(self, http_client: httpx.AsyncClient):
        super().__init__(provider_id="deepseek", display_name="DeepSeek", http_client=http_client)
        self.client_impl = DeepSeekClient(http_client)

    def get_models(self) -> List[ModelInfo]:
        return AVAILABLE_MODELS

    def is_authenticated(self) -> bool:
        return credentials_manager.is_authenticated("deepseek")

    def get_current_session_id(self) -> Optional[str]:
        from app.services.session_manager import session_manager
        return session_manager.get_current_session_id()

    def set_session_id(self, session_id: str) -> None:
        from app.services.session_manager import session_manager
        session_manager.set_current_session_id(session_id)

    def reset_session(self) -> None:
        from app.services.session_manager import session_manager
        session_manager.reset_context()

    async def stream_chat(
        self,
        request: DeepSeekChatRequest,
    ) -> AsyncGenerator[StreamChunk, None]:
        async for chunk in self.client_impl.stream_chat(request):
            yield chunk

    async def send_message(
        self,
        request: DeepSeekChatRequest,
    ) -> DeepSeekChatResponse:
        return await self.client_impl.send_message(request)
