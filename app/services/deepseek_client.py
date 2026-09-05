import json
import logging
from typing import AsyncGenerator, List, Optional
from fastapi import HTTPException, status
import httpx

from app.core.config import settings
from app.core.credentials import credentials_manager
from app.schemas.chat import (
    DeepSeekChatRequest,
    DeepSeekChatResponse,
    ModelInfo,
    StreamChunk,
)
from app.core.pow_solver import pow_solver
from app.services.session_manager import session_manager
from app.services.sse_parser import parse_sse_lines, parse_sse_stream

logger = logging.getLogger(__name__)


AVAILABLE_MODELS = [
    ModelInfo(
        id="deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        description="Флагманская модель 1.6T MoE (49B active) для сложного программирования, математики и глубоких рассуждений.",
        model_type="expert",
        supports_thinking=True,
        supports_search=False,
    ),
    ModelInfo(
        id="deepseek-v4-flash",
        name="DeepSeek V4 Flash",
        description="Сверхбыстрая и эффективная модель 284B MoE (13B active) для быстрого чата и оперативных задач.",
        model_type="default",
        supports_thinking=False,
        supports_search=True,
    ),
    ModelInfo(
        id="deepseek-v4-flash-vision-exp",
        name="DeepSeek V4 Flash Vision",
        description="Мультимодальная модель DeepSeek V4 с визуальным пониманием графиков, кода и документов.",
        model_type="vision",
        supports_thinking=False,
        supports_search=True,
    ),
    ModelInfo(
        id="deepseek-reasoner",
        name="DeepSeek R1 (Reasoner)",
        description="Специализированная модель рассуждений DeepSeek-R1 с подробным пошаговым выводом мыслей.",
        model_type="expert",
        supports_thinking=True,
        supports_search=False,
    ),
    ModelInfo(
        id="deepseek-chat",
        name="DeepSeek V3 (Chat)",
        description="Быстрая языковая модель общего назначения DeepSeek V3 (режим expert).",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="deepseek-search",
        name="DeepSeek V3 (Web Search)",
        description="DeepSeek V3 с включенным веб-поиском по актуальной информации в реальном времени.",
        model_type="default",
        supports_thinking=False,
        supports_search=True,
    ),
]


class DeepSeekClient:
    """Клиент для взаимодействия с веб-интерфейсом DeepSeek (с поддержкой PoW и SSE-стриминга)."""

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.client = http_client or httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT)

    def get_models(self) -> List[ModelInfo]:
        return AVAILABLE_MODELS

    def resolve_model_params(
        self,
        model_name: str,
        thinking_enabled: Optional[bool] = None,
        search_enabled: Optional[bool] = None,
    ) -> tuple[str, bool, bool]:
        """Определяет внутренний model_type и флаги thinking / search."""
        model_lower = model_name.lower().strip()

        # 1. Модели семейства DeepSeek V4
        if model_lower in ["deepseek-v4-pro", "v4-pro", "v4", "deepseek-v4", "pro"]:
            model_type = "expert"
            think = True if thinking_enabled is None else thinking_enabled
            search = False
        elif model_lower in ["deepseek-v4-flash", "v4-flash", "flash"]:
            model_type = "default"
            think = True if thinking_enabled is None else thinking_enabled
            search = search_enabled if search_enabled is not None else False
        elif model_lower in ["deepseek-v4-flash-vision-exp", "v4-vision", "vision", "deepseek-vision"]:
            model_type = "vision"
            think = True if thinking_enabled is None else thinking_enabled
            search = search_enabled if search_enabled is not None else False

        # 2. Модели поиска
        elif search_enabled is True or model_lower in ["deepseek-search", "search"]:
            model_type = "default"
            think = True if thinking_enabled is None else thinking_enabled
            search = True

        # 3. Модели рассуждений (DeepSeek-R1)
        elif thinking_enabled is True or model_lower in ["deepseek-reasoner", "r1", "reasoner", "deepseek_reasoner"]:
            model_type = "expert"
            think = True
            search = False

        # 4. Базовый чат DeepSeek-V3
        else:
            model_type = "expert"
            think = True if thinking_enabled is None else thinking_enabled
            search = False

        return model_type, think, search

    async def stream_chat(
        self,
        request: DeepSeekChatRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """Выполняет стриминговый запрос к DeepSeek с автоматическим решением PoW."""
        if not credentials_manager.is_authenticated():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Учетные данные DeepSeek не настроены. Укажите токен через /api/v1/auth/token или файл credentials.json."
            )

        # 1. Получаем или создаем сессию
        session_id = await session_manager.get_or_create_session(self.client, request.chat_session_id)
        
        # 2. Определяем parent_message_id
        parent_msg_id = request.parent_message_id
        if parent_msg_id is None:
            parent_msg_id = session_manager.get_parent_message_id(session_id)

        # 3. Определяем параметры модели
        model_type, thinking_enabled, search_enabled = self.resolve_model_params(
            request.model, request.thinking_enabled, request.search_enabled
        )

        # 4. Решаем PoW challenge
        target_path = "/api/v0/chat/completion"
        try:
            pow_header = await pow_solver.get_pow_header(self.client, target_path)
        except Exception as e:
            logger.error(f"Ошибка вычисления PoW: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ошибка решения Proof-of-Work для DeepSeek: {str(e)}"
            )

        headers = {
            "accept": "*/*",
            "authorization": credentials_manager.auth_header,
            "content-type": "application/json",
            "sec-ch-ua": '"Not=A?Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "x-client-bundle-id": settings.CLIENT_BUNDLE_ID,
            "x-client-locale": settings.CLIENT_LOCALE,
            "x-client-platform": settings.CLIENT_PLATFORM,
            "x-client-timezone-offset": settings.CLIENT_TIMEZONE_OFFSET,
            "x-client-version": settings.CLIENT_VERSION,
            "x-ds-pow-response": pow_header,
            "referrer": f"{settings.DEEPSEEK_BASE_URL}/a/chat/s/{session_id}",
            "user-agent": settings.USER_AGENT,
        }

        payload = {
            "chat_session_id": session_id,
            "parent_message_id": parent_msg_id,
            "model_type": model_type,
            "prompt": request.prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "action": None,
            "preempt": False,
        }

        url = f"{settings.DEEPSEEK_BASE_URL}{target_path}"
        last_message_id: Optional[int] = None
        extracted_title: Optional[str] = None

        try:
            req = self.client.build_request("POST", url, json=payload, headers=headers, timeout=settings.REQUEST_TIMEOUT)
            resp = await self.client.send(req, stream=True)

            if resp.status_code != 200:
                body = await resp.aread()
                err_text = body.decode("utf-8", errors="replace")
                logger.error(f"DeepSeek returned status {resp.status_code}: {err_text}")
                if resp.status_code in [400, 404] or "session" in err_text.lower():
                    session_manager.invalidate_current_session()
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"DeepSeek API error ({resp.status_code}): {err_text}"
                )

            async for chunk in parse_sse_lines(resp.aiter_lines(), session_id=session_id):
                if chunk.type == "error":
                    if "session" in chunk.text.lower() or "not found" in chunk.text.lower():
                        session_manager.invalidate_current_session()
                if chunk.message_id is not None:
                    last_message_id = chunk.message_id
                if chunk.type == "title" and chunk.text:
                    extracted_title = chunk.text
                yield chunk

        finally:
            if last_message_id is not None:
                session_manager.update_session_state(session_id, last_message_id, extracted_title)

    async def send_message(self, request: DeepSeekChatRequest) -> DeepSeekChatResponse:
        """Синхронная обертка над stream_chat, возвращающая полный ответ целиком."""
        full_thinking = []
        full_content = []
        message_id = 0
        token_usage = None
        error_msg = None

        async for chunk in self.stream_chat(request):
            if chunk.session_id:
                session_id = chunk.session_id
            if chunk.message_id:
                message_id = chunk.message_id
            if chunk.token_usage:
                token_usage = chunk.token_usage

            if chunk.type == "error":
                error_msg = chunk.text
            elif chunk.type == "thinking":
                full_thinking.append(chunk.text)
            elif chunk.type == "content":
                full_content.append(chunk.text)

        if not full_content and error_msg:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS if ("частые" in error_msg.lower() or "too frequent" in error_msg.lower()) else status.HTTP_502_BAD_GATEWAY,
                detail=error_msg
            )

        return DeepSeekChatResponse(
            session_id=session_id,
            message_id=message_id,
            parent_message_id=request.parent_message_id,
            thinking="".join(full_thinking) if full_thinking else None,
            content="".join(full_content),
            token_usage=token_usage,
            status="FINISHED",
        )
