import datetime
import json
import logging
import re
import time
from typing import AsyncGenerator, List, Optional
import uuid
import httpx
from fastapi import HTTPException, status

from app.core.credentials import credentials_manager
from app.providers.base import BaseLLMProvider
from app.schemas.chat import (
    DeepSeekChatRequest,
    DeepSeekChatResponse,
    ModelInfo,
    StreamChunk,
)

logger = logging.getLogger(__name__)

QWEN_MODELS = [
    ModelInfo(
        id="qwen3.7-plus",
        name="Qwen 3.7 Plus",
        description="Актуальная веб-модель Qwen 3.7 Plus с режимом глубоких рассуждений (Thinking).",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-3.8",
        name="Qwen 3.8",
        description="Флагманская модель Qwen 3-го поколения с глубоким пониманием и рассуждениями.",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-3.8-coder",
        name="Qwen 3.8 Coder",
        description="Передовая специализированная модель для сложного программирования, рефакторинга и агентных пайплайнов.",
        model_type="expert",
        supports_thinking=True,
        supports_search=False,
    ),
    ModelInfo(
        id="qwen-3-max",
        name="Qwen 3 Max",
        description="Максимальная по интеллектуальной мощности модель линейки Qwen 3.",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-3-plus",
        name="Qwen 3 Plus",
        description="Сбалансированная и быстрая модель Qwen 3 общего назначения.",
        model_type="default",
        supports_thinking=False,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-3-flash",
        name="Qwen 3 Flash",
        description="Сверхбыстрая легковесная модель для мгновенных ответов.",
        model_type="default",
        supports_thinking=False,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-2.5-coder-32b",
        name="Qwen 2.5 Coder 32B",
        description="Классическая кодовая модель Qwen 2.5 Coder.",
        model_type="expert",
        supports_thinking=False,
        supports_search=False,
    ),
]


class QwenProvider(BaseLLMProvider):
    """
    Провайдер для прямого веб-API chat.qwen.ai (/api/v2/chat/completions).
    На 100% воспроизводит протокол v2.1, заголовки браузера, куки и автосоздание чат-сессий (/api/v2/chats/new).
    """

    def __init__(self, http_client: httpx.AsyncClient):
        super().__init__(provider_id="qwen", display_name="Qwen (Alibaba)", http_client=http_client)
        self.base_url = "https://chat.qwen.ai"
        self._current_chat_id: Optional[str] = None

    def get_models(self) -> List[ModelInfo]:
        return QWEN_MODELS

    def is_authenticated(self) -> bool:
        return credentials_manager.is_authenticated("qwen")

    def get_current_session_id(self) -> Optional[str]:
        return self._current_chat_id

    def set_session_id(self, session_id: str) -> None:
        self._current_chat_id = session_id
        from app.services.session_manager import session_manager
        if session_manager.single_session_mode:
            session_manager.set_provider_session("qwen", session_id)

    def reset_session(self) -> None:
        self._current_chat_id = None
        from app.services.session_manager import session_manager
        session_manager.clear_provider_session("qwen")

    async def list_sessions(self) -> List[dict]:
        """Возвращает список существующих чатов с сервера Qwen."""
        token = credentials_manager.get_token("qwen")
        if not token:
            return []
        headers = self._build_headers(token, "")
        url = f"{self.base_url}/api/v2/chats"
        try:
            resp = await self.client.get(url, headers=headers, timeout=20.0)
            if resp.status_code == 200:
                data = resp.json() or {}
                items = data.get("data", [])
                results = []
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            results.append({
                                "id": it.get("id"),
                                "title": it.get("title") or "Без названия",
                                "created_at": it.get("created_at"),
                                "updated_at": it.get("updated_at"),
                                "provider": "qwen"
                            })
                return results
        except Exception as e:
            logger.warning(f"Ошибка получения списка сессий Qwen: {e}")
        return []

    def _resolve_qwen_model(self, requested_model: str) -> str:
        req_lower = requested_model.lower().strip()
        if req_lower in ["qwen-3.8-coder", "3.8-coder", "qwen-coder", "coder"]:
            return "qwen3.8-max"
        if req_lower in ["qwen-3.8", "3.8", "qwen3", "qwen3.8-max", "3.8-max"]:
            return "qwen3.8-max"
        if req_lower in ["qwen3.7-plus", "3.7-plus", "qwen-3.7", "3.7", "qwen", ""]:
            return "qwen3.7-plus"
        if req_lower in ["qwen3.7-max", "3.7-max"]:
            return "qwen3.7-max"
        if req_lower in ["qwen-3-max", "max", "qwen-max"]:
            return "qwen3.8-max"
        if req_lower in ["qwen-3-flash", "flash", "qwen-flash"]:
            return "qwen3.7-plus"
        if req_lower in ["qwen-3-plus", "plus"]:
            return "qwen3.7-plus"
        return requested_model

    def _build_headers(self, token_or_cookie: str, chat_id: str = "", thinking_enabled: bool = True) -> dict:
        """Формирует заголовки браузера для chat.qwen.ai."""
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%a %b %d %Y %H:%M:%S GMT+0000")
        req_id = str(uuid.uuid4())

        think_mode = "Thinking" if thinking_enabled else "Normal"

        if "token=" in token_or_cookie or "; " in token_or_cookie or "_bl_uid=" in token_or_cookie:
            cookie_header = token_or_cookie.strip()
            if "qwen-thinking_mode=" in cookie_header:
                cookie_header = re.sub(r'qwen-thinking_mode=[^;]+', f'qwen-thinking_mode={think_mode}', cookie_header)
            else:
                cookie_header = f"{cookie_header.rstrip(';')}; qwen-thinking_mode={think_mode};"
            jwt_token = ""
            for part in token_or_cookie.split(";"):
                part = part.strip()
                if part.startswith("token="):
                    jwt_token = part[6:].strip()
        else:
            jwt_token = token_or_cookie.strip()
            cookie_header = f"token={jwt_token}; qwen-thinking_mode={think_mode}; qwen-locale=zh-CN; qwen-theme=dark;"

        referer = f"https://chat.qwen.ai/c/{chat_id}" if chat_id else "https://chat.qwen.ai/"

        headers = {
            "Accept": "application/json, text/event-stream",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "Cookie": cookie_header,
            "Host": "chat.qwen.ai",
            "Origin": "https://chat.qwen.ai",
            "Referer": referer,
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Timezone": now_str,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
            "Version": "0.2.89",
            "X-Accel-Buffering": "no",
            "X-Request-Id": req_id,
            "bx-v": "2.5.37",
            "sec-ch-ua": '"Not=A?Brand";v="99", "Google Chrome";v="151", "Chromium";v="151"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "source": "web",
        }

        if jwt_token:
            headers["Authorization"] = f"Bearer {jwt_token}"

        return headers

    async def _create_new_chat(self) -> str:
        """Создает новую сессию через POST /api/v2/chats/new."""
        token = credentials_manager.get_token("qwen")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Учетные данные Qwen не настроены. Укажите токен или Cookie через команду /token qwen <токен_или_куки>."
            )

        headers = self._build_headers(token, "")
        url = f"{self.base_url}/api/v2/chats/new"

        try:
            resp = await self.client.post(url, json={"title": "New Chat", "models": ["qwen3.7-plus"]}, headers=headers, timeout=20.0)
            if resp.status_code == 200:
                data = resp.json() or {}
                if data.get("success") and "data" in data and isinstance(data["data"], dict):
                    new_id = data["data"].get("id")
                    if new_id:
                        self._current_chat_id = new_id
                        logger.info(f"Создан новый чат Qwen: {new_id}")
                        return new_id
        except Exception as e:
            logger.warning(f"Ошибка при создании чата Qwen через /chats/new: {e}")

        # Резервный fallback: получаем список существующих чатов
        try:
            resp = await self.client.get(f"{self.base_url}/api/v2/chats", headers=headers, timeout=20.0)
            if resp.status_code == 200:
                data = resp.json() or {}
                chat_list = data.get("data", [])
                if chat_list and isinstance(chat_list, list) and isinstance(chat_list[0], dict):
                    found_id = chat_list[0].get("id")
                    if found_id:
                        self._current_chat_id = found_id
                        return found_id
        except Exception:
            pass

        generated_id = str(uuid.uuid4())
        self._current_chat_id = generated_id
        return generated_id

    async def get_or_create_chat(self, chat_id: Optional[str] = None) -> str:
        """
        Получает существующий chat_id, либо использует единый постоянный чат (в режиме single_session_mode),
        либо создает новую сессию через POST /api/v2/chats/new.
        
        В single_session_mode сессия хранится в session_manager per-provider, 
        поэтому при смене провайдеров сессия Qwen восстанавливается.
        """
        from app.services.session_manager import session_manager

        if chat_id:
            self._current_chat_id = chat_id
            # Сохраняем в per-provider хранилище
            if session_manager.single_session_mode:
                session_manager.set_provider_session("qwen", chat_id)
            return chat_id

        if session_manager.single_session_mode:
            # В single-режиме: берём локальный _current_chat_id (если есть),
            # иначе пробуем восстановить из session_manager (сессия другой вкладки/перезапуска)
            existing = self._current_chat_id or session_manager.get_provider_session("qwen")
            if existing:
                self._current_chat_id = existing
                # Синхронизируем в session_manager чтобы следующие провайдеры тоже видели
                session_manager.set_provider_session("qwen", existing)
                logger.debug(f"Переиспользование текущего чата Qwen (Single-Session): {existing}")
                return existing

        new_id = await self._create_new_chat()
        if session_manager.single_session_mode and new_id:
            session_manager.set_provider_session("qwen", new_id)
        return new_id

    def _build_payload(self, prompt: str, model: str, chat_id: str, thinking_enabled: bool, search_enabled: bool) -> dict:
        """Формирует точный JSON payload протокола v2.1 chat.qwen.ai."""
        now_ts = int(time.time())
        fid = str(uuid.uuid4())
        child_id = str(uuid.uuid4())
        think_mode = "Thinking" if thinking_enabled else "Normal"

        return {
            "stream": True,
            "version": "2.1",
            "incremental_output": True,
            "chatId": chat_id,
            "parentId": "",
            "chat_id": chat_id,
            "chat_mode": "normal",
            "model": model,
            "parent_id": None,
            "messages": [
                {
                    "id": None,
                    "fid": fid,
                    "parentId": None,
                    "childrenIds": [child_id],
                    "role": "user",
                    "content": prompt,
                    "user_action": "chat",
                    "files": [],
                    "timestamp": now_ts - 2,
                    "models": [model],
                    "model": "",
                    "chat_type": "t2t",
                    "feature_config": {
                        "thinking_enabled": thinking_enabled,
                        "output_schema": "phase",
                        "research_mode": "normal",
                        "auto_thinking": False,
                        "thinking_mode": think_mode,
                        "thinking_format": "summary",
                        "auto_search": search_enabled,
                    },
                    "extra": {
                        "meta": {
                            "subChatType": "t2t"
                        }
                    },
                    "sub_chat_type": "t2t",
                    "parent_id": None,
                }
            ],
            "timestamp": now_ts,
        }

    async def stream_chat(
        self,
        request: DeepSeekChatRequest,
    ) -> AsyncGenerator[StreamChunk, None]:
        token = credentials_manager.get_token("qwen")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Учетные данные Qwen не настроены. Укажите токен или Cookie через команду /token qwen <токен_или_куки>."
            )

        from app.services.context_compressor import context_compressor, estimate_tokens
        if request.prompt:
            request.prompt = context_compressor.compress_raw_prompt(
                request.prompt, max_tokens=context_compressor.QWEN_MAX_WEB_TOKENS
            )

        chat_id = await self.get_or_create_chat(request.chat_session_id)
        resolved_model = self._resolve_qwen_model(request.model)
        # Default to False (fast mode) when not explicitly set - user controls this via toggle
        thinking_enabled = request.thinking_enabled if request.thinking_enabled is not None else False
        search_enabled = request.search_enabled if request.search_enabled is not None else False

        headers = self._build_headers(token, chat_id, thinking_enabled)
        payload = self._build_payload(request.prompt, resolved_model, chat_id, thinking_enabled, search_enabled)

        logger.info(f"Отправка запроса в Qwen API (chat_id: {chat_id}, модель: {resolved_model}, промпт: ~{estimate_tokens(request.prompt):,} токенов)")

        # Отправляем начальный чанк с session_id
        yield StreamChunk(type="session", text=chat_id, session_id=chat_id)

        url = f"{self.base_url}/api/v2/chat/completions?chat_id={chat_id}"

        try:
            req = self.client.build_request("POST", url, json=payload, headers=headers, timeout=180.0)
            resp = await self.client.send(req, stream=True)

            if resp.status_code != 200:
                body = await resp.aread()
                err_text = body.decode("utf-8", errors="replace")
                logger.error(f"Qwen HTTP {resp.status_code} error: {err_text}")
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Qwen API error ({resp.status_code}): {err_text}"
                )

            last_thought_len = 0
            token_usage = None
            received_chunks_count = 0

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                # 1. Проверка на ошибки Alibaba WAF / Капчу (ответ не в формате SSE)
                if line.startswith("{"):
                    try:
                        err_json = json.loads(line)
                        ret_list = err_json.get("ret", [])
                        ret_str = str(ret_list)
                        if "FAIL_SYS_USER_VALIDATE" in ret_str or "RGV587_ERROR" in ret_str or "punish" in str(err_json):
                            logger.error(f"❌ Alibaba Cloud WAF заблокировал запрос (капча / rate limit): {err_json}")
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail="Alibaba WAF (Капча/Блокировка): Сессия Qwen требует подтверждения в браузере. Выполните команду /login qwen в терминале."
                            )
                        if "error" in err_json or "code" in err_json:
                            err_msg = err_json.get("message") or err_json.get("error") or err_json.get("code")
                            logger.error(f"❌ Qwen API ошибка в ответе: {err_json}")
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Qwen API error: {err_msg}"
                            )
                    except json.JSONDecodeError:
                        pass

                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    logger.debug("Qwen stream [DONE] получен.")
                    yield StreamChunk(type="status", text="FINISHED", session_id=chat_id, token_usage=token_usage)
                    break

                try:
                    data = json.loads(data_str)

                    # Проверка ошибок внутри SSE
                    if "error" in data or ("code" in data and data["code"] not in [200, "200", 0, "0"]):
                        err_msg = data.get("message") or data.get("error") or data.get("code")
                        logger.error(f"❌ Ошибка в SSE потоке Qwen: {data}")
                        self.reset_session()
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Qwen SSE error: {err_msg}"
                        )

                    if "usage" in data and isinstance(data["usage"], dict):
                        token_usage = data["usage"].get("total_tokens") or data["usage"].get("output_tokens")

                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})

                        # 1. Мысли Qwen
                        extra = delta.get("extra", {})
                        if "summary_thought" in extra and isinstance(extra["summary_thought"], dict):
                            st_content = extra["summary_thought"].get("content", [])
                            if isinstance(st_content, list):
                                full_thought = "\n".join(st_content)
                            else:
                                full_thought = str(st_content)

                            if len(full_thought) > last_thought_len:
                                new_thought_piece = full_thought[last_thought_len:]
                                last_thought_len = len(full_thought)
                                received_chunks_count += 1
                                yield StreamChunk(type="thinking", text=new_thought_piece, session_id=chat_id)

                        elif delta.get("reasoning_content"):
                            received_chunks_count += 1
                            yield StreamChunk(type="thinking", text=delta["reasoning_content"], session_id=chat_id)
                        elif delta.get("thought"):
                            received_chunks_count += 1
                            yield StreamChunk(type="thinking", text=delta["thought"], session_id=chat_id)

                        # 2. Ответ
                        content = delta.get("content")
                        if content:
                            received_chunks_count += 1
                            yield StreamChunk(type="content", text=content, session_id=chat_id, token_usage=token_usage)

                    elif "response" in data and isinstance(data["response"], dict):
                        resp_obj = data["response"]
                        if resp_obj.get("thinking"):
                            received_chunks_count += 1
                            yield StreamChunk(type="thinking", text=resp_obj["thinking"], session_id=chat_id)
                        if resp_obj.get("content"):
                            received_chunks_count += 1
                            yield StreamChunk(type="content", text=resp_obj["content"], session_id=chat_id, token_usage=token_usage)

                    elif "output" in data and isinstance(data["output"], dict):
                        out_text = data["output"].get("text", "")
                        if out_text:
                            received_chunks_count += 1
                            yield StreamChunk(type="content", text=out_text, session_id=chat_id, token_usage=token_usage)

                except HTTPException:
                    raise
                except Exception as e:
                    logger.debug(f"Исключение при парсинге чанка Qwen: {e}")

            if received_chunks_count == 0:
                logger.warning(f"⚠️ Qwen API вернул 0 токенов (chat_id: {chat_id}). Возможно сессия устарела или сработала защита.")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка при вызове Qwen: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ошибка соединения с Qwen API: {str(e)}"
            )

    async def send_message(
        self,
        request: DeepSeekChatRequest,
    ) -> DeepSeekChatResponse:
        full_thinking = []
        full_content = []
        token_usage = None
        session_id = request.chat_session_id or self._current_chat_id or ""

        async for chunk in self.stream_chat(request):
            if chunk.session_id:
                session_id = chunk.session_id
            if chunk.type == "thinking":
                full_thinking.append(chunk.text)
            elif chunk.type == "content":
                full_content.append(chunk.text)
            if chunk.token_usage:
                token_usage = chunk.token_usage

        return DeepSeekChatResponse(
            session_id=session_id,
            message_id=1,
            thinking="".join(full_thinking) if full_thinking else None,
            content="".join(full_content),
            token_usage=token_usage,
            status="FINISHED",
        )
