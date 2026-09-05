import json
import logging
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

GLM_MODELS = [
    ModelInfo(
        id="glm-5.3",
        name="GLM 5.3",
        description="Флагманская веб-модель Zhipu AI GLM 5.3 с мощными рассуждениями.",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="glm-5.2",
        name="GLM 5.2",
        description="Модель GLM 5.2 общего назначения.",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="glm-5-pro",
        name="GLM 5 Pro",
        description="Профессиональная модель GLM-5 для сложных задач анализа и синтеза информации.",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="glm-5-coder",
        name="GLM 5 Coder",
        description="Специализированная модель линейки GLM-5 для программирования и рефакторинга.",
        model_type="expert",
        supports_thinking=True,
        supports_search=False,
    ),
    ModelInfo(
        id="glm-5-flash",
        name="GLM 5 Flash",
        description="Высокоскоростная модель Zhipu AI с минимальной задержкой.",
        model_type="default",
        supports_thinking=False,
        supports_search=True,
    ),
    ModelInfo(
        id="glm-4-plus",
        name="GLM 4 Plus",
        description="Флагман поколения GLM-4.",
        model_type="expert",
        supports_thinking=False,
        supports_search=True,
    ),
    ModelInfo(
        id="glm-4-flash",
        name="GLM 4 Flash (Бесплатный API)",
        description="Полностью бесплатная и сверхбыстрая модель Zhipu BigModel без капчи.",
        model_type="default",
        supports_thinking=False,
        supports_search=True,
    ),
]


class GLMProvider(BaseLLMProvider):
    """
    Провайдер для GLM (Zhipu AI).
    Поддерживает:
    1. Веб-интерфейс chat.z.ai (/api/v2/chat/completions и /api/v1/chats).
    2. BigModel API (open.bigmodel.cn/api/paas/v4/chat/completions) - без капчи и с бесплатным GLM-4-Flash.
    """

    def __init__(self, http_client: httpx.AsyncClient):
        super().__init__(provider_id="glm", display_name="GLM (Zhipu AI)", http_client=http_client)
        self.web_base_url = "https://chat.z.ai"
        self.open_api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self._current_chat_id: Optional[str] = None

    def get_models(self) -> List[ModelInfo]:
        return GLM_MODELS

    def is_authenticated(self) -> bool:
        return credentials_manager.is_authenticated("glm")

    def get_current_session_id(self) -> Optional[str]:
        return self._current_chat_id

    def set_session_id(self, session_id: str) -> None:
        self._current_chat_id = session_id

    def reset_session(self) -> None:
        self._current_chat_id = None

    async def list_sessions(self) -> List[dict]:
        token = credentials_manager.get_token("glm")
        if not token:
            return []
        headers = self._build_web_headers(token, "")
        url = f"{self.web_base_url}/api/v1/chats/?page=1&type=default"
        try:
            resp = await self.client.get(url, headers=headers, timeout=20.0)
            if resp.status_code == 200:
                items = resp.json()
                results = []
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            results.append({
                                "id": it.get("id"),
                                "title": it.get("title") or "Без названия",
                                "created_at": it.get("created_at"),
                                "updated_at": it.get("updated_at"),
                                "provider": "glm"
                            })
                return results
        except Exception as e:
            logger.warning(f"Ошибка получения списка сессий GLM: {e}")
        return []

    def _extract_user_id(self, token_or_cookie: str) -> str:
        """Извлекает user_id или id из JWT токена."""
        try:
            jwt_token = token_or_cookie
            if "token=" in token_or_cookie:
                for part in token_or_cookie.split(";"):
                    part = part.strip()
                    if part.startswith("token="):
                        jwt_token = part[6:].strip()
            import base64
            parts = jwt_token.split(".")
            if len(parts) >= 2:
                padding = "=" * (4 - len(parts[1]) % 4)
                payload_json = base64.urlsafe_b64decode(parts[1] + padding).decode("utf-8")
                data = json.loads(payload_json)
                return data.get("id") or data.get("user_id") or "unknown"
        except Exception:
            pass
        return "unknown"

    def _build_web_headers(self, token_or_cookie: str, chat_id: str = "") -> dict:
        now_str = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())

        if "token=" in token_or_cookie or "; " in token_or_cookie or "_c_WBKFRo=" in token_or_cookie:
            cookie_header = token_or_cookie.strip()
            jwt_token = ""
            for part in token_or_cookie.split(";"):
                part = part.strip()
                if part.startswith("token="):
                    jwt_token = part[6:].strip()
        else:
            jwt_token = token_or_cookie.strip()
            cookie_header = f"token={jwt_token};"

        referer = f"https://chat.z.ai/c/{chat_id}" if chat_id else "https://chat.z.ai/"

        headers = {
            "Host": "chat.z.ai",
            "Origin": "https://chat.z.ai",
            "Referer": referer,
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9,en-US;q=0.8",
            "authorization": f"Bearer {jwt_token}" if jwt_token else "",
            "content-type": "application/json",
            "sec-ch-ua": '"Not=A?Brand";v="99", "Google Chrome";v="151", "Chromium";v="151"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "x-device-id": f"uid_{uuid.uuid4().hex[:16]}",
            "x-fe-version": "prod-fe-1.1.92",
            "x-region": "overseas",
            "Cookie": cookie_header,
        }
        return headers

    async def stream_chat(
        self,
        request: DeepSeekChatRequest,
    ) -> AsyncGenerator[StreamChunk, None]:
        token = credentials_manager.get_token("glm")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Учетные данные GLM не настроены. Укажите токен через команду /token glm <токен>."
            )

        chat_id = request.chat_session_id or self._current_chat_id or str(uuid.uuid4())
        self._current_chat_id = chat_id

        yield StreamChunk(type="session", text=chat_id, session_id=chat_id)

        # 1. Если это официальный BigModel API ключ (содержит точку, но не JWT с 3 секциями)
        if "." in token and len(token.split(".")) == 2:
            async for chunk in self._stream_bigmodel_api(token, request, chat_id):
                yield chunk
            return

        # 2. Иначе работаем через chat.z.ai веб-интерфейс
        async for chunk in self._stream_web_api(token, request, chat_id):
            yield chunk

    async def _stream_bigmodel_api(
        self,
        api_key: str,
        request: DeepSeekChatRequest,
        chat_id: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Прямой вызов официального BigModel API без капчи."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        model_name = request.model if request.model in ["glm-4-flash", "glm-4-plus", "glm-4-air"] else "glm-4-flash"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": True,
        }
        try:
            req = self.client.build_request("POST", self.open_api_url, json=payload, headers=headers, timeout=120.0)
            resp = await self.client.send(req, stream=True)
            if resp.status_code != 200:
                err = await resp.aread()
                raise HTTPException(status_code=resp.status_code, detail=f"BigModel API error: {err.decode('utf-8')}")

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    yield StreamChunk(type="status", text="FINISHED", session_id=chat_id)
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            yield StreamChunk(type="content", text=delta["content"], session_id=chat_id)
                except Exception:
                    pass
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))

    async def _stream_web_api(
        self,
        token: str,
        request: DeepSeekChatRequest,
        chat_id: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Вызов веб-интерфейса chat.z.ai."""
        user_id = self._extract_user_id(token)
        now_ms = int(time.time() * 1000)
        req_id = str(uuid.uuid4())
        msg_id = str(uuid.uuid4())
        parent_msg_id = str(uuid.uuid4())

        query_params = {
            "timestamp": str(now_ms),
            "requestId": req_id,
            "user_id": user_id,
            "version": "0.0.1",
            "platform": "web",
            "token": token if not "token=" in token else token.split("token=")[1].split(";")[0].strip(),
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
            "language": "zh-CN",
            "languages": "zh-CN,en-US,en",
            "timezone": "Asia/Shanghai",
            "cookie_enabled": "true",
            "screen_width": "1920",
            "screen_height": "1080",
            "screen_resolution": "1920x1080",
            "viewport_height": "903",
            "viewport_width": "1015",
            "viewport_size": "1015x903",
            "color_depth": "24",
            "pixel_ratio": "1",
            "current_url": f"https://chat.z.ai/c/{chat_id}",
            "pathname": f"/c/{chat_id}",
            "search": "",
            "hash": "",
            "host": "chat.z.ai",
            "hostname": "chat.z.ai",
            "protocol": "https:",
            "referrer": "",
            "title": "Z.ai - Advanced AI Chatbot",
            "timezone_offset": "-480",
            "local_time": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "utc_time": time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()),
            "is_mobile": "false",
            "is_touch": "false",
            "max_touch_points": "0",
            "browser_name": "Chrome",
            "os_name": "Windows",
            "signature_timestamp": str(now_ms),
        }

        headers = self._build_web_headers(token, chat_id)
        model_name = request.model if request.model.startswith("glm-") else "glm-5.3"
        thinking_enabled = request.thinking_enabled if request.thinking_enabled is not None else True

        payload = {
            "stream": True,
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            "signature_prompt": request.prompt,
            "params": {},
            "extra": {},
            "features": {
                "image_generation": False,
                "web_search": bool(request.search_enabled),
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "vlm_tools_enable": False,
                "vlm_web_search_enable": False,
                "vlm_website_mode": False,
                "enable_thinking": thinking_enabled,
                "reasoning_effort": "max" if thinking_enabled else "low"
            },
            "variables": {
                "{{USER_NAME}}": "User",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": time.strftime("%Y-%m-%d %H:%M:%S"),
                "{{CURRENT_DATE}}": time.strftime("%Y-%m-%d"),
                "{{CURRENT_TIME}}": time.strftime("%H:%M:%S"),
                "{{CURRENT_WEEKDAY}}": time.strftime("%A"),
                "{{CURRENT_TIMEZONE}}": "Asia/Shanghai",
                "{{USER_LANGUAGE}}": "zh-CN"
            },
            "chat_id": chat_id,
            "id": str(uuid.uuid4()),
            "current_user_message_id": msg_id,
            "current_user_message_parent_id": parent_msg_id,
            "background_tasks": {
                "title_generation": True,
                "tags_generation": True
            }
        }

        url = f"{self.web_base_url}/api/v2/chat/completions"

        try:
            req = self.client.build_request("POST", url, params=query_params, json=payload, headers=headers, timeout=120.0)
            resp = await self.client.send(req, stream=True)

            if resp.status_code != 200:
                body = await resp.aread()
                err_text = body.decode("utf-8", errors="replace")
                logger.error(f"GLM error ({resp.status_code}): {err_text}")
                raise HTTPException(status_code=resp.status_code, detail=f"GLM Web error ({resp.status_code}): {err_text}")

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    yield StreamChunk(type="status", text="FINISHED", session_id=chat_id)
                    break

                try:
                    data = json.loads(data_str)

                    # Проверка на капчу в потоке
                    if "error" in data or ("data" in data and isinstance(data["data"], dict) and "error" in data["data"]):
                        err_obj = data.get("error") or data["data"].get("error")
                        code = err_obj.get("code") or err_obj.get("error_code")
                        if "CAPTCHA" in str(code):
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail="Веб-сервер GLM (chat.z.ai) запросил верификацию капчи (CAPTCHA). Для бесплатного использования без капчи используйте API-ключ Zhipu BigModel (модель GLM-4-Flash) через /token glm <api_key>."
                            )

                    # Парсинг SSE контента и мыслей
                    delta = None
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                    elif "data" in data and isinstance(data["data"], dict):
                        delta = data["data"].get("delta") or data["data"]

                    if delta and isinstance(delta, dict):
                        if delta.get("reasoning_content"):
                            yield StreamChunk(type="thinking", text=delta["reasoning_content"], session_id=chat_id)
                        if delta.get("content"):
                            yield StreamChunk(type="content", text=delta["content"], session_id=chat_id)

                except HTTPException:
                    raise
                except Exception:
                    pass

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка вызова GLM: {e}")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Ошибка GLM API: {str(e)}")

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
