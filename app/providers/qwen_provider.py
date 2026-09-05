import datetime
import json
import logging
import re
import time
from typing import AsyncGenerator, List, Optional
import uuid
from fastapi import HTTPException, status
import httpx

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
        description="通义千问 3.7 Plus 旗舰 Web 模型，支持深度思考 (Thinking)。",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-3.8",
        name="Qwen 3.8",
        description="第 3 代通义千问旗舰通用大模型，具备深层推理理解能力。",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-3.8-coder",
        name="Qwen 3.8 Coder",
        description="先进的代码专项大模型，专为复杂软件工程、重构与 Agent 流水线优化。",
        model_type="expert",
        supports_thinking=True,
        supports_search=False,
    ),
    ModelInfo(
        id="qwen-3-max",
        name="Qwen 3 Max",
        description="通义千问 3 系列计算能力最强的全功能大模型。",
        model_type="expert",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-3-plus",
        name="Qwen 3 Plus",
        description="均衡高效的高性价比通用大模型。",
        model_type="default",
        supports_thinking=False,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-3-flash",
        name="Qwen 3 Flash",
        description="极速轻量化模型，实现毫秒级首字响应。",
        model_type="default",
        supports_thinking=False,
        supports_search=True,
    ),
    ModelInfo(
        id="qwen-2.5-coder-32b",
        name="Qwen 2.5 Coder 32B",
        description="经典的开源 32B 编程专用大模型。",
        model_type="expert",
        supports_thinking=False,
        supports_search=False,
    ),
]


class QwenProvider(BaseLLMProvider):
    """
    通义千问网页直连 API 提供商 (chat.qwen.ai/api/v2/chat/completions)。
    完整实现 v2.1 通信协议、浏览器请求头伪装与会话管理。
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
        """获取 Qwen 服务端历史会话列表。"""
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
                                "title": it.get("title") or "未命名",
                                "created_at": it.get("created_at"),
                                "updated_at": it.get("updated_at"),
                                "provider": "qwen"
                            })
                return results
        except Exception as e:
            logger.warning(f"获取 Qwen 会话列表失败: {e}")
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
        """构建 chat.qwen.ai 所需的浏览器伪装头。"""
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%a %b %d %Y %H:%M:%S GMT+0000")
        req_id = str(uuid.uuid4())
        think_mode = "Thinking" if thinking_enabled else "Normal"

        if "token=" in token_or_cookie or "; " in token_or_cookie or "_bl_uid=" in token_or_cookie:
            cookie_header = token_or_cookie.strip()
            if "qwen-thinking_mode=" in cookie_header:
                cookie_header = re.sub(r'qwen-thinking_mode=[^;]+', f'qwen-thinking_mode={think_mode}', cookie_header)
            else:
                cookie_header += f"; qwen-thinking_mode={think_mode}"

            auth_token = ""
            m = re.search(r'(?:^|;\s*)token=([^;]+)', cookie_header)
            if m:
                auth_token = m.group(1).strip()
            elif not cookie_header.startswith("token="):
                first_val = cookie_header.split(";")[0].strip()
                if "=" not in first_val:
                    auth_token = first_val
        else:
            auth_token = token_or_cookie.strip()
            cookie_header = (
                f"token={auth_token}; "
                f"qwen-thinking_mode={think_mode}; "
                f"tongyi_sso_ticket={auth_token}; "
                f"login_tongyi_ticket={auth_token}; "
                f"channel=default; "
                f"timezone=Asia/Shanghai"
            )

        headers = {
            "Accept": "text/event-stream",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Authorization": f"Bearer {auth_token}" if auth_token and not auth_token.startswith("Bearer ") else auth_token,
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "Cookie": cookie_header,
            "Origin": "https://chat.qwen.ai",
            "Priority": "u=1, i",
            "Referer": f"https://chat.qwen.ai/c/{chat_id}" if chat_id else "https://chat.qwen.ai/",
            "Sec-Ch-Ua": '"Chromium";v="133", "Google Chrome";v="133", "Not?A_Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "X-Client-Date": now_str,
            "X-Platform": "pc_web",
            "X-Request-Id": req_id,
        }

        return headers

    async def _create_new_chat(self) -> str:
        """通过 POST /api/v2/chats/new 创建新会话。"""
        token = credentials_manager.get_token("qwen")
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Qwen 凭证未配置。请通过命令 /token qwen <token> 设置凭证。"
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
                        logger.info(f"创建 Qwen 新会话: {new_id}")
                        return new_id
        except Exception as e:
            logger.warning(f"通过 /chats/new 创建 Qwen 会话失败: {e}")

        # 回退: 获取现有会话列表
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
        """获取现有 chat_id 或创建新会话。"""
        from app.services.session_manager import session_manager

        if chat_id:
            self._current_chat_id = chat_id
            if session_manager.single_session_mode:
                session_manager.set_provider_session("qwen", chat_id)
            return chat_id

        if session_manager.single_session_mode:
            existing = self._current_chat_id or session_manager.get_provider_session("qwen")
            if existing:
                self._current_chat_id = existing
                session_manager.set_provider_session("qwen", existing)
                logger.debug(f"复用当前 Qwen 单会话 (Single-Session): {existing}")
                return existing

        new_id = await self._create_new_chat()
        if session_manager.single_session_mode and new_id:
            session_manager.set_provider_session("qwen", new_id)
        return new_id

    def _build_payload(self, prompt: str, model: str, chat_id: str, thinking_enabled: bool, search_enabled: bool) -> dict:
        """构建 chat.qwen.ai v2.1 协议的 JSON 请求体。"""
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
                detail="Qwen 凭证未配置。请通过命令 /token qwen <token> 设置凭证。"
            )

        from app.services.context_compressor import context_compressor, estimate_tokens
        if request.prompt:
            request.prompt = context_compressor.compress_raw_prompt(
                request.prompt, max_tokens=context_compressor.QWEN_MAX_WEB_TOKENS
            )

        chat_id = await self.get_or_create_chat(request.chat_session_id)
        resolved_model = self._resolve_qwen_model(request.model)
        thinking_enabled = request.thinking_enabled if request.thinking_enabled is not None else False
        search_enabled = request.search_enabled if request.search_enabled is not None else False

        headers = self._build_headers(token, chat_id, thinking_enabled)
        payload = self._build_payload(request.prompt, resolved_model, chat_id, thinking_enabled, search_enabled)

        logger.info(f"发送请求至 Qwen API (chat_id: {chat_id}, 模型: {resolved_model}, 提示词: ~{estimate_tokens(request.prompt):,} Token)")

        yield StreamChunk(type="session", text=chat_id, session_id=chat_id)

        url = f"{self.base_url}/api/v2/chat/completions?chat_id={chat_id}"

        try:
            req = self.client.build_request("POST", url, json=payload, headers=headers, timeout=180.0)
            resp = await self.client.send(req, stream=True)

            if resp.status_code != 200:
                body = await resp.aread()
                err_text = body.decode("utf-8", errors="replace")
                logger.error(f"Qwen HTTP {resp.status_code} 错误: {err_text}")
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Qwen API 错误 ({resp.status_code}): {err_text}"
                )

            last_thought_len = 0
            token_usage = None
            received_chunks_count = 0

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                # 1. 阿里云 WAF / 验证码检测
                if line.startswith("{"):
                    try:
                        err_json = json.loads(line)
                        ret_list = err_json.get("ret", [])
                        ret_str = str(ret_list)
                        if "FAIL_SYS_USER_VALIDATE" in ret_str or "RGV587_ERROR" in ret_str or "punish" in str(err_json):
                            logger.error(f"❌ 阿里云 WAF 拦截请求 (验证码 / 限流): {err_json}")
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail="阿里云 WAF 触发人机验证。请在终端执行 /login qwen 命令完成验证。"
                            )
                        if "error" in err_json or "code" in err_json:
                            err_msg = err_json.get("message") or err_json.get("error") or err_json.get("code")
                            logger.error(f"❌ Qwen API 返回错误: {err_json}")
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
                    logger.debug("Qwen stream [DONE] 接收完成。")
                    yield StreamChunk(type="status", text="FINISHED", session_id=chat_id, token_usage=token_usage)
                    break

                try:
                    data = json.loads(data_str)

                    if "error" in data or ("code" in data and data["code"] not in [200, "200", 0, "0"]):
                        err_msg = data.get("message") or data.get("error") or data.get("code")
                        logger.error(f"❌ Qwen SSE 流异常: {data}")
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

                        # 1. 思考链 (Thinking)
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

                        # 2. 正文 (Content)
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
                    logger.debug(f"解析 Qwen 分块异常: {e}")

            if received_chunks_count == 0:
                logger.warning(f"⚠️ Qwen API 返回了 0 个 Token (chat_id: {chat_id})。可能触发了会话过期或防护。")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"调用 Qwen 失败: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"连接 Qwen API 失败: {str(e)}"
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
            message_id=0,
            thinking="".join(full_thinking) if full_thinking else None,
            content="".join(full_content),
            token_usage=token_usage,
            status="FINISHED",
        )
