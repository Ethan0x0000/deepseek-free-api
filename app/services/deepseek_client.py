import json
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Any
from fastapi import HTTPException, status
import httpx

from app.core.config import settings
from app.core.credentials import credentials_manager
from app.core.pow_solver import pow_solver
from app.schemas.chat import (
    DeepSeekChatRequest,
    DeepSeekChatResponse,
    ModelInfo,
    StreamChunk,
)
from app.services.session_manager import session_manager
from app.services.sse_parser import parse_sse_lines

logger = logging.getLogger(__name__)


AVAILABLE_MODELS = [
    ModelInfo(
        id="deepseek-v4.1-flash",
        name="DeepSeek V4.1 Flash (Unified)",
        description="DeepSeek 2026年9月全新三合一统一多模态模型，原生融合极速文本、深度思考、联网搜索与图像视觉理解。",
        model_type="default",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        description="1.6T MoE 旗舰模型 (49B 激活参数)，专为复杂编程、代码重构、数学与深度推理优化。",
        model_type="expert",
        supports_thinking=True,
        supports_search=False,
    ),
    ModelInfo(
        id="deepseek-v4-flash",
        name="DeepSeek V4 Flash",
        description="284B MoE 超高速模型 (13B 激活参数)，低延迟快速响应，适合轻量级任务与高频调用。",
        model_type="default",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="deepseek-v4-flash-vision-exp",
        name="DeepSeek V4 Flash Vision",
        description="DeepSeek V4 视觉多模态模型，支持图片理解、图表识别与视觉代码分析。",
        model_type="default",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="deepseek-reasoner",
        name="DeepSeek R1 (Reasoner)",
        description="DeepSeek-R1 深度思考推理模型，提供完整的思维链推理过程输出。",
        model_type="default",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="deepseek-chat",
        name="DeepSeek V3",
        description="DeepSeek 通用对话模型 (默认统一智能模式)。",
        model_type="default",
        supports_thinking=True,
        supports_search=True,
    ),
    ModelInfo(
        id="deepseek-search",
        name="DeepSeek V3 (Search)",
        description="内置实时联网搜索增强的 DeepSeek 对话模型。",
        model_type="default",
        supports_thinking=True,
        supports_search=True,
    ),
]


class DeepSeekClient:
    """与 DeepSeek Web 界面通信的客户端 (支持自动 PoW 求解与 SSE 流式解析)。"""

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self._external_client = client
        self._internal_client: Optional[httpx.AsyncClient] = None
        self.session_manager = session_manager

    @property
    def client(self) -> httpx.AsyncClient:
        if self._external_client:
            return self._external_client
        if self._internal_client is None or self._internal_client.is_closed:
            self._internal_client = httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT)
        return self._internal_client

    def resolve_model_params(
        self,
        model_name: str,
        thinking_enabled: Optional[bool] = None,
        search_enabled: Optional[bool] = None
    ) -> Tuple[str, bool, bool]:
        """解析并返回内部 model_type 以及 thinking / search 标志。默认开启思维链。"""
        model_lower = model_name.lower().strip()

        # 1. DeepSeek V4 / V4.1 系列
        if model_lower in ["deepseek-v4.1", "deepseek-v4.1-flash", "v4.1", "v4.1-flash"]:
            model_type = "default"
            think = True if thinking_enabled is None else thinking_enabled
            search = search_enabled if search_enabled is not None else False
        elif model_lower in ["deepseek-v4-pro", "v4-pro", "v4", "deepseek-v4", "pro"]:
            model_type = "expert"
            think = True if thinking_enabled is None else thinking_enabled
            search = search_enabled if search_enabled is not None else False
        elif model_lower in ["deepseek-v4-flash", "v4-flash", "flash"]:
            model_type = "default"
            think = True if thinking_enabled is None else thinking_enabled
            search = search_enabled if search_enabled is not None else False
        elif model_lower in ["deepseek-v4-flash-vision-exp", "v4-vision", "vision", "deepseek-vision"]:
            # 新版统一支持 default 多模态
            model_type = "default"
            think = True if thinking_enabled is None else thinking_enabled
            search = search_enabled if search_enabled is not None else False

        # 2. 联网搜索模型
        elif search_enabled is True or model_lower in ["deepseek-search", "search"]:
            model_type = "default"
            think = True if thinking_enabled is None else thinking_enabled
            search = True

        # 3. 推理模型 (DeepSeek-R1)
        elif thinking_enabled is True or model_lower in ["deepseek-reasoner", "r1", "reasoner", "deepseek_reasoner"]:
            model_type = "default"
            think = True
            search = search_enabled if search_enabled is not None else False

        # 4. 默认通用对话
        else:
            model_type = "default"
            think = True if thinking_enabled is None else thinking_enabled
            search = search_enabled if search_enabled is not None else False

        return model_type, think, search

    async def stream_chat(
        self,
        request: DeepSeekChatRequest
    ) -> AsyncGenerator[StreamChunk, None]:
        """向 DeepSeek 发送流式对话请求并自动求解 PoW，具备多账号熔断与快速故障转移机制。"""
        if not credentials_manager.is_authenticated("deepseek"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="DeepSeek 认证凭证未配置。请通过 /api/v1/auth/token 接口或 credentials.json 提供 Token。"
            )

        all_tokens = credentials_manager.get_all_tokens("deepseek")
        max_attempts = max(3, len(all_tokens) + 1)
        last_exception = None

        for attempt in range(max_attempts):
            # 1. 确定当前请求绑定的 Token (优先使用上游已锁定的 active_token，其次复用会话对应账号，否则从 Token 池轮询健康 Token)
            active_token: Optional[str] = request.active_token
            if not active_token and request.chat_session_id:
                active_token = session_manager.get_session_token(request.chat_session_id)
            if not active_token:
                active_token = credentials_manager.get_token("deepseek", rotate=True)

            if not active_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="DeepSeek 所有可用 Token 均处于冷却或不可用状态。"
                )

            # 2. 获取或创建会话 (锁定 active_token)
            session_id = await session_manager.get_or_create_session(
                self.client, request.chat_session_id, token=active_token
            )

            # 3. 确定 parent_message_id
            parent_msg_id = request.parent_message_id
            if parent_msg_id is None:
                parent_msg_id = session_manager.get_parent_message_id(session_id)

            # 4. 确定模型参数
            model_type, thinking_enabled, search_enabled = self.resolve_model_params(
                request.model, request.thinking_enabled, request.search_enabled
            )

            # 5. 计算 PoW challenge (传入匹配的 active_token)
            target_path = "/api/v0/chat/completion"
            try:
                pow_header = await pow_solver.get_pow_header(self.client, target_path, token=active_token)
            except Exception as e:
                masked_tok = f"{active_token[:6]}...{active_token[-4:]}" if len(active_token) > 10 else "***"
                logger.error(f"计算 PoW 挑战失败 (Token: {masked_tok}): {e}")
                credentials_manager.mark_token_status("deepseek", active_token, cooldown_seconds=60, error=f"PoW failure: {e}")
                last_exception = HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"求解 DeepSeek Proof-of-Work 失败: {str(e)}"
                )
                request.active_token = None
                continue

            headers = {
                "accept": "*/*",
                "authorization": f"Bearer {active_token}",
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

            # 5.1. 针对多模态与视觉处理 (新版 V4.1 统一模型支持 default 原生多模态及看图联网搜索)
            if model_type == "vision":
                search_enabled = False
            elif request.ref_file_ids:
                model_type = "default"

            if request.ref_file_ids or model_type == "vision":
                token = active_token
                if token:
                    try:
                        from app.services.hif_provider import hif_provider
                        hif_headers = await hif_provider.get_headers(self.client, token)
                        headers.update(hif_headers)
                    except Exception as hif_err:
                        logger.warning(f"获取 Vision HIF 签名失败: {hif_err}")
                headers["x-client-version"] = "2.3.0"
                headers["x-app-version"] = "2.3.0"

            payload = {
                "chat_session_id": session_id,
                "parent_message_id": parent_msg_id,
                "model_type": model_type,
                "prompt": request.prompt,
                "ref_file_ids": request.ref_file_ids or [],
                "thinking_enabled": thinking_enabled,
                "search_enabled": search_enabled,
                "action": None,
                "preempt": False,
            }

            url = f"{settings.DEEPSEEK_BASE_URL}{target_path}"
            last_message_id: Optional[int] = None
            extracted_title: Optional[str] = None
            chunk_streamed = False

            try:
                req = self.client.build_request("POST", url, json=payload, headers=headers, timeout=settings.REQUEST_TIMEOUT)
                resp = await self.client.send(req, stream=True)

                if resp.status_code != 200:
                    body = await resp.aread()
                    err_text = body.decode("utf-8", errors="replace")
                    masked_tok = f"{active_token[:6]}...{active_token[-4:]}" if len(active_token) > 10 else "***"
                    logger.error(f"DeepSeek [Token: {masked_tok}] 返回错误状态码 {resp.status_code}: {err_text}")

                    lower_err = err_text.lower()
                    if resp.status_code in [401, 403] or "已被禁言" in err_text or "规范" in err_text or "authorization failed" in lower_err:
                        credentials_manager.mark_token_status(
                            "deepseek", active_token, cooldown_seconds=86400, error=f"HTTP {resp.status_code}: {err_text[:120]}"
                        )
                        logger.warning(f"Token [{masked_tok}] 触发官方封禁/鉴权失败，已自动隔离冷却 24 小时")
                    elif resp.status_code == 429 or "too many" in lower_err or "频繁" in err_text:
                        credentials_manager.mark_token_status(
                            "deepseek", active_token, cooldown_seconds=60, error="HTTP 429 Too Many Requests"
                        )
                        logger.warning(f"Token [{masked_tok}] 触发频率限制 (429)，已自动冷却 60 秒")
                    else:
                        credentials_manager.mark_token_status("deepseek", active_token, error=f"HTTP {resp.status_code}")

                    if resp.status_code in [400, 404] or "session" in err_text.lower():
                        session_manager.invalidate_current_session()
                    session_manager._session_tokens.pop(session_id, None)
                    request.active_token = None
                    last_exception = HTTPException(
                        status_code=resp.status_code,
                        detail=f"DeepSeek 错误: {err_text}"
                    )
                    continue

                # 检查是否返回了非 SSE 的 JSON 业务报错 (如 HTTP 200 但包含 user is muted)
                content_type = resp.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    body = await resp.aread()
                    try:
                        data = json.loads(body.decode("utf-8", errors="replace"))
                    except Exception:
                        data = {}
                    biz_data = data.get("data", {})
                    biz_code = biz_data.get("biz_code") if isinstance(biz_data, dict) else None
                    biz_msg = biz_data.get("biz_msg") if isinstance(biz_data, dict) else data.get("msg", "")

                    masked_tok = f"{active_token[:6]}...{active_token[-4:]}" if len(active_token) > 10 else "***"
                    logger.error(f"DeepSeek [Token: {masked_tok}] 返回业务异常 JSON: {data}")

                    if biz_code == 5 or "user is muted" in str(biz_msg).lower() or "muted" in str(biz_msg).lower() or "禁言" in str(biz_msg):
                        mute_until = None
                        if isinstance(biz_data, dict):
                            inner = biz_data.get("biz_data", {})
                            if isinstance(inner, dict):
                                mute_until = inner.get("mute_until")
                        cooldown = 86400
                        if mute_until and mute_until > time.time():
                            cooldown = mute_until - time.time() + 60
                        credentials_manager.mark_token_status(
                            "deepseek", active_token, cooldown_seconds=cooldown, error=f"官方禁言: {biz_msg} (直至 {mute_until})"
                        )
                        logger.warning(f"Token [{masked_tok}] 处于官方禁言状态，已自动隔离冷却 {int(cooldown)} 秒")
                    else:
                        credentials_manager.mark_token_status("deepseek", active_token, cooldown_seconds=60, error=f"业务错误: {biz_msg}")

                    session_manager._session_tokens.pop(session_id, None)
                    request.active_token = None
                    last_exception = HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN if biz_code == 5 else status.HTTP_502_BAD_GATEWAY,
                        detail=f"DeepSeek 错误: {biz_msg or data}"
                    )
                    continue

                # 正常 SSE 流式解析并向下游推送
                rate_limited_early = False
                async for chunk in parse_sse_lines(resp.aiter_lines(), session_id):
                    if chunk.type == "error":
                        err_lower = chunk.text.lower()
                        if "muted" in err_lower or "禁言" in chunk.text:
                            credentials_manager.mark_token_status("deepseek", active_token, cooldown_seconds=86400, error=chunk.text)
                        elif "too many" in err_lower or "频繁" in chunk.text or "frequent" in err_lower or "rate_limit" in err_lower:
                            if not chunk_streamed:
                                rate_limited_early = True
                                logger.warning(f"DeepSeek 网页端触发瞬时频控 (rate_limit_reached)，准备短暂退避重试...")
                                break
                    chunk_streamed = True
                    if chunk.message_id:
                        last_message_id = chunk.message_id
                    if chunk.type == "title" and chunk.text:
                        extracted_title = chunk.text
                    yield chunk

                if rate_limited_early:
                    import asyncio
                    await asyncio.sleep(2.5)
                    last_exception = HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="DeepSeek 触发瞬时频控，已自动退避重试"
                    )
                    continue

                if last_message_id:
                    session_manager.update_session_state(session_id, last_message_id, extracted_title)
                    credentials_manager.mark_token_status("deepseek", active_token, is_success=True)
                return

            except HTTPException as e:
                if chunk_streamed:
                    raise e
                last_exception = e
                session_manager._session_tokens.pop(session_id, None)
                request.active_token = None
                continue

        if last_exception:
            raise last_exception

    async def send_message(
        self,
        request: DeepSeekChatRequest
    ) -> DeepSeekChatResponse:
        """stream_chat 的同步聚合包装，等待并返回完整回答。"""
        full_thinking = []
        full_content = []
        session_id = request.chat_session_id or ""
        last_msg_id = 0
        error_msg = None
        token_usage = None

        async for chunk in self.stream_chat(request):
            if chunk.type == "thinking":
                full_thinking.append(chunk.text)
            elif chunk.type == "content":
                full_content.append(chunk.text)
            elif chunk.type == "error":
                error_msg = chunk.text
            if chunk.session_id:
                session_id = chunk.session_id
            if chunk.message_id:
                last_msg_id = chunk.message_id
            if chunk.token_usage:
                token_usage = chunk.token_usage

        if error_msg:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS if ("频繁" in error_msg.lower() or "too frequent" in error_msg.lower()) else status.HTTP_502_BAD_GATEWAY,
                detail=error_msg
            )

        thinking_text = "".join(full_thinking)
        content_text = "".join(full_content)

        return DeepSeekChatResponse(
            session_id=session_id,
            message_id=last_msg_id,
            thinking=thinking_text if thinking_text else None,
            content=content_text,
            token_usage=token_usage,
            status="FINISHED",
        )
