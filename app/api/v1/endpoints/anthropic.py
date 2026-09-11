import json
import logging
import uuid
from typing import Annotated, AsyncGenerator, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
import httpx

logger = logging.getLogger(__name__)


from app.api.deps import get_http_client
from app.core.credentials import credentials_manager
from app.providers.registry import provider_registry
from app.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from app.services.anthropic_converter import (
    convert_anthropic_request_to_deepseek,
    convert_deepseek_response_to_anthropic,
)
from app.services.tool_parser import extract_tool_calls

router = APIRouter(tags=["Anthropic"])


@router.post("/v1/messages", summary="Anthropic Messages API 兼容端点")
@router.post("/api/v1/messages", summary="Anthropic Messages API 兼容端点")
async def anthropic_messages(
    request: AnthropicMessagesRequest,
    raw_req: Request,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
):
    """
    兼容 Anthropic Claude Messages API 规范 (/v1/messages)。
    自动根据模型路由，并支持完整的 Tool Use 与 Thinking 推理链。
    """
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="messages 字段为必填项且不能为空"
        )

    provider = provider_registry.resolve_provider_for_model(request.model)
    deepseek_req, has_tools = convert_anthropic_request_to_deepseek(request)

    # 提前锁定本次请求的 active_token，确保全流程一致 (支持多账号容灾重试)
    active_token: Optional[str] = None
    if request.chat_session_id or getattr(request, "session_id", None):
        sid = request.chat_session_id or getattr(request, "session_id", None)
        active_token = session_manager.get_session_token(sid)

    # 图像多模态处理 (Vision Multimodal): 提取图片、计算 PoW、上传并 fork 给 Vision 模型
    from app.services.image_manager import image_manager
    has_images = bool(image_manager.extract_images_from_messages(request.messages))
    vision_file_ids = []
    if has_images:
        max_upload_attempts = max(1, len(credentials_manager.get_all_tokens("deepseek")))
        last_upload_err = None
        for _ in range(max_upload_attempts):
            if not active_token:
                active_token = credentials_manager.get_token("deepseek", rotate=True)
            try:
                vision_file_ids = await image_manager.process_images(client, request.messages, token=active_token)
                break
            except Exception as e:
                last_upload_err = e
                active_token = None
                continue
        if not vision_file_ids and last_upload_err:
            raise HTTPException(status_code=400, detail=f"图片上传解析失败: {last_upload_err}")
    elif not active_token:
        active_token = credentials_manager.get_token("deepseek", rotate=True)

    if vision_file_ids:
        deepseek_req.ref_file_ids = vision_file_ids
        if deepseek_req.model in ["deepseek-chat", "deepseek"]:
            deepseek_req.model = "deepseek-flash"

    deepseek_req.active_token = active_token

    from app.services.context_compressor import context_compressor, estimate_tokens
    from app.services.proxy_logger import proxy_logger

    provider_token_limit = context_compressor.get_limit_for_provider(provider.provider_id)
    if deepseek_req.prompt:
        deepseek_req.prompt = context_compressor.compress_raw_prompt(
            deepseek_req.prompt, max_tokens=provider_token_limit
        )

    # 精确计算输入与缓存 Token 数量 (Prompt Caching)
    prompt_tokens = estimate_tokens(deepseek_req.prompt or "")
    if len(request.messages) > 1:
        try:
            prefix_req, _ = convert_anthropic_request_to_deepseek(
                AnthropicMessagesRequest(
                    model=request.model,
                    messages=request.messages[:-1],
                    system=request.system,
                    tools=request.tools,
                )
            )
            cached_tokens = min(estimate_tokens(prefix_req.prompt or ""), max(0, prompt_tokens - 1))
        except Exception:
            cached_tokens = 0
    else:
        cached_tokens = 0

    msg_id = f"msg_{uuid.uuid4().hex[:20]}"

    tools_names = [t.name for t in (request.tools or [])]
    ua = raw_req.headers.get("user-agent", "Anthropic Client")
    client_ip = raw_req.client.host if raw_req.client else "127.0.0.1"

    log_id = proxy_logger.log_request_start(
        protocol="Anthropic",
        endpoint="/v1/messages",
        model=request.model,
        provider_name=provider.display_name,
        messages_count=len(request.messages),
        estimated_tokens=estimate_tokens(deepseek_req.prompt or ""),
        tools_names=tools_names,
        user_agent=ua,
        client_ip=client_ip,
    )

    # ── 1. 流式模式 (Anthropic SSE Streaming) ───────────────────────────
    if request.stream:
        async def anthropic_sse_generator() -> AsyncGenerator[str, None]:
            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
            block_index = 0
            in_thinking_block = False
            in_text_block = False
            message_started = False
            accumulated_content = []
            accumulated_thinking = []
            has_tools = bool(request.tools)
            active_provider = provider
            active_session_id: Optional[str] = None
            latest_token_usage: Optional[int] = None

            def emit_start():
                nonlocal message_started
                if not message_started:
                    start_event = {
                        "type": "message_start",
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "model": request.model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": prompt_tokens,
                                "output_tokens": 0,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": cached_tokens,
                            },
                        }
                    }
                    message_started = True
                    return f"event: message_start\ndata: {json.dumps(start_event, ensure_ascii=False)}\n\n"
                return ""

            latest_token_usage: Optional[int] = None

            try:
                async for chunk in active_provider.stream_chat(deepseek_req):
                    if chunk.type == "error":
                        raise HTTPException(status_code=400, detail=chunk.text)

                    if chunk.token_usage is not None:
                        latest_token_usage = chunk.token_usage

                    if chunk.session_id:
                        active_session_id = chunk.session_id

                    # 思考链 (Thinking)
                    if chunk.type == "thinking":
                        proxy_logger.log_thinking_chunk(log_id, chunk.text)
                        accumulated_thinking.append(chunk.text)
                        if not has_tools:
                            ev = emit_start()
                            if ev:
                                yield ev

                            if not in_thinking_block:
                                cb_start = {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {"type": "thinking", "thinking": ""},
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n"
                                in_thinking_block = True

                            cb_delta = {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "thinking_delta", "thinking": chunk.text},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n"

                    # 文本内容 (Content)
                    elif chunk.type == "content":
                        accumulated_content.append(chunk.text)
                        if not has_tools:
                            ev = emit_start()
                            if ev:
                                yield ev

                            proxy_logger.log_content_chunk(log_id, chunk.text)
                            if in_thinking_block:
                                cb_stop = {"type": "content_block_stop", "index": block_index}
                                yield f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n"
                                in_thinking_block = False
                                block_index += 1

                            if not in_text_block:
                                cb_start = {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {"type": "text", "text": ""},
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n"
                                in_text_block = True

                            cb_delta = {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "text_delta", "text": chunk.text},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n"

                if in_thinking_block:
                    cb_stop = {"type": "content_block_stop", "index": block_index}
                    yield f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n"
                    in_thinking_block = False
                    block_index += 1

                stop_reason = "end_turn"
                full_text = "".join(accumulated_content)

                # 工具调用解析
                if has_tools:
                    allowed_tool_names = {t.name for t in (request.tools or []) if t.name} if request.tools else None
                    clean_text, tool_calls = extract_tool_calls(full_text, allowed_tool_names=allowed_tool_names)
                    # 兜底：如果正文中没有工具调用，但 thinking 思考链中误输出了工具调用，智能拦截纠偏
                    if not tool_calls and accumulated_thinking:
                        th_full = "".join(accumulated_thinking)
                        thinking_clean, thinking_tools = extract_tool_calls(th_full, allowed_tool_names=allowed_tool_names)
                        if thinking_tools:
                            logger.warning(f"Anthropic streaming: 成功从 thinking 中拯救 {len(thinking_tools)} 个工具调用！")
                            tool_calls = thinking_tools
                            clean_text = ""

                    if tool_calls:
                        stop_reason = "tool_use"
                        ev = emit_start()
                        if ev:
                            yield ev

                        for tc in tool_calls:
                            proxy_logger.log_tool_call(log_id, tc.function.name, tc.function.arguments)
                            try:
                                args_dict = json.loads(tc.function.arguments)
                            except Exception:
                                args_dict = {"raw": tc.function.arguments}

                            tu_id = f"toolu_{uuid.uuid4().hex[:16]}"
                            cb_start = {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tu_id,
                                    "name": tc.function.name,
                                    "input": {},
                                },
                            }
                            yield f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n"

                            cb_delta = {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(args_dict, ensure_ascii=False),
                                },
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n"

                            cb_stop = {"type": "content_block_stop", "index": block_index}
                            yield f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n"
                            block_index += 1
                    else:
                        ev = emit_start()
                        if ev:
                            yield ev

                        if accumulated_thinking:
                            th_full = "".join(accumulated_thinking)
                            cb_start = {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {"type": "thinking", "thinking": ""},
                            }
                            yield f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n"
                            cb_delta = {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "thinking_delta", "thinking": th_full},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n"
                            cb_stop = {"type": "content_block_stop", "index": block_index}
                            yield f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n"
                            block_index += 1

                        text_out = clean_text or full_text
                        if text_out:
                            cb_start = {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {"type": "text", "text": ""},
                            }
                            yield f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n"
                            cb_delta = {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "text_delta", "text": text_out},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n"
                            cb_stop = {"type": "content_block_stop", "index": block_index}
                            yield f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n"
                            block_index += 1
                else:
                    if in_text_block:
                        cb_stop = {"type": "content_block_stop", "index": block_index}
                        yield f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n"

                # message_delta
                full_text = "".join(accumulated_content)
                full_thinking = "".join(accumulated_thinking)
                reasoning_tokens = estimate_tokens(full_thinking)
                if latest_token_usage is not None:
                    completion_tokens = latest_token_usage
                else:
                    completion_tokens = estimate_tokens(full_text) + reasoning_tokens

                msg_delta = {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": completion_tokens},
                }
                yield f"event: message_delta\ndata: {json.dumps(msg_delta, ensure_ascii=False)}\n\n"

                # message_stop
                yield "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n"
                proxy_logger.log_request_end(log_id, status_code=200, tokens_out=completion_tokens)

                # 后台静默回收网页端临时会话
                if settings.AUTO_CLEAN_WEB_SESSIONS and not (request.chat_session_id or request.session_id):
                    clean_sid = active_session_id or session_manager.get_current_session_id()
                    if clean_sid:
                        asyncio.create_task(session_manager.delete_session(client, clean_sid))
            except Exception as e:
                try:
                    active_provider.reset_session()
                except Exception:
                    pass
                err_detail = getattr(e, "detail", str(e))
                err_status = getattr(e, "status_code", 500)
                proxy_logger.log_request_end(log_id, status_code=err_status, error=str(err_detail))
                err_event = {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": str(err_detail),
                    }
                }
                yield f"event: error\ndata: {json.dumps(err_event, ensure_ascii=False)}\n\n"

                if settings.AUTO_CLEAN_WEB_SESSIONS and not (request.chat_session_id or request.session_id):
                    clean_sid = active_session_id or session_manager.get_current_session_id()
                    if clean_sid:
                        asyncio.create_task(session_manager.delete_session(client, clean_sid))

        return StreamingResponse(
            anthropic_sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── 2. 同步非流式模式 (Non-streaming) ──────────────────────────────────
    else:
        try:
            resp = await provider.send_message(deepseek_req)

            result = convert_deepseek_response_to_anthropic(
                resp,
                model=request.model,
                has_tools=has_tools,
                input_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
            )
            proxy_logger.log_request_end(log_id, status_code=200, tokens_out=resp.token_usage or 0)

            # 后台静默回收网页端临时会话
            if settings.AUTO_CLEAN_WEB_SESSIONS and not (request.chat_session_id or request.session_id):
                clean_sid = getattr(resp, "session_id", None) or session_manager.get_current_session_id()
                if clean_sid:
                    asyncio.create_task(session_manager.delete_session(client, clean_sid))

            return result
        except Exception as e:
            proxy_logger.log_request_end(log_id, status_code=500, error=str(e))
            raise
