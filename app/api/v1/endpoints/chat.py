import asyncio
import json
import logging
import re
import time
import uuid
from typing import Annotated, AsyncGenerator, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
import httpx

logger = logging.getLogger(__name__)


from app.api.deps import get_http_client
from app.core.config import settings
from app.core.credentials import credentials_manager
from app.providers.registry import provider_registry
from app.schemas.chat import DeepSeekChatRequest, DeepSeekChatResponse, StreamChunk
from app.schemas.openai import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChoice,
    OpenAIChoiceMessage,
    OpenAIChunkChoice,
    OpenAIDelta,
    OpenAIDeltaToolCall,
    OpenAIDeltaToolCallFunction,
    OpenAIToolCall,
    OpenAIUsage,
    OpenAIPromptTokensDetails,
    OpenAICompletionTokensDetails,
)
from app.services.session_manager import session_manager
from app.services.tool_parser import extract_tool_calls, format_messages_to_prompt

# 意图检测正则：检测模型是否仅口头表达了行动意图但未输出 <tool_call>
INTENT_PAT = re.compile(
    r'(?:'
    r'let\s+me\s+(?:study|examine|investigate|analyze|review|search|scan|see|find|check|read|explore|inspect|run|look|implement)|'
    r'i\s*(?:will|\'ll|\s+need\s+to)\s+(?:study|examine|investigate|analyze|review|search|scan|see|find|check|read|explore|inspect|run|look|understand|implement)|'
    r'(?:next|first|now),?\s+(?:i\s+will|let\s+me)|'
    # 中文意图模式
    r'我(?:来|将|会|准备)?(?:看看|看下|查看|检查|分析|读取|看一下|跑一下|运行|执行|探索|搜索|检索|浏览|找找|列出|了解|排查|确认|扫描|定位|统计)|'
    r'让我(?:来|先)?(?:看看|看下|查看|检查|分析|读取|看一下|跑一下|运行|执行|探索|搜索|检索|浏览|找找|列出|了解|排查|确认|扫描|定位|统计)|'
    r'我们(?:需要|先)?(?:看看|看下|查看|检查|分析|读取|看一下|跑一下|运行|执行|探索|搜索|检索|浏览|找找|列出|了解|排查|确认|扫描|定位|统计)|'
    r'先(?:看看|看下|查看|检查|分析|读取|看一下|跑一下|运行|执行|探索|搜索|检索|浏览|找找|列出|排查|确认)|'
    r'接下来(?:我将|让我|我来|我们会|先)|'
    r'帮(?:你|您)(?:查看|检查|分析|读取|列出|搜索|排查)|'
    r'现在(?:我来|让我|我们)'
    r')'
    r'(?:(?![。！？\n]|\.\s).){0,120}'
    r'(?:'
    r'file|code|dir|repo|output|struct|backend|frontend|project|folder|plan|service|parser|content|log|path|'
    r'文件|代码|目录|文件夹|内容|结构|项目|依赖|产物|日志|配置|环境|路径|命令|数据|信息'
    r')',
    re.IGNORECASE,
)

router = APIRouter(tags=["Chat"])


@router.post("/api/v1/chat/send", summary="发送消息 (原生接口，按模型自动路由)")
async def send_chat_message(
    request: DeepSeekChatRequest,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
):
    """
    向所选的 LLM 提供商发送请求 (DeepSeek, Qwen 等)。
    支持流式 (SSE) 与一次性完整同步响应。
    """
    provider = provider_registry.resolve_provider_for_model(request.model)

    if request.stream:
        async def event_generator() -> AsyncGenerator[str, None]:
            async for chunk in provider.stream_chat(request):
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return await provider.send_message(request)


@router.post("/v1/chat/completions", summary="OpenAI 兼容聊天接口 (多提供商、工具调用、思考链与视觉多模态)")
async def openai_chat_completions(
    request: OpenAIChatCompletionRequest,
    raw_req: Request,
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
):
    """
    高度兼容 OpenAI API 规范 (/v1/chat/completions)。
    支持功能:
    - 自动根据模型名路由到对应提供商 (DeepSeek, Qwen)
    - 完整支持 Tool Use (Function Calling) 与自主 Agent 执行模式
    - 原生支持视觉多模态 (Vision image_url / base64)
    - 流式输出 reasoning_content (思考链) 与精确 Token 统计
    - 请求完成后自动在后台回收临时会话，防止网页端列表被刷屏
    """
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="messages 数组不能为空"
        )

    provider = provider_registry.resolve_provider_for_model(request.model)

    from app.services.context_compressor import context_compressor, estimate_tokens
    from app.services.proxy_logger import proxy_logger

    # 1. 格式化所有历史消息与工具定义为上下文提示词，考虑提供商预算限制
    provider_token_limit = context_compressor.get_limit_for_provider(provider.provider_id)
    compiled_prompt = format_messages_to_prompt(
        request.messages,
        request.tools,
        max_tokens=provider_token_limit,
        tool_choice=request.tool_choice,
    )

    # 提前锁定本次请求的 active_token，确保图片上传、会话创建、PoW 求解和推理全程使用同一 Token
    active_token: Optional[str] = None
    incoming_sid = request.chat_session_id or getattr(request, "session_id", None)
    if incoming_sid:
        active_token = session_manager.get_session_token(incoming_sid)
    if not active_token:
        active_token = credentials_manager.get_token("deepseek", rotate=True)

    # 1.1. 处理图像 (Vision 多模态): 提取、计算 PoW、上传并分支到 Vision 模型
    from app.services.image_manager import image_manager
    vision_file_ids = await image_manager.process_images(client, request.messages, token=active_token)

    # 精确计算输入与缓存 Token 数量 (Prompt Caching / LCP)
    prompt_tokens = estimate_tokens(compiled_prompt)
    if len(request.messages) > 1:
        prefix_prompt = format_messages_to_prompt(
            request.messages[:-1],
            request.tools,
            max_tokens=provider_token_limit,
            tool_choice=request.tool_choice,
        )
        cached_tokens = min(estimate_tokens(prefix_prompt), max(0, prompt_tokens - 1))
    else:
        cached_tokens = 0

    # 提取 thinking_enabled 参数 (支持 boolean, dict, extra_body)
    thinking_val: Optional[bool] = None
    if request.thinking_enabled is not None:
        thinking_val = request.thinking_enabled
    elif request.thinking is not None:
        if isinstance(request.thinking, bool):
            thinking_val = request.thinking
        elif isinstance(request.thinking, dict):
            t_type = request.thinking.get("type")
            if t_type == "disabled":
                thinking_val = False
            elif t_type == "enabled":
                thinking_val = True
    elif hasattr(request, "model_extra") and request.model_extra:
        if "thinking_enabled" in request.model_extra:
            thinking_val = bool(request.model_extra["thinking_enabled"])
        elif "thinking" in request.model_extra:
            t = request.model_extra["thinking"]
            if isinstance(t, bool):
                thinking_val = t
            elif isinstance(t, dict):
                t_type = t.get("type")
                if t_type == "disabled":
                    thinking_val = False
                elif t_type == "enabled":
                    thinking_val = True

    search_val: Optional[bool] = request.search_enabled
    if search_val is None and hasattr(request, "model_extra") and request.model_extra:
        if "search_enabled" in request.model_extra:
            search_val = bool(request.model_extra["search_enabled"])

    model_to_use = request.model
    if vision_file_ids:
        # 当存在图片输入时，自动切换为 Vision 视觉多模态模型
        model_to_use = "deepseek-v4-flash-vision-exp"
        search_val = False

    deepseek_req = DeepSeekChatRequest(
        prompt=compiled_prompt,
        chat_session_id=request.chat_session_id or request.session_id,
        ref_file_ids=vision_file_ids or None,
        model=model_to_use,
        stream=request.stream,
        thinking_enabled=thinking_val,
        search_enabled=search_val,
        active_token=active_token,
    )

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    tools_names = [t.function.name for t in (request.tools or []) if t.function]
    ua = raw_req.headers.get("user-agent", "OpenAI Client")
    client_ip = raw_req.client.host if raw_req.client else "127.0.0.1"

    log_id = proxy_logger.log_request_start(
        protocol="OpenAI",
        endpoint="/v1/chat/completions",
        model=request.model,
        provider_name=provider.display_name,
        messages_count=len(request.messages),
        estimated_tokens=estimate_tokens(compiled_prompt),
        tools_names=tools_names,
        user_agent=ua,
        client_ip=client_ip,
    )

    # ── 2. 流式模式 (Streaming) ───────────────────────────────────────────
    if request.stream:
        async def sse_generator() -> AsyncGenerator[str, None]:
            first_chunk_sent = False
            accumulated_content = []
            accumulated_thinking = []
            latest_token_usage: Optional[int] = None
            active_session_id: Optional[str] = None
            sessions_to_clean: set = set()
            has_tools = bool(request.tools)
            active_provider = provider
            # tools 模式滑动窗口缓冲：避免将 <tool_call> 标签碎片过早泄露给客户端
            pending_tail = ""
            TOOL_OPEN_PAT = re.compile(r"<[｜\|]*\s*(?:tool_calls?|invoke|DSML)\b", re.IGNORECASE)

            def flush_live_content(text_piece: str) -> str:
                """保留滑动尾部，安全释放正文内容"""
                nonlocal pending_tail
                pending_tail += text_piece
                m = TOOL_OPEN_PAT.search(pending_tail)
                if m:
                    safe = pending_tail[: m.start()]
                    pending_tail = pending_tail[m.start():]
                    return safe
                hold = 12
                cut = len(pending_tail) - hold if len(pending_tail) > hold else 0
                safe = pending_tail[:cut]
                pending_tail = pending_tail[cut:]
                return safe

            try:
                async for chunk in active_provider.stream_chat(deepseek_req):
                    if chunk.type == "error":
                        raise HTTPException(status_code=400, detail=chunk.text)

                    if chunk.session_id:
                        active_session_id = chunk.session_id

                    if chunk.token_usage is not None:
                        latest_token_usage = chunk.token_usage

                    # 思考链过程 (Thinking)
                    if chunk.type == "thinking":
                        proxy_logger.log_thinking_chunk(log_id, chunk.text)
                        accumulated_thinking.append(chunk.text)
                        if not first_chunk_sent:
                            first_chunk = OpenAIChatCompletionChunk(
                                id=req_id,
                                model=request.model,
                                choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
                            )
                            yield f"data: {first_chunk.model_dump_json()}\n\n"
                            first_chunk_sent = True

                        c = OpenAIChatCompletionChunk(
                            id=req_id,
                            model=request.model,
                            choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(reasoning_content=chunk.text))],
                        )
                        yield f"data: {c.model_dump_json()}\n\n"

                    # 正文内容 (Content)
                    elif chunk.type == "content":
                        accumulated_content.append(chunk.text)
                        if not has_tools:
                            if not first_chunk_sent:
                                first_chunk = OpenAIChatCompletionChunk(
                                    id=req_id,
                                    model=request.model,
                                    choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
                                )
                                yield f"data: {first_chunk.model_dump_json()}\n\n"
                                first_chunk_sent = True

                            proxy_logger.log_content_chunk(log_id, chunk.text)
                            c = OpenAIChatCompletionChunk(
                                id=req_id,
                                model=request.model,
                                choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(content=chunk.text))],
                            )
                            yield f"data: {c.model_dump_json()}\n\n"
                        else:
                            safe = flush_live_content(chunk.text)
                            if safe and not first_chunk_sent:
                                first_chunk = OpenAIChatCompletionChunk(
                                    id=req_id,
                                    model=request.model,
                                    choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
                                )
                                yield f"data: {first_chunk.model_dump_json()}\n\n"
                                first_chunk_sent = True
                            if safe:
                                proxy_logger.log_content_chunk(log_id, safe)
                                c = OpenAIChatCompletionChunk(
                                    id=req_id,
                                    model=request.model,
                                    choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(content=safe))],
                                )
                                yield f"data: {c.model_dump_json()}\n\n"

                # 释放剩余缓冲区 (在没有 tool 匹配风险时安全释放)
                if has_tools and pending_tail and not TOOL_OPEN_PAT.search(pending_tail) and "<tool_call" not in pending_tail and "<invoke" not in pending_tail:
                    if not first_chunk_sent:
                        first_chunk = OpenAIChatCompletionChunk(
                            id=req_id,
                            model=request.model,
                            choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
                        )
                        yield f"data: {first_chunk.model_dump_json()}\n\n"
                        first_chunk_sent = True
                    proxy_logger.log_content_chunk(log_id, pending_tail)
                    c = OpenAIChatCompletionChunk(
                        id=req_id,
                        model=request.model,
                        choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(content=pending_tail))],
                    )
                    yield f"data: {c.model_dump_json()}\n\n"
                    pending_tail = ""

                # 工具调用解析与补全恢复
                finish_reason = "stop"
                full_text = "".join(accumulated_content)
                if active_session_id:
                    sessions_to_clean.add(active_session_id)

                allowed_tool_names = {t.function.name for t in (request.tools or []) if t.function} if request.tools else None

                if has_tools:
                    clean_text, tool_calls = extract_tool_calls(full_text, allowed_tool_names=allowed_tool_names)
                    clean_prefix = re.split(r'<[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?[^>]*>', full_text)[0].strip()

                    # 意图检测与自动补全恢复 (Continuation Recovery)
                    # 仅在模型确实有未闭合的 tool_call 标签或表达了行动意图但未输出工具时触发，避免纯 Markdown 解释误触发
                    has_unclosed_tool = (
                        ("<tool_call" in full_text and "</tool_call" not in full_text)
                        or ("<invoke" in full_text and "</invoke" not in full_text)
                        or ("DSML" in full_text and "invoke" in full_text and ("</invoke" not in full_text and "</｜DSML｜" not in full_text))
                    )
                    if not tool_calls and (INTENT_PAT.search(full_text) or has_unclosed_tool):
                        logger.info("检测到行动意图声明或未闭合 tool_call，启动自动补全 (Continuation Recovery)...")
                        try:
                            action_hint = clean_prefix[-150:] if len(clean_prefix) > 150 else clean_prefix
                            cont_req = DeepSeekChatRequest(
                                prompt=(
                                    f"{deepseek_req.prompt}\n\n"
                                    f"[Assistant response so far]\n{clean_prefix}\n\n"
                                    f"[STRICT INSTRUCTION: Execute the tool call for \"{action_hint}\" immediately. "
                                    f"Do NOT output raw code or file paths directly. "
                                    f"Output valid JSON inside <tool_call>: "
                                    f"<tool_call>\n{{\"name\": \"<function_name>\", \"arguments\": {{...}}}}\n</tool_call>]"
                                ),
                                chat_session_id=active_session_id or deepseek_req.chat_session_id,
                                model=deepseek_req.model,
                                stream=False,
                            )
                            cont_resp = await asyncio.wait_for(active_provider.send_message(cont_req), timeout=15.0)
                            if getattr(cont_resp, "session_id", None):
                                sessions_to_clean.add(cont_resp.session_id)
                            logger.info(f"Continuation 响应内容: {cont_resp.content!r}")
                            cont_clean, cont_tools = extract_tool_calls(cont_resp.content, allowed_tool_names=allowed_tool_names)
                            if cont_tools:
                                tool_calls = cont_tools
                                if cont_clean:
                                    clean_text = (clean_text or clean_prefix) + "\n" + cont_clean
                                logger.info(f"✓ 成功通过 Continuation Recovery 恢复 {len(cont_tools)} 个工具调用!")
                        except Exception as cont_err:
                            logger.warning(f"Continuation Recovery 补全异常: {cont_err}")

                    # 如果最终没有工具调用（例如是纯文本回答或文档解释），把因包含 <tool_call> 等文本而在滑动窗口中被扣留的尾部正文完整释放
                    if not tool_calls and pending_tail:
                        if not first_chunk_sent:
                            first_chunk = OpenAIChatCompletionChunk(
                                id=req_id,
                                model=request.model,
                                choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
                            )
                            yield f"data: {first_chunk.model_dump_json()}\n\n"
                            first_chunk_sent = True
                        proxy_logger.log_content_chunk(log_id, pending_tail)
                        c = OpenAIChatCompletionChunk(
                            id=req_id,
                            model=request.model,
                            choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(content=pending_tail))],
                        )
                        yield f"data: {c.model_dump_json()}\n\n"
                        pending_tail = ""

                    if tool_calls:
                        finish_reason = "tool_calls"
                        delta_tools = []
                        for idx, tc in enumerate(tool_calls):
                            proxy_logger.log_tool_call(log_id, tc.function.name, tc.function.arguments)
                            delta_tools.append(
                                OpenAIDeltaToolCall(
                                    index=idx,
                                    id=tc.id,
                                    type="function",
                                    function=OpenAIDeltaToolCallFunction(
                                        name=tc.function.name,
                                        arguments=tc.function.arguments,
                                    ),
                                )
                            )

                        if not first_chunk_sent:
                            first_chunk = OpenAIChatCompletionChunk(
                                id=req_id,
                                model=request.model,
                                choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(role="assistant"))],
                            )
                            yield f"data: {first_chunk.model_dump_json()}\n\n"
                            first_chunk_sent = True

                        c = OpenAIChatCompletionChunk(
                            id=req_id,
                            model=request.model,
                            choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(tool_calls=delta_tools))],
                        )
                        yield f"data: {c.model_dump_json(exclude_none=True)}\n\n"

                # 计算 Token Usage (Input, Output, Reasoning, Cached)
                full_text = "".join(accumulated_content)
                full_thinking = "".join(accumulated_thinking)
                reasoning_tokens = estimate_tokens(full_thinking)

                if latest_token_usage is not None:
                    completion_tokens = latest_token_usage
                else:
                    completion_tokens = estimate_tokens(full_text) + reasoning_tokens

                usage_obj = OpenAIUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    prompt_tokens_details=OpenAIPromptTokensDetails(cached_tokens=cached_tokens),
                    completion_tokens_details=OpenAICompletionTokensDetails(reasoning_tokens=reasoning_tokens),
                )

                # 结束分块
                final_chunk = OpenAIChatCompletionChunk(
                    id=req_id,
                    model=request.model,
                    choices=[OpenAIChunkChoice(index=0, delta=OpenAIDelta(), finish_reason=finish_reason)],
                    usage=usage_obj,
                )
                yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"

                # 兼容 OpenAI stream_options: {"include_usage": true} 规范
                usage_chunk = OpenAIChatCompletionChunk(
                    id=req_id,
                    model=request.model,
                    choices=[],
                    usage=usage_obj,
                )
                yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"
                yield "data: [DONE]\n\n"
                proxy_logger.log_request_end(log_id, status_code=200, tokens_out=completion_tokens)

                # 后台静默回收网页端临时会话，保持左侧列表整洁
                if settings.AUTO_CLEAN_WEB_SESSIONS and not (request.chat_session_id or request.session_id):
                    clean_sid = active_session_id or session_manager.get_current_session_id()
                    if clean_sid:
                        sessions_to_clean.add(clean_sid)
                    for sid in sessions_to_clean:
                        asyncio.create_task(session_manager.delete_session(client, sid))

            except Exception as e:
                try:
                    active_provider.reset_session()
                except Exception:
                    pass
                err_detail = getattr(e, "detail", str(e))
                err_status = getattr(e, "status_code", 500)
                proxy_logger.log_request_end(log_id, status_code=err_status, error=str(err_detail))
                err_chunk = {
                    "error": {
                        "message": str(err_detail),
                        "type": "provider_error",
                        "code": err_status,
                    }
                }
                yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

                if settings.AUTO_CLEAN_WEB_SESSIONS and not (request.chat_session_id or request.session_id):
                    clean_sid = active_session_id or session_manager.get_current_session_id()
                    if clean_sid:
                        sessions_to_clean.add(clean_sid)
                    for sid in sessions_to_clean:
                        asyncio.create_task(session_manager.delete_session(client, sid))

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── 3. 同步非流式模式 (Non-streaming) ──────────────────────────────────
    else:
        sessions_to_clean: set = set()
        try:
            resp = await provider.send_message(deepseek_req)

            clean_text = resp.content
            tool_calls: Optional[List[OpenAIToolCall]] = None
            finish_reason = "stop"
            reasoning_to_return = resp.thinking or None
            if getattr(resp, "session_id", None):
                sessions_to_clean.add(resp.session_id)

            allowed_tool_names = {t.function.name for t in (request.tools or []) if t.function} if request.tools else None

            if request.tools:
                clean_text, found_tool_calls = extract_tool_calls(resp.content, allowed_tool_names=allowed_tool_names)
                clean_prefix = re.split(r'<[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?[^>]*>', resp.content)[0].strip()

                has_unclosed_tool = (
                    ("<tool_call" in resp.content and "</tool_call" not in resp.content)
                    or ("<invoke" in resp.content and "</invoke" not in resp.content)
                    or ("DSML" in resp.content and "invoke" in resp.content and ("</invoke" not in resp.content and "</｜DSML｜" not in resp.content))
                )
                if not found_tool_calls and (INTENT_PAT.search(resp.content) or has_unclosed_tool):
                    logger.info("Non-streaming: 检测到行动意图声明或未闭合 tool_call，启动自动补全...")
                    try:
                        action_hint = clean_prefix[-150:] if len(clean_prefix) > 150 else clean_prefix
                        cont_req = DeepSeekChatRequest(
                            prompt=(
                                f"{deepseek_req.prompt}\n\n"
                                f"[Assistant response so far]\n{clean_prefix}\n\n"
                                f"[STRICT INSTRUCTION: Execute the tool call for \"{action_hint}\" immediately. "
                                f"Do NOT output raw code or file paths directly. "
                                f"Output valid JSON inside <tool_call>: "
                                f"<tool_call>\n{{\"name\": \"<function_name>\", \"arguments\": {{...}}}}\n</tool_call>]"
                            ),
                            chat_session_id=getattr(resp, "session_id", None) or deepseek_req.chat_session_id,
                            model=deepseek_req.model,
                            stream=False,
                        )
                        cont_resp = await asyncio.wait_for(provider.send_message(cont_req), timeout=15.0)
                        if getattr(cont_resp, "session_id", None):
                            sessions_to_clean.add(cont_resp.session_id)
                        cont_clean, cont_tools = extract_tool_calls(cont_resp.content, allowed_tool_names=allowed_tool_names)
                        if cont_tools:
                            found_tool_calls = cont_tools
                            if cont_clean:
                                clean_text = (clean_text or clean_prefix) + "\n" + cont_clean
                            logger.info(f"✓ Non-streaming 成功恢复 {len(cont_tools)} 个工具调用!")
                    except Exception as cont_err:
                        logger.warning(f"Non-streaming 自动补全异常: {cont_err}")

                if found_tool_calls:
                    tool_calls = found_tool_calls
                    finish_reason = "tool_calls"
                    reasoning_to_return = None
                    clean_text = None
                    for tc in found_tool_calls:
                        proxy_logger.log_tool_call(log_id, tc.function.name, tc.function.arguments)

            choice_message = OpenAIChoiceMessage(
                role="assistant",
                content=clean_text,
                reasoning_content=reasoning_to_return,
                tool_calls=tool_calls,
            )

            # 计算 Token Usage (Input, Output, Reasoning, Cached)
            reasoning_tokens = estimate_tokens(resp.thinking or "")
            if resp.token_usage:
                completion_tokens = resp.token_usage
            else:
                completion_tokens = estimate_tokens(resp.content) + reasoning_tokens

            usage_obj = OpenAIUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                prompt_tokens_details=OpenAIPromptTokensDetails(cached_tokens=cached_tokens),
                completion_tokens_details=OpenAICompletionTokensDetails(reasoning_tokens=reasoning_tokens),
            )
            proxy_logger.log_request_end(log_id, status_code=200, tokens_out=completion_tokens)

            # 后台静默回收网页端临时会话
            if settings.AUTO_CLEAN_WEB_SESSIONS and not (request.chat_session_id or request.session_id):
                clean_sid = getattr(resp, "session_id", None) or session_manager.get_current_session_id()
                if clean_sid:
                    sessions_to_clean.add(clean_sid)
                for sid in sessions_to_clean:
                    asyncio.create_task(session_manager.delete_session(client, sid))

            return OpenAIChatCompletionResponse(
                id=req_id,
                model=request.model,
                choices=[
                    OpenAIChoice(
                        index=0,
                        message=choice_message,
                        finish_reason=finish_reason,
                    )
                ],
                usage=usage_obj,
            )
        except Exception as e:
            proxy_logger.log_request_end(log_id, status_code=500, error=str(e))
            raise
