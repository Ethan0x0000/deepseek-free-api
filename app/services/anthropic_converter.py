import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from app.schemas.anthropic import (
    AnthropicContentBlock,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicTool,
    AnthropicUsage,
)
from app.schemas.chat import DeepSeekChatRequest, DeepSeekChatResponse
from app.services.tool_parser import compact_tool_schema, extract_tool_calls


def build_anthropic_tools_prompt(
    tools: List[AnthropicTool],
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> str:
    """生成 Anthropic 格式工具的系统指令。"""
    if tool_choice == "none":
        return ""

    raw_schema_chars = sum(
        len(json.dumps(tool.input_schema or {}))
        for tool in tools
    )
    should_compact = raw_schema_chars > 12_000 or len(tools) > 15

    tool_lines = []
    for tool in tools:
        desc = str(tool.description or "").strip().replace("\r\n", " ")
        if len(desc) > 300:
            desc = desc[:297] + "..."
        params = tool.input_schema or {"type": "object", "properties": {}}
        if should_compact:
            params = compact_tool_schema(params)
        params_str = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
        tool_lines.append(f"## {tool.name}\nDescription: {desc}\nParameters: {params_str}")

    tools_text = "\n\n".join(tool_lines)

    choice_instruction = ""
    if tool_choice is not None:
        if tool_choice in ["required", "any"]:
            choice_instruction = (
                "\n### MANDATORY TOOL EXECUTION\n"
                "You MUST call at least one tool in this turn using `<tool_call>`.\n"
            )
        elif isinstance(tool_choice, dict) and tool_choice.get("name"):
            choice_instruction = (
                f"\n### MANDATORY TOOL EXECUTION\n"
                f"You MUST call the tool '{tool_choice['name']}' in this turn using `<tool_call>`.\n"
            )

    return f"""
# Available Tools
You have access to the following functions/tools to assist the user:

{tools_text}
{choice_instruction}
# Autonomous Engineering Agent Instructions
You are operating as an autonomous, expert software engineering agent in an interactive development workspace.
Your primary objective is to ACCOMPLISH TASKS DIRECTLY using the provided tools, rather than merely giving instructions.

## CORE AGENT PRINCIPLES (CRITICAL):
1. ACTION OVER TALK (多做少说，行动优先):
   - When the user asks you to inspect, search, modify, test, or implement something: YOU MUST EXECUTE THE TOOLS YOURSELF.
   - NEVER tell the user to run commands or perform steps manually when you have tools to perform them.
   - DO NOT say "You can check...", "To find out, run...", or "I suggest checking...". TAKE ACTION DIRECTLY.

2. PERSISTENCE & TASK COMPLETION (不达目的不罢休):
   - Do NOT stop halfway. Real-world tasks require multi-step loops (e.g. search -> read -> analyze -> edit -> verify).
   - Continue calling tools iteratively until the user's request is completely solved and verified.
   - Never stop after one step just to ask "Should I proceed?" on obvious follow-ups. Autonomously continue to the finish.

3. ZERO EMPTY PROMISES (严禁只说不调):
   - Never output conversational promises (e.g. "我来看看...", "让我检查一下...", "Let me check...") without immediately outputting the `<tool_call>` in the SAME message!
   - Keep natural language before tool calls to a minimum (1 brief sentence or zero).

4. NEVER GUESS OR SIMULATE:
   - Never guess file contents, environment states, or command outputs. Execute the tool and wait for real output from the environment.

5. TOOL CALL FORMAT:
   When requesting a tool, output valid JSON inside `<tool_call>...</tool_call>`:
<tool_call>
{{"name": "<function_name>", "arguments": {{...}}}}
</tool_call>

Alternatively, standard DSML format is also accepted:
<|DSML|tool_calls>
<|DSML|invoke name="<function_name>">
<|DSML|parameter name="<param_name>"><![CDATA[<param_val>]]></|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>

6. If no tool call is needed and the entire task is 100% complete, provide your normal conversational response directly.
""".strip()


def convert_anthropic_request_to_deepseek(request: AnthropicMessagesRequest) -> Tuple[DeepSeekChatRequest, bool]:
    """
    将 Anthropic Messages API 请求转换为 DeepSeekChatRequest:
    - 提取 system prompt
    - 解析 content 内容块 (text, image, tool_use, tool_result)
    - 转换 tools 定义为模型提示词
    """
    prompt_parts = []

    # 1. 工具调用指令
    has_tools = bool(request.tools)
    if request.tools:
        tool_choice = getattr(request, "tool_choice", None)
        prompt_parts.append(build_anthropic_tools_prompt(request.tools, tool_choice=tool_choice))

    # 2. 系统提示词 (Anthropic 独立传递)
    if request.system:
        if isinstance(request.system, str):
            prompt_parts.append(f"System Instructions:\n{request.system.strip()}")
        elif isinstance(request.system, list):
            sys_texts = []
            for block in request.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    sys_texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    sys_texts.append(block)
            if sys_texts:
                prompt_parts.append("System Instructions:\n" + "\n".join(sys_texts))

    # 3. 对话历史消息
    history_messages = []
    for msg in request.messages:
        role = msg.role
        role_label = "User" if role == "user" else "Assistant"

        if isinstance(msg.content, str):
            history_messages.append(f"{role_label}: {msg.content}")
        elif isinstance(msg.content, list):
            block_texts = []
            for block in msg.content:
                if isinstance(block, str):
                    block_texts.append(block)
                elif isinstance(block, dict):
                    b_type = block.get("type", "text")
                    if b_type == "text":
                        block_texts.append(block.get("text", ""))
                    elif b_type == "thinking":
                        block_texts.append(f"[Thinking: {block.get('thinking', '')}]")
                    elif b_type in ["image", "image_url"]:
                        block_texts.append("[User provided an image attachment]")
                    elif b_type == "tool_use":
                        fn_name = block.get("name", "")
                        fn_input = json.dumps(block.get("input", {}), ensure_ascii=False)
                        block_texts.append(f"\n<tool_call>\n{{\"name\": \"{fn_name}\", \"arguments\": {fn_input}}}\n</tool_call>")
                    elif b_type == "tool_result":
                        tool_id = block.get("tool_use_id", "tool")
                        res_content = block.get("content", "")
                        if isinstance(res_content, list):
                            res_content = " ".join([c.get("text", "") for c in res_content if isinstance(c, dict)])
                        is_err = " (Error)" if block.get("is_error") else ""
                        block_texts.append(f"Tool [{tool_id}]{is_err} Result:\n{res_content}")
                elif hasattr(block, "type"):
                    if block.type == "text" and block.text:
                        block_texts.append(block.text)
                    elif block.type == "thinking" and block.thinking:
                        block_texts.append(f"[Thinking: {block.thinking}]")
                    elif block.type in ["image", "image_url"]:
                        block_texts.append("[User provided an image attachment]")
                    elif block.type == "tool_use":
                        fn_input = json.dumps(block.input or {}, ensure_ascii=False)
                        block_texts.append(f"\n<tool_call>\n{{\"name\": \"{block.name}\", \"arguments\": {fn_input}}}\n</tool_call>")
                    elif block.type == "tool_result":
                        res_content = block.content or ""
                        block_texts.append(f"Tool [{block.tool_use_id}] Result:\n{res_content}")

            combined_msg = " ".join(block_texts)
            history_messages.append(f"{role_label}: {combined_msg}")

    if history_messages:
        prompt_parts.append("Conversation History:\n" + "\n".join(history_messages))

    # 4. 如果最后一条消息是工具执行结果，指令模型立即执行后续工具调用直到任务彻底完成
    if request.messages:
        last_msg = request.messages[-1]
        is_tool_turn = False
        if isinstance(last_msg.content, list):
            for part in last_msg.content:
                if (isinstance(part, dict) and part.get("type") == "tool_result") or getattr(part, "type", "") == "tool_result":
                    is_tool_turn = True
                    break
        if is_tool_turn:
            prompt_parts.append(
                "\n[Autonomous Directive: The tool execution result is provided above. Proceed with the task immediately. "
                "Analyze the output and invoke the next tool call NOW if more investigation, code editing, or verification is needed: "
                "<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>. "
                "DO NOT stop halfway with an intermediate conversational summary. Work relentlessly until the user's objective is 100% completed!]"
            )

    final_prompt = "\n\n".join(prompt_parts)

    from app.services.context_compressor import context_compressor
    final_prompt = context_compressor.compress_raw_prompt(final_prompt)

    # 判定是否启用思考模式
    thinking_enabled = None
    if request.thinking and request.thinking.type == "enabled":
        thinking_enabled = True

    deepseek_req = DeepSeekChatRequest(
        prompt=final_prompt,
        chat_session_id=request.chat_session_id or request.session_id,
        model=request.model,
        thinking_enabled=thinking_enabled,
        stream=request.stream,
    )

    return deepseek_req, has_tools


def convert_deepseek_response_to_anthropic(
    resp: DeepSeekChatResponse,
    model: str,
    has_tools: bool = False,
    input_tokens: int = 0,
    cached_tokens: int = 0,
) -> AnthropicMessagesResponse:
    """将 DeepSeek 同步响应转换为 AnthropicMessagesResponse。"""
    content_blocks: List[AnthropicContentBlock] = []

    # 1. 处理工具调用
    clean_text = resp.content
    stop_reason = "end_turn"
    found_tool_calls = None

    if has_tools:
        clean_text, found_tool_calls = extract_tool_calls(resp.content)
        if found_tool_calls:
            stop_reason = "tool_use"
            for tc in found_tool_calls:
                try:
                    args_dict = json.loads(tc.function.arguments)
                except Exception:
                    args_dict = {"raw": tc.function.arguments}
                content_blocks.append(
                    AnthropicContentBlock(
                        type="tool_use",
                        id=f"toolu_{uuid.uuid4().hex[:16]}",
                        name=tc.function.name,
                        input=args_dict,
                    )
                )

    # 2. 如果没有工具调用，添加 thinking 和 text 块
    if not found_tool_calls:
        if resp.thinking:
            content_blocks.append(
                AnthropicContentBlock(
                    type="thinking",
                    thinking=resp.thinking,
                )
            )
        if clean_text:
            content_blocks.append(AnthropicContentBlock(type="text", text=clean_text))

    return AnthropicMessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:20]}",
        model=model,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=resp.token_usage or 0,
            cache_read_input_tokens=cached_tokens,
        ),
    )
