import json
import re
import uuid
from typing import List, Optional, Tuple, Any, Dict, Union
from app.schemas.openai import OpenAIChatMessage, OpenAITool, OpenAIToolCall, OpenAIToolCallFunction


def compact_tool_schema(value: Any, is_root: bool = True) -> Any:
    """
    精简工具参数的 JSON Schema 结构：
    - 移除冗余元数据 (title, $comment, verbose examples)
    - 保留核心验证结构 (type, properties, required, enum, const, items)
    - 截断过长的参数描述 (>120 字符)
    """
    if isinstance(value, list):
        return [compact_tool_schema(item, is_root=False) for item in value]
    if not isinstance(value, dict):
        return value

    compact = {}
    for k, v in value.items():
        if not is_root and k in {"title", "$comment"}:
            continue
        if not is_root and k == "description" and isinstance(v, str) and len(v) > 120:
            compact[k] = v[:117] + "..."
            continue

        if k in {"properties", "patternProperties", "definitions", "$defs"} and isinstance(v, dict):
            compact[k] = {pk: compact_tool_schema(pv, is_root=False) for pk, pv in v.items()}
        elif k in {"items", "additionalProperties", "contains"} and isinstance(v, dict):
            compact[k] = compact_tool_schema(v, is_root=False)
        elif k in {"anyOf", "allOf", "oneOf", "prefixItems"} and isinstance(v, list):
            compact[k] = [compact_tool_schema(item, is_root=False) for item in v]
        else:
            compact[k] = v
    return compact


def build_tool_system_prompt(
    tools: List[OpenAITool],
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> str:
    """生成包含可用工具定义的系统指令。"""
    if tool_choice == "none":
        return ""

    raw_schema_chars = sum(
        len(json.dumps(tool.function.parameters or {}))
        for tool in tools
        if tool.type == "function" and tool.function
    )
    should_compact = raw_schema_chars > 12_000 or len(tools) > 15

    tool_lines = []
    for tool in tools:
        if tool.type == "function" and tool.function:
            fn_name = tool.function.name
            desc = str(tool.function.description or "").strip().replace("\r\n", " ")
            if len(desc) > 300:
                desc = desc[:297] + "..."
            params = tool.function.parameters or {"type": "object", "properties": {}}
            if should_compact:
                params = compact_tool_schema(params)
            params_str = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
            tool_lines.append(f"## {fn_name}\nDescription: {desc}\nParameters: {params_str}")

    tools_text = "\n\n".join(tool_lines)

    choice_instruction = ""
    if tool_choice is not None:
        if tool_choice in ["required", "any"]:
            choice_instruction = (
                "\n### MANDATORY TOOL EXECUTION (tool_choice='required')\n"
                "You MUST call at least one tool in this turn. "
                "A direct conversational answer without a `<tool_call>` block is strictly forbidden!\n"
            )
        elif isinstance(tool_choice, dict):
            fn_target = tool_choice.get("function", {}).get("name") or tool_choice.get("name", "")
            if fn_target:
                choice_instruction = (
                    f"\n### MANDATORY TOOL EXECUTION (tool_choice='{fn_target}')\n"
                    f"You MUST call the tool '{fn_target}' in this turn using `<tool_call>`.\n"
                )
        elif isinstance(tool_choice, str) and tool_choice not in ["auto", "none"]:
            choice_instruction = (
                f"\n### MANDATORY TOOL EXECUTION (tool_choice='{tool_choice}')\n"
                f"You MUST call the tool '{tool_choice}' in this turn using `<tool_call>`.\n"
            )

    prompt = f"""
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

Alternatively, standard DSML or JSON format is also accepted:
<|DSML|tool_calls>
<|DSML|invoke name="<function_name>">
<|DSML|parameter name="<param_name>"><![CDATA[<param_val>]]></|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>

6. EXAMPLES OF CORRECT BEHAVIOR:
Example 1 (English):
User: "Explore the codebase"
Assistant:
Let me study the files to understand the project structure.
<tool_call>
{{"name": "shell", "arguments": {{"command": "ls -la"}}}}
</tool_call>

Example 2 (Chinese):
User: "当前文件夹中有哪些是生产垃圾"
Assistant:
我来查看当前目录下的文件与结构。
<tool_call>
{{"name": "shell", "arguments": {{"command": "ls -la"}}}}
</tool_call>

FORBIDDEN BEHAVIOR (NEVER DO THIS / 严禁只说不调):
Assistant: "我来看看当前文件夹里有什么内容，帮你识别生产垃圾..." -> WRONG! Never stop without the `<tool_call>` block! Always invoke the tool call immediately!

7. CRITICAL RULES FOR FILE EDITING / WRITING:
When modifying or editing a file, ALWAYS invoke `edit` or `write` with valid JSON arguments!
NEVER output raw code or file paths directly inside `<tool_call>` without the JSON structure:
CORRECT:
<tool_call>
{{"name": "edit", "arguments": {{"path": "path/to/file.py", "oldString": "exact old code", "newString": "new code"}}}}
</tool_call>
or:
<tool_call>
{{"name": "write", "arguments": {{"path": "path/to/file.py", "content": "full new content"}}}}
</tool_call>

8. If no tool call is needed and the entire task is 100% complete, provide your normal conversational response directly.
"""
    return prompt.strip()


def format_messages_to_prompt(
    messages: List[OpenAIChatMessage],
    tools: Optional[List[OpenAITool]] = None,
    max_tokens: Optional[int] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> str:
    """
    将 OpenAI 消息历史 (system, user, assistant, tool) 编译为 Web 端统一提示词。
    在超出 Token 预算时自动执行上下文压缩。
    """
    from app.services.context_compressor import context_compressor
    compressed_messages = context_compressor.compress_openai_messages(messages, max_tokens=max_tokens)

    prompt_parts = []

    # 1. 如果提供了 tools，添加工具调用系统指令
    if tools:
        tool_instruction = build_tool_system_prompt(tools, tool_choice=tool_choice)
        if tool_instruction:
            prompt_parts.append(tool_instruction)

    # 2. 处理系统消息与历史对话
    system_messages = []
    history_messages = []

    for msg in compressed_messages:
        role = msg.role
        content = msg.content or ""
        if isinstance(content, list):
            # 兼容 multipart 格式 (文本 + 图片附件)
            text_pieces = []
            for piece in content:
                if isinstance(piece, dict):
                    p_type = piece.get("type", "text")
                    if p_type == "text":
                        text_pieces.append(piece.get("text", ""))
                    elif p_type in ["image_url", "image"]:
                        text_pieces.append("[User provided an image attachment]")
                elif isinstance(piece, str):
                    text_pieces.append(piece)
            content = " ".join(text_pieces)

        if role == "system":
            system_messages.append(content)
        elif role == "user":
            history_messages.append(f"User: {content}")
        elif role == "assistant":
            if msg.tool_calls:
                tc_str = ""
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = tc.function.arguments
                    tc_str += f"\n<tool_call>\n{{\"name\": \"{fn_name}\", \"arguments\": {fn_args}}}\n</tool_call>"
                history_messages.append(f"Assistant: {content}{tc_str}")
            else:
                history_messages.append(f"Assistant: {content}")
        elif role in ["tool", "function"]:
            tool_id = msg.tool_call_id or msg.name or "tool"
            history_messages.append(f"Tool [{tool_id}] Output:\n{content}")

    if system_messages:
        prompt_parts.append("System Instructions:\n" + "\n".join(system_messages))

    if history_messages:
        prompt_parts.append("\nConversation History:\n" + "\n".join(history_messages))

    # 3. 如果上一条是工具执行结果，指令模型立即分析并执行下一步工具调用，直到任务彻底完成
    if compressed_messages and compressed_messages[-1].role in ["tool", "function"]:
        prompt_parts.append(
            "\n[Autonomous Directive: The tool execution result is provided above. Proceed with the task immediately. "
            "Analyze the output and invoke the next tool call NOW if more investigation, code editing, or verification is needed: "
            "<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>. "
            "DO NOT stop halfway with an intermediate conversational summary. Work relentlessly until the user's objective is 100% completed!]"
        )
    # 4. 如果提供了 tools 且最后一条是用户指令，注入行动优先指令
    elif tools and compressed_messages and compressed_messages[-1].role == "user":
        prompt_parts.append(
            "\n[Autonomous Directive: Tools are available. Action over words: If answering this request requires inspecting files, exploring directories, searching code, or executing commands, invoke the tool call directly in this turn: <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>. "
            "DO NOT respond with advice or tell the user to do it manually.]"
        )

    full_prompt = "\n\n".join(prompt_parts)
    if max_tokens:
        full_prompt = context_compressor.compress_raw_prompt(full_prompt, max_tokens=max_tokens)
    return full_prompt


def normalize_qwen_parameter_tags(text: str) -> str:
    """标准化 Qwen 参数标签 (<parameter=key>...</parameter>) 为标准 JSON。"""
    if not text or "parameter" not in text:
        return text
    # 1. 替换参数间过渡
    normalized = re.sub(r'\s*</parameter>\s*<parameter=([a-zA-Z0-9_\-]+)>\s*', r'", "\1": ', text)
    # 2. 替换单独闭合标签
    normalized = re.sub(r'\s*</parameter>', r'"', normalized)
    # 3. 替换起始标签
    normalized = re.sub(r'<parameter=([a-zA-Z0-9_\-]+)>\s*', r'"\1": ', normalized)
    return normalized


def _parse_broken_arguments(args_str: str) -> Dict[str, Any]:
    """
    容错参数解析器：
    - 修复带有未转义内部引号的 JSON (如 shell 命令里的 echo "...", grep '...')
    - 支持通过键值对位置切分提取字段
    """
    s = args_str.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    try:
        data = json.loads(args_str, strict=False)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    key_pat = re.compile(r'"([a-zA-Z0-9_\-]+)"\s*:\s*')
    matches = list(key_pat.finditer(s))
    if not matches:
        return {}

    result = {}
    for i in range(len(matches)):
        key = matches[i].group(1)
        val_start = matches[i].end()
        val_end = matches[i + 1].start() if i + 1 < len(matches) else len(s)

        raw_val = s[val_start:val_end].strip()
        if raw_val.endswith(","):
            raw_val = raw_val[:-1].strip()
        if raw_val.startswith('"') and raw_val.endswith('"') and len(raw_val) >= 2:
            raw_val = raw_val[1:-1]
        elif raw_val.startswith('"'):
            raw_val = raw_val[1:]
        elif raw_val.endswith('"'):
            raw_val = raw_val[:-1]

        if (raw_val.startswith("{") and raw_val.endswith("}")) or (raw_val.startswith("[") and raw_val.endswith("]")):
            try:
                raw_val = json.loads(raw_val)
            except Exception:
                pass

        result[key] = raw_val

    return result


def _parse_all_tool_json(raw_json: str) -> List[Tuple[str, str]]:
    """解析 tool_call 块内的所有 JSON 对象 (支持单对象或并行多对象)。"""
    results: List[Tuple[str, str]] = []
    if not raw_json:
        return results

    s = normalize_qwen_parameter_tags(raw_json.strip())
    decoder = json.JSONDecoder(strict=False)
    idx = 0
    while idx < len(s):
        while idx < len(s) and s[idx].isspace():
            idx += 1
        if idx >= len(s):
            break
        try:
            obj, end_idx = decoder.raw_decode(s, idx)
            if isinstance(obj, dict):
                name = obj.get("name") or obj.get("function")
                args = obj.get("arguments") or obj.get("parameters") or obj.get("input", {})
                if name:
                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    results.append((str(name).strip(), args_str))
            idx = end_idx
        except Exception:
            next_brace = s.find('{', idx + 1)
            if next_brace != -1:
                idx = next_brace
            else:
                break

    # 回退 1: 标准 json.loads
    if not results:
        try:
            data = json.loads(s, strict=False)
            if isinstance(data, dict):
                name = data.get("name") or data.get("function")
                args = data.get("arguments") or data.get("parameters") or data.get("input", {})
                if name:
                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    results.append((str(name).strip(), args_str))
        except Exception:
            pass

    # 回退 2: 转义原始换行符
    if not results:
        try:
            sanitized = re.sub(r'[\r\n]+', '\\n', s)
            data = json.loads(sanitized, strict=False)
            if isinstance(data, dict):
                name = data.get("name") or data.get("function")
                args = data.get("arguments") or data.get("parameters") or data.get("input", {})
                if name:
                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    results.append((str(name).strip(), args_str))
        except Exception:
            pass

    # 回退 3: 正则提取被破坏的内部引号 JSON
    if not results:
        name_match = re.search(r'"(?:name|function)"\s*:\s*"([a-zA-Z0-9_\-\.]+)"', s)
        if name_match:
            name = name_match.group(1).strip()
            args_start = re.search(r'"(?:arguments|parameters|input)"\s*:\s*(\{)', s)
            if args_start:
                brace_start = args_start.start(1)
                brace_count = 0
                brace_end = -1
                for i in range(brace_start, len(s)):
                    if s[i] == '{':
                        brace_count += 1
                    elif s[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            brace_end = i + 1
                            break
                if brace_end != -1:
                    args_raw = s[brace_start:brace_end]
                    args_dict = _parse_broken_arguments(args_raw)
                    if args_dict:
                        results.append((name, json.dumps(args_dict, ensure_ascii=False)))

    return results


def extract_tool_calls(text: str) -> Tuple[str, List[OpenAIToolCall]]:
    """
    从模型输出内容中提取工具调用 (Tool Calls)：
    - 支持单个或多个 JSON 对象封装在 <tool_call>...</tool_call> 内
    - 支持 DeepSeek 官方 DSML 格式 (<|DSML|invoke name="...">...</|DSML|invoke>)
    - 支持 Anthropic/Claude 格式 (<invoke name="...">...</invoke>)
    - 支持 Qwen 原生标签格式 (<function=name>...</function>)
    - 支持 Markdown 代码块语法 (```tool_call...```)
    - 自动去重相同调用并返回 (清洗后的正文文本, 工具调用列表)
    """
    tool_calls: List[OpenAIToolCall] = []
    seen_calls = set()
    clean_text = text

    # 0. 检验 DeepSeek 原生 DSML 格式
    dsml_invoke_pat = r"<[｜\|]*\s*DSML\s*[｜\|]*invoke\s+name=[\"']?([^\"'>]+)[\"']?[^>]*>\s*(.*?)\s*</[｜\|]*\s*DSML\s*[｜\|]*invoke>"
    dsml_param_pat = r"<[｜\|]*\s*DSML\s*[｜\|]*parameter\s+name=[\"']?([^\"'>]+)[\"']?[^>]*>\s*(.*?)\s*</[｜\|]*\s*DSML\s*[｜\|]*parameter>"

    for match in re.finditer(dsml_invoke_pat, text, re.DOTALL):
        name = match.group(1).strip()
        body = match.group(2).strip()
        args_dict = {}
        for pm in re.finditer(dsml_param_pat, body, re.DOTALL):
            p_name = pm.group(1).strip()
            p_val = pm.group(2).strip()
            if (p_val.startswith("{") and p_val.endswith("}")) or (p_val.startswith("[") and p_val.endswith("]")):
                try:
                    p_val = json.loads(p_val)
                except Exception:
                    pass
            args_dict[p_name] = p_val

        args_str = json.dumps(args_dict, ensure_ascii=False)
        call_key = (name, args_str)
        if call_key not in seen_calls:
            seen_calls.add(call_key)
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                OpenAIToolCall(
                    id=call_id,
                    type="function",
                    function=OpenAIToolCallFunction(name=name, arguments=args_str),
                )
            )

    clean_text = re.sub(r"<[｜\|]*\s*DSML\s*[｜\|]*tool_calls?>.*?</[｜\|]*\s*DSML\s*[｜\|]*tool_calls?>", "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(dsml_invoke_pat, "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(r"</?[｜\|]*\s*DSML\s*[｜\|]*[^>]*>", "", clean_text)

    # 1. 检验 Claude / Anthropic 格式: <invoke name="...">...</invoke>
    invoke_pat = r"<invoke\s+name=[\"']?([a-zA-Z0-9_\-\.]+)[\"']?[^>]*>\s*(.*?)\s*</invoke>"
    param_pat = r"<parameter\s+(?:name=[\"']?([a-zA-Z0-9_\-]+)[\"']?|=([a-zA-Z0-9_\-]+)|([a-zA-Z0-9_\-]+))[^>]*>\s*(.*?)\s*</parameter>"

    for match in re.finditer(invoke_pat, text, re.DOTALL):
        name = match.group(1).strip()
        body = match.group(2).strip()
        args_dict = {}
        for pm in re.finditer(param_pat, body, re.DOTALL):
            p_name = pm.group(1) or pm.group(2) or pm.group(3)
            p_val = pm.group(4).strip()
            if (p_val.startswith("{") and p_val.endswith("}")) or (p_val.startswith("[") and p_val.endswith("]")):
                try:
                    p_val = json.loads(p_val)
                except Exception:
                    pass
            args_dict[p_name] = p_val

        args_str = json.dumps(args_dict, ensure_ascii=False)
        call_key = (name, args_str)
        if call_key not in seen_calls:
            seen_calls.add(call_key)
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                OpenAIToolCall(
                    id=call_id,
                    type="function",
                    function=OpenAIToolCallFunction(name=name, arguments=args_str),
                )
            )

    clean_text = re.sub(r"<[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?[^>]*>\s*(?:<invoke\b.*?</invoke>\s*)+</[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?>", "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(invoke_pat, "", clean_text, flags=re.DOTALL)

    # 2. 匹配标准 JSON tool_call 块
    patterns = [
        r"<[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?[^>]*>\s*(.*?)\s*</[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?[^>]*>",
        r"```(?:tool_call|tool_calls|function_call)\s*(.*?)\s*```",
    ]

    for pat in patterns:
        for match in re.finditer(pat, text, re.DOTALL):
            raw_content = match.group(1)
            parsed_list = _parse_all_tool_json(raw_content)
            for name, args_str in parsed_list:
                call_key = (name, args_str)
                if call_key not in seen_calls:
                    seen_calls.add(call_key)
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    tool_calls.append(
                        OpenAIToolCall(
                            id=call_id,
                            type="function",
                            function=OpenAIToolCallFunction(name=name, arguments=args_str),
                        )
                    )
            clean_text = re.sub(pat, "", clean_text, flags=re.DOTALL)

    # 3. 匹配 Qwen 原生格式: <function=name>args</function>
    func_pat = r"<function=([a-zA-Z0-9_\-\.]+)[^>]*>\s*(.*?)\s*</function>"
    for match in re.finditer(func_pat, text, re.DOTALL):
        name = match.group(1).strip()
        raw_args = match.group(2).strip()
        args_str = raw_args
        if "<parameter" in raw_args:
            param_dict = {}
            for p in re.finditer(r'<parameter=([a-zA-Z0-9_\-]+)>\s*(.*?)\s*(?:</parameter>|$)', raw_args, re.DOTALL):
                param_dict[p.group(1)] = p.group(2).strip().strip("\"'")
            if param_dict:
                args_str = json.dumps(param_dict, ensure_ascii=False)
        else:
            try:
                args_obj = json.loads(raw_args, strict=False)
                args_str = json.dumps(args_obj, ensure_ascii=False) if isinstance(args_obj, dict) else str(args_obj)
            except Exception:
                args_str = raw_args

        call_key = (name, args_str)
        if call_key not in seen_calls:
            seen_calls.add(call_key)
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                OpenAIToolCall(
                    id=call_id,
                    type="function",
                    function=OpenAIToolCallFunction(name=name, arguments=args_str),
                )
            )
    clean_text = re.sub(func_pat, "", clean_text, flags=re.DOTALL)

    # 4. 匹配无标签裸露的 JSON 工具调用 (Naked JSON tool call)
    naked_pat = re.compile(
        r'\{\s*"(?:name|function)"\s*:\s*"([a-zA-Z0-9_\-\.]+)"\s*,\s*"(?:arguments|parameters|input)"\s*:\s*(\{)',
        re.DOTALL
    )
    for match in naked_pat.finditer(clean_text):
        name = match.group(1).strip()
        start_idx = match.start()
        args_brace_start = match.start(2)

        brace_count = 0
        args_end_idx = -1
        for i in range(args_brace_start, len(clean_text)):
            if clean_text[i] == '{':
                brace_count += 1
            elif clean_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    args_end_idx = i + 1
                    break

        if args_end_idx == -1:
            continue

        args_raw = clean_text[args_brace_start:args_end_idx]
        outer_end_idx = clean_text.find('}', args_end_idx)
        if outer_end_idx != -1:
            outer_end_idx += 1
        else:
            outer_end_idx = args_end_idx

        block = clean_text[start_idx:outer_end_idx]
        args_dict = _parse_broken_arguments(args_raw)
        args_str = json.dumps(args_dict, ensure_ascii=False) if args_dict else "{}"

        call_key = (name, args_str)
        if call_key not in seen_calls:
            seen_calls.add(call_key)
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                OpenAIToolCall(
                    id=call_id,
                    type="function",
                    function=OpenAIToolCallFunction(name=name, arguments=args_str),
                )
            )
        clean_text = clean_text.replace(block, "")

    # 反序结构兼容: {"arguments": ..., "name": "..."}
    naked_rev_pat = re.compile(
        r'\{\s*"(?:arguments|parameters|input)"\s*:\s*(\{.*?\}).*?,\s*"(?:name|function)"\s*:\s*"([a-zA-Z0-9_\-\.]+)"\s*\}',
        re.DOTALL
    )
    for match in naked_rev_pat.finditer(clean_text):
        args_raw = match.group(1).strip()
        name = match.group(2).strip()
        block = match.group(0)

        args_dict = _parse_broken_arguments(args_raw)
        args_str = json.dumps(args_dict, ensure_ascii=False) if args_dict else "{}"

        call_key = (name, args_str)
        if call_key not in seen_calls:
            seen_calls.add(call_key)
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                OpenAIToolCall(
                    id=call_id,
                    type="function",
                    function=OpenAIToolCallFunction(name=name, arguments=args_str),
                )
            )
        clean_text = clean_text.replace(block, "")

    # 5. 容错提取非标准裸文件编辑指令
    raw_file_call_pat = re.compile(
        r'<[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?[^>]*>\s*([a-zA-Z]:[\\/][^ \r\n\t]+|[a-zA-Z0-9_\-\.\/]+\.[a-zA-Z0-9]+)\s+([\s\S]+?)(?:</[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?>|$)',
        re.DOTALL
    )
    for match in raw_file_call_pat.finditer(clean_text):
        fpath = match.group(1).strip()
        code_body = match.group(2).strip()

        if '"name"' in code_body or '"arguments"' in code_body:
            continue

        old_str = ""
        new_str = ""
        is_edit = False

        prefix_candidates = [code_body[:n] for n in [50, 40, 30, 25, 20] if len(code_body) > n * 2]
        for p in prefix_candidates:
            sec = code_body.find(p, len(p))
            if sec != -1:
                old_str = code_body[:sec].strip()
                new_str = code_body[sec:].strip()
                is_edit = True
                break

        if is_edit:
            tool_name = "Edit"
            args_obj = {"file_path": fpath, "old_string": old_str, "new_string": new_str}
        else:
            tool_name = "Write"
            args_obj = {"file_path": fpath, "content": code_body}

        args_str = json.dumps(args_obj, ensure_ascii=False)
        call_key = (tool_name, args_str)
        if call_key not in seen_calls:
            seen_calls.add(call_key)
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                OpenAIToolCall(
                    id=call_id,
                    type="function",
                    function=OpenAIToolCallFunction(name=tool_name, arguments=args_str),
                )
            )
        clean_text = clean_text.replace(match.group(0), "")

    # 清理残留空标签
    clean_text = re.sub(r'</?[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?[^>]*>', '', clean_text)

    return clean_text.strip(), tool_calls
