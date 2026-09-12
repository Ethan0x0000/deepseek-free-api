import json
import re
import uuid
from typing import List, Optional, Tuple, Any, Dict, Union, Set
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
   When requesting a tool, you can emit DeepSeek standard DSML format (RECOMMENDED):
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="<function_name>">
<｜｜DSML｜｜ parameter name="<param_name>"><param_val></｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>

   Alternatively, standard JSON format is also fully supported:
<tool_call>
{{"name": "<function_name>", "arguments": {{...}}}}
</tool_call>

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

8. CRITICAL RULES FOR REASONING / THINKING MODELS (严禁在思考阶段输出工具调用):
- SEPARATION OF THOUGHT AND ACTION: In your internal thinking/reasoning process (the thought monologue), you must ONLY think, analyze, and plan. NEVER, UNDER ANY CIRCUMSTANCES, emit `<tool_call>` or `<|DSML|...>` tags inside the thinking phase!
- TOOL CALLS IN FINAL RESPONSE ONLY: All `<tool_call>` blocks MUST be emitted in your final assistant response AFTER thinking has completely finished.
- ZERO FALSE CLAIMS: Never claim, hallucinate, or pretend that you have already executed a command, created a directory, or completed an action unless you have actually received the tool execution output from the environment. If an action needs to be taken, output `<tool_call>` and STOP.

9. If no tool call is needed and the entire task is 100% complete, provide your normal conversational response directly.
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
                    try:
                        parsed_args = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                        while isinstance(parsed_args, dict) and len(parsed_args) == 1 and any(k in parsed_args for k in ["arguments", "parameters", "input"]):
                            nested = list(parsed_args.values())[0]
                            if isinstance(nested, dict):
                                parsed_args = nested
                            elif isinstance(nested, str):
                                try:
                                    parsed_args = json.loads(nested)
                                except Exception:
                                    break
                            else:
                                break
                        fn_args = json.dumps(parsed_args, ensure_ascii=False) if isinstance(parsed_args, dict) else str(fn_args)
                    except Exception:
                        pass
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
        last_tool_content = str(compressed_messages[-1].content or "")
        is_error = any(
            kw in last_tool_content.lower()
            for kw in ["error", "fail", "invalid", "not found", "exception", "exit status", "command not found"]
        )
        if is_error:
            prompt_parts.append(
                "\n[Autonomous Directive: ATTENTION - The previous tool call returned an ERROR or failure indicated above. "
                "Carefully inspect the error message. If it was a SchemaError, argument error, or type mismatch, you MUST correct your argument values and types (for example, pass numbers as raw unquoted numbers, booleans as true/false, or fix syntax). "
                "Do NOT repeat the exact same failing tool call without fixing the issue! "
                "Adjust your approach, fix the arguments, and invoke the corrected tool call or try an alternative solution: "
                "<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>. "
                "Plan in thought, emit <tool_call> ONLY in your final response, and solve the problem!]"
            )
        else:
            prompt_parts.append(
                "\n[Autonomous Directive: The tool execution result is provided above. Proceed with the task immediately. "
                "Analyze the output and invoke the next tool call in your final response if more investigation, code editing, or verification is needed: "
                "<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>. "
                "Plan in thought, but emit <tool_call> ONLY in your final response, NEVER inside thinking. "
                "DO NOT stop halfway with an intermediate conversational summary. Work relentlessly until the user's objective is 100% completed!]"
            )
    # 4. 如果提供了 tools 且最后一条是用户指令，注入行动优先指令
    elif tools and compressed_messages and compressed_messages[-1].role == "user":
        prompt_parts.append(
            "\n[Autonomous Directive: Tools are available. Action over words: If answering this request requires inspecting files, exploring directories, searching code, or executing commands, plan in thought, then invoke the tool call in your final response: <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>. "
            "DO NOT emit <tool_call> inside your thinking, and DO NOT respond with conversational promises or claim you completed an action without real tool execution.]"
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


def _parse_param_value(val_str: str, explicit_string: bool = False) -> Any:
    """
    智能解析参数值：
    - 当 explicit_string=True 时直接作为字符串保留
    - 否则自动恢复数字 (int/float)、布尔 (True/False)、null (None)、JSON 对象与数组等原生类型
    - 避免将 DSML 中的原生数字或布尔字面量粗暴当作字符串处理
    """
    if explicit_string or not isinstance(val_str, str):
        return val_str
    val = val_str.strip()
    if not val:
        return val_str

    # 尝试按 JSON 标准语法反序列化 (支持 int, float, bool, null, dict, list)
    try:
        return json.loads(val)
    except Exception:
        pass

    # 针对不带标准引号但显然是布尔或空值的情况容错
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None

    # 尝试解析带符号或纯数字
    try:
        if "." in val or "e" in low:
            return float(val)
        return int(val)
    except ValueError:
        pass

    return val_str


def get_expected_schema_types(prop_def: Any) -> Set[str]:
    """从参数的 JSON Schema 属性定义中提取期望的数据类型集合。"""
    types: Set[str] = set()
    if not isinstance(prop_def, dict):
        return types
    t = prop_def.get("type")
    if isinstance(t, str):
        types.add(t.lower())
    elif isinstance(t, list):
        types.update(str(x).lower() for x in t)
    for branch in prop_def.get("anyOf", []) + prop_def.get("oneOf", []):
        if isinstance(branch, dict) and "type" in branch:
            bt = branch["type"]
            if isinstance(bt, str):
                types.add(bt.lower())
            elif isinstance(bt, list):
                types.update(str(x).lower() for x in bt)
    return types


def coerce_args_to_schema(
    args_dict: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    依据工具的 JSON Schema 参数定义，对传入的参数值进行自动类型校准与自愈转换：
    - 将字符串数字 ("300000") 纠正为数值类型 (300000)
    - 将字符串布尔 ("true"/"false") 纠正为布尔值 (True/False)
    - 将嵌套 JSON 字符串纠正为字典或列表
    - 彻底杜绝客户端 (如 OpenCode) 因 Schema 强类型校验抛出 SchemaError 导致的无限重试循环
    """
    if not schema or not isinstance(schema, dict) or not isinstance(args_dict, dict):
        return args_dict
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        return args_dict

    coerced = dict(args_dict)
    for k, v in list(coerced.items()):
        if k not in props:
            continue
        prop_def = props[k]
        expected = get_expected_schema_types(prop_def)
        if not expected:
            continue

        # 1. 期望 number / integer，而当前是字符串
        if "number" in expected or "integer" in expected:
            if isinstance(v, str):
                s = v.strip()
                try:
                    if "." in s or "e" in s.lower():
                        val = float(s)
                        if "integer" in expected and "number" not in expected:
                            val = int(val)
                    else:
                        val = int(s) if "integer" in expected or "." not in s else float(s)
                    coerced[k] = val
                except ValueError:
                    pass

        # 2. 期望 boolean，而当前是字符串或 0/1
        elif "boolean" in expected:
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("true", "1"):
                    coerced[k] = True
                elif s in ("false", "0"):
                    coerced[k] = False
            elif isinstance(v, (int, float)) and v in (0, 1):
                coerced[k] = bool(v)

        # 3. 期望 string，而当前是标量数字/布尔
        elif "string" in expected:
            if not isinstance(v, str) and v is not None:
                if not isinstance(v, (dict, list)):
                    coerced[k] = str(v)

        # 4. 期望 array，而当前是 JSON 字符串
        elif "array" in expected:
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        coerced[k] = json.loads(s)
                    except Exception:
                        pass

        # 5. 期望 object，而当前是 JSON 字符串
        elif "object" in expected:
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("{") and s.endswith("}"):
                    try:
                        coerced[k] = json.loads(s)
                    except Exception:
                        pass

    return coerced


def _parse_broken_arguments(args_str: str) -> Dict[str, Any]:
    """
    容错参数解析器：
    - 修复带有未转义内部引号的 JSON (如 shell 命令里的 echo "...", grep '...')
    - 支持通过键值对位置切分提取字段
    - 自动保留裸露数字、布尔等字面量的原生数据类型
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

        # 清理由于多层闭合失衡导致的多余尾随反花括号 (例如 300000}})
        while raw_val.endswith("}") and raw_val.count("}") > raw_val.count("{"):
            raw_val = raw_val[:-1].strip()

        has_quotes = False
        if raw_val.startswith('"') and raw_val.endswith('"') and len(raw_val) >= 2:
            raw_val = raw_val[1:-1]
            has_quotes = True
        elif raw_val.startswith('"'):
            raw_val = raw_val[1:]
            has_quotes = True
        elif raw_val.endswith('"'):
            raw_val = raw_val[:-1]
            has_quotes = True

        if not has_quotes:
            result[key] = _parse_param_value(raw_val)
        else:
            if (raw_val.startswith("{") and raw_val.endswith("}")) or (raw_val.startswith("[") and raw_val.endswith("]")):
                try:
                    raw_val = json.loads(raw_val)
                except Exception:
                    pass
            result[key] = raw_val

    return result


def _parse_all_tool_json(raw_json: str, default_name: Optional[str] = None) -> List[Tuple[str, str]]:
    """解析 tool_call 块内的所有 JSON 对象 (支持单对象、并行多对象以及外置 name 属性模式)。"""
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
                name = obj.get("name") or obj.get("function") or default_name
                args = obj.get("arguments") or obj.get("parameters") or obj.get("input")
                # 兼容扁平结构: {"name": "bash", "command": "...", "timeout": 120000}
                if args is None:
                    args = {k: v for k, v in obj.items() if k not in ["name", "function", "type", "id"]}
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
                name = data.get("name") or data.get("function") or default_name
                args = data.get("arguments") or data.get("parameters") or data.get("input")
                if args is None:
                    args = {k: v for k, v in data.items() if k not in ["name", "function", "type", "id"]}
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
                name = data.get("name") or data.get("function") or default_name
                args = data.get("arguments") or data.get("parameters") or data.get("input")
                if args is None:
                    args = {k: v for k, v in data.items() if k not in ["name", "function", "type", "id"]}
                if name:
                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    results.append((str(name).strip(), args_str))
        except Exception:
            pass

    # 回退 3: 正则提取被破坏的内部引号 JSON
    if not results:
        name_match = re.search(r'"(?:name|function)"\s*:\s*"([a-zA-Z0-9_\-\.]+)"', s)
        name = (name_match.group(1).strip() if name_match else None) or default_name
        if name:
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
            elif default_name and s.startswith("{") and s.endswith("}"):
                args_dict = _parse_broken_arguments(s)
                if args_dict:
                    results.append((name, json.dumps(args_dict, ensure_ascii=False)))

    return results

    return results


def extract_tool_calls(
    text: str,
    allowed_tool_names: Optional[Set[str]] = None,
    tools_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[str, List[OpenAIToolCall]]:
    """
    从模型输出内容中提取工具调用 (Tool Calls)：
    - 支持单个或多个 JSON 对象封装在 <tool_call>...</tool_call> 内
    - 支持 DeepSeek 官方 DSML 格式 (<|DSML|invoke name="...">...</|DSML|invoke>)
    - 支持 Anthropic/Claude 格式 (<invoke name="...">...</invoke>)
    - 支持 Qwen 原生标签格式 (<function=name>...</function>)
    - 支持 Markdown 代码块语法 (```tool_call...```)
    - 支持基于 tools_schemas 的参数强类型自动矫正 (Coercion)，杜绝客户端 SchemaError
    - 可选通过 allowed_tool_names 过滤仅属于当前请求的合法工具，避免将文档说明或伪代码误识别为工具调用
    - 自动去重相同调用并返回 (清洗后的正文文本, 工具调用列表)
    """
    tool_calls: List[OpenAIToolCall] = []
    seen_calls = set()
    clean_text = text

    def add_tool_call(name: str, args_input: Any) -> bool:
        name = str(name).strip()
        if allowed_tool_names is not None and name not in allowed_tool_names:
            return False

        if isinstance(args_input, dict):
            args_dict = dict(args_input)
        elif isinstance(args_input, str):
            try:
                parsed = json.loads(args_input, strict=False)
                args_dict = parsed if isinstance(parsed, dict) else _parse_broken_arguments(args_input)
            except Exception:
                args_dict = _parse_broken_arguments(args_input) if args_input else {}
        else:
            args_dict = {}

        # 核心防线：递归解包被嵌套包裹在 arguments/parameters/input 内的参数 (无论内层是 dict 还是序列化 JSON 字符串)
        while len(args_dict) == 1 and any(k in args_dict for k in ["arguments", "parameters", "input"]):
            nested = list(args_dict.values())[0]
            if isinstance(nested, dict):
                args_dict = nested
            elif isinstance(nested, str):
                try:
                    p = json.loads(nested, strict=False)
                    if isinstance(p, dict):
                        args_dict = p
                    else:
                        args_dict = _parse_broken_arguments(nested)
                except Exception:
                    args_dict = _parse_broken_arguments(nested)
                break
            else:
                break

        # 依据 Schema 进行类型自动矫正 (Coercion)，如将 "300000" 纠正为数值 300000
        schema = tools_schemas.get(name) if tools_schemas else None
        if schema:
            args_dict = coerce_args_to_schema(args_dict, schema)

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
            return True
        return False

    # 0. 检验 DeepSeek 原生 DSML 与 Claude/Anthropic 标签格式 (<｜｜DSML｜｜ invoke ...> / <invoke ...>)
    dsml_invoke_pat = re.compile(
        r"<[｜\|]*\s*(?:DSML\s*[｜\|]*\s*)?invoke\b([^>]*)>(.*?)</[｜\|]*\s*(?:DSML\s*[｜\|]*\s*)?invoke>",
        re.DOTALL | re.IGNORECASE,
    )
    dsml_param_pat = re.compile(
        r"<[｜\|]*\s*(?:DSML\s*[｜\|]*\s*)?parameter\b([^>]*)>(.*?)</[｜\|]*\s*(?:DSML\s*[｜\|]*\s*)?parameter>|<parameter\s*=\s*[\"']?([a-zA-Z0-9_\-]+)[\"']?[^>]*>(.*?)</parameter>",
        re.DOTALL | re.IGNORECASE,
    )
    hybrid_dsml_pat = re.compile(
        r"<[｜\|]*\s*(?:DSML\s*[｜\|]*\s*)?(?:tool_calls?|calls)[^>]*>\s*"
        r"<[｜\|]*\s*DSML\s*[｜\|]*\s*parameter\s+[^>]*name=[\"']?name[\"']?[^>]*>(.*?)</[｜\|]*\s*DSML\s*[｜\|]*\s*parameter>\s*"
        r"<[｜\|]*\s*DSML\s*[｜\|]*\s*parameter\s+[^>]*name=[\"']?(?:arguments|input|parameters)[\"']?[^>]*>(.*?)"
        r"(?:</[｜\|]*\s*DSML\s*[｜\|]*\s*parameter>|\s*</[｜\|]*\s*DSML\s*[｜\|]*\s*invoke>|\s*</(?:tool_calls?|calls)>|$)",
        re.DOTALL | re.IGNORECASE,
    )

    # 0.1 优先匹配 Hybrid DSML
    for match in hybrid_dsml_pat.finditer(text):
        name = match.group(1).strip()
        args_raw = match.group(2).strip()
        add_tool_call(name, args_raw)

    # 0.2 统一匹配 DSML 及 Anthropic invoke 块
    for match in dsml_invoke_pat.finditer(text):
        attrs = match.group(1) or ""
        body = match.group(2).strip()
        name_match = re.search(r'\bname=[\"\']?([^\"\'\s>]+)[\"\']?', attrs)
        if not name_match:
            continue
        name = name_match.group(1).strip()
        if allowed_tool_names is not None and name not in allowed_tool_names:
            continue

        args_dict = {}
        found_params = False
        for pm in dsml_param_pat.finditer(body):
            found_params = True
            if pm.group(3) is not None:
                p_name = pm.group(3).strip()
                p_val = pm.group(4).strip()
                p_attrs = ""
            else:
                p_attrs = pm.group(1) or ""
                p_val = pm.group(2).strip()
                p_name_match = re.search(r'\bname=[\"\']?([^\"\'\s>]+)[\"\']?|=([a-zA-Z0-9_\-]+)', p_attrs)
                p_name = (p_name_match.group(1) or p_name_match.group(2)).strip() if p_name_match else ""

            if not p_name:
                continue

            # 解包 CDATA: <![CDATA[...]]>
            cdata_match = re.search(r'<!\[CDATA\[([\s\S]*?)\]\]>', p_val)
            if cdata_match:
                p_val = cdata_match.group(1)

            is_explicit_str = bool(re.search(r'\bstring=[\"\']?true[\"\']?', p_attrs, re.IGNORECASE))
            args_dict[p_name] = _parse_param_value(p_val, explicit_string=is_explicit_str)

        # 容错：如果 body 内部没有 parameter 标签，但有 JSON 对象
        if not found_params and body:
            parsed_args = _parse_broken_arguments(body)
            if parsed_args:
                args_dict = parsed_args

        # 如果只有一个 arguments/parameters/input 字典，自动解包展开
        if len(args_dict) == 1 and any(k in args_dict for k in ["arguments", "parameters", "input"]) and isinstance(list(args_dict.values())[0], dict):
            args_dict = list(args_dict.values())[0]

        add_tool_call(name, args_dict)

    clean_text = re.sub(
        r"<[｜\|]*\s*(?:DSML\s*[｜\|]*\s*)?(?:calls|tool_calls?|function_calls?)[^>]*>.*?</[｜\|]*\s*(?:DSML\s*[｜\|]*\s*)?(?:calls|tool_calls?|function_calls?)[^>]*>",
        "",
        clean_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    clean_text = re.sub(
        r"<[｜\|]*\s*(?:DSML\s*[｜\|]*\s*)?invoke\b[^>]*>.*?</[｜\|]*\s*(?:DSML\s*[｜\|]*\s*)?invoke>",
        "",
        clean_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    clean_text = re.sub(hybrid_dsml_pat, "", clean_text)
    clean_text = re.sub(r"</?[｜\|]*\s*DSML\s*[｜\|]*[^>]*>", "", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"</?[｜\|]*\s*(?:calls|tool_calls?|function_calls?|invoke)\b[^>]*>", "", clean_text, flags=re.IGNORECASE)

    # 2. 匹配标准 JSON tool_call / function_call 块 (支持在标签属性或后缀中指定 name="..." 或 :name)
    tag_pat = r"<[｜\|]*\s*(?:DSML\s*[｜\|]*)?(?:tool_calls?|function_calls?|tool)(?:\s+[^>]*?name=[\"']?([^\"'>\s]+)[\"']?|:([a-zA-Z0-9_\-\.]+))?[^>]*>\s*(.*?)\s*</[｜\|]*\s*(?:DSML\s*[｜\|]*)?(?:tool_calls?|function_calls?|tool)[^>]*>"
    md_pat = r"```(?:tool_call|tool_calls|function_call)(?::([a-zA-Z0-9_\-\.]+))?\s*(.*?)\s*```"

    for match in re.finditer(tag_pat, text, re.DOTALL):
        def_name = match.group(1) or match.group(2)
        raw_content = match.group(3)
        parsed_list = _parse_all_tool_json(raw_content, default_name=def_name)
        found_valid = False
        for name, args_str in parsed_list:
            if add_tool_call(name, args_str):
                found_valid = True
        if found_valid:
            clean_text = clean_text.replace(match.group(0), "")

    for match in re.finditer(md_pat, text, re.DOTALL):
        def_name = match.group(1)
        raw_content = match.group(2)
        parsed_list = _parse_all_tool_json(raw_content, default_name=def_name)
        found_valid = False
        for name, args_str in parsed_list:
            if add_tool_call(name, args_str):
                found_valid = True
        if found_valid:
            clean_text = clean_text.replace(match.group(0), "")

    # 3. 匹配 Qwen 原生格式: <function=name>args</function>
    func_pat = r"<function=([a-zA-Z0-9_\-\.]+)[^>]*>\s*(.*?)\s*</function>"
    for match in re.finditer(func_pat, text, re.DOTALL):
        name = match.group(1).strip()
        raw_args = match.group(2).strip()
        args_str = raw_args
        if "<parameter" in raw_args:
            param_dict = {}
            for p in re.finditer(r'<parameter=([a-zA-Z0-9_\-]+)>\s*(.*?)\s*(?:</parameter>|$)', raw_args, re.DOTALL):
                param_dict[p.group(1)] = _parse_param_value(p.group(2).strip().strip("\"'"))
            add_tool_call(name, param_dict)
        else:
            add_tool_call(name, raw_args)
        clean_text = clean_text.replace(match.group(0), "")

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
        if add_tool_call(name, args_dict):
            clean_text = clean_text.replace(block, "")

    # 反序结构兼容: {"arguments": ..., "name": "..."}
    naked_rev_pat = re.compile(
        r'\{\s*"(?:arguments|parameters|input)"\s*:\s*(\{.*?\}).*?,\s*"(?:name|function)"\s*:\s*"([a-zA-Z0-9_\-\.]+)"\s*\}',
        re.DOTALL
    )
    for match in naked_rev_pat.finditer(clean_text):
        name = match.group(2).strip()
        args_raw = match.group(1).strip()
        block = match.group(0)

        args_dict = _parse_broken_arguments(args_raw)
        if add_tool_call(name, args_dict):
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

        tool_name = "Edit" if is_edit else "Write"
        args_obj = {"file_path": fpath, "old_string": old_str, "new_string": new_str} if is_edit else {"file_path": fpath, "content": code_body}
        if add_tool_call(tool_name, args_obj):
            clean_text = clean_text.replace(match.group(0), "")

    # 仅当实际提取出工具调用时清理外围空标签
    if tool_calls:
        clean_text = re.sub(r'</?[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?[^>]*>', '', clean_text)

    return clean_text.strip(), tool_calls
