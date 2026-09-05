import json
import re
import uuid
from typing import List, Optional, Tuple, Any, Dict
from app.schemas.openai import OpenAIChatMessage, OpenAITool, OpenAIToolCall, OpenAIToolCallFunction


def compact_tool_schema(value: Any, is_root: bool = True) -> Any:
    """
    Компактизирует JSON Schema параметров инструмента:
    - Удаляет избыточные метаданные (title, $comment, verbose examples).
    - Сохраняет валидационную структуру (type, properties, required, enum, const, items).
    - Сокращает чрезмерно длинные описания параметров (>120 симв.).
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


def build_tool_system_prompt(tools: List[OpenAITool]) -> str:
    """Формирует системную инструкцию с описанием доступных инструментов."""
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

    prompt = f"""
# Available Tools
You have access to the following functions/tools to assist the user:

{tools_text}

# Tool Call Instructions
CRITICAL RULES FOR TOOL CALLS:
1. You ONLY REASON and REQUEST tool executions. You do NOT execute any commands or files yourself.
2. DO NOT STOP with just a text promise or declaration of intent (such as "Let me study...", "I will check...", "我来看看...", "让我检查一下...", "我来读取...").
   【严格禁令】当需要检查、读取、搜索、编辑文件或运行命令时，严禁仅用自然语言表达行动意向或口头承诺！你必须在【同一条回复中】立即输出 `<tool_call>` 标签调用工具！
3. NEVER simulate, guess, or fabricate command or tool output — output the tool call and wait for the actual result from the system.
4. When requesting a tool, output valid JSON inside `<tool_call>...</tool_call>`:
<tool_call>
{{"name": "<function_name>", "arguments": {{...}}}}
</tool_call>

Alternatively, standard JSON format is also accepted:
{{"tool_call": {{"name": "<function_name>", "arguments": {{...}}}}}}

5. EXAMPLES OF CORRECT BEHAVIOR:
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

6. CRITICAL RULES FOR FILE EDITING / WRITING:
When modifying or editing a file, ALWAYS invoke `Edit` or `Write` with valid JSON arguments!
NEVER output raw code or file paths directly inside `<tool_call>` without the JSON structure:
CORRECT:
<tool_call>
{{"name": "Edit", "arguments": {{"file_path": "path/to/file.py", "old_string": "exact old code", "new_string": "new code"}}}}
</tool_call>
or:
<tool_call>
{{"name": "Write", "arguments": {{"file_path": "path/to/file.py", "content": "full new content"}}}}
</tool_call>

FORBIDDEN (NEVER DO THIS):
<tool_call>
path/to/file.py
# code...
</tool_call>
-> WRONG! This is invalid and causes errors. Always format as valid JSON.

7. If no tool call is needed and the entire task is complete, provide your normal conversational response directly.
"""
    return prompt.strip()


def format_messages_to_prompt(
    messages: List[OpenAIChatMessage],
    tools: Optional[List[OpenAITool]] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Преобразует историю сообщений OpenAI (system, user, assistant, tool)
    в единый контекстный промпт для веб-интерфейса DeepSeek/Qwen.
    Автоматически применяет сжатие контекста при превышении лимита токенов.
    """
    from app.services.context_compressor import context_compressor
    compressed_messages = context_compressor.compress_openai_messages(messages, max_tokens=max_tokens)

    prompt_parts = []

    # 1. Если переданы tools, добавляем системную инструкцию по инструментам
    if tools:
        tool_instruction = build_tool_system_prompt(tools)
        prompt_parts.append(tool_instruction)

    # 2. Обрабатываем системные и пользовательские сообщения
    system_messages = []
    history_messages = []

    for msg in compressed_messages:
        role = msg.role
        content = msg.content or ""
        if isinstance(content, list):
            # Если переданы multipart сообщения (текст + изображения)
            text_pieces = []
            for piece in content:
                if isinstance(piece, dict) and piece.get("type") == "text":
                    text_pieces.append(piece.get("text", ""))
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

    # 3. Если последнее сообщение — вывод инструмента (tool output),
    # добавляем директиву немедленно продолжить и вызвать инструмент, а не останавливаться на обещании
    if compressed_messages and compressed_messages[-1].role in ["tool", "function"]:
        prompt_parts.append(
            "\n[System Directive: The previous tool execution has finished and its output is provided above. Proceed with the task immediately. If you need to inspect more files or run commands, invoke the tool call NOW: <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>. Do NOT stop with only a conversational promise or intent.]"
        )
    # 4. Если переданы tools и последнее сообщение от пользователя, добавляем хвостовую директиву вызова инструментов
    elif tools and compressed_messages and compressed_messages[-1].role == "user":
        prompt_parts.append(
            "\n[System Directive: Tools are available. If you need to inspect files, check directories, read code, or execute commands to answer the user, you MUST invoke the tool call in this turn using `<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>`. DO NOT stop with only a verbal promise or intent statement like '我来看看...' or 'Let me check...'.]"
        )

    full_prompt = "\n\n".join(prompt_parts)
    if max_tokens:
        full_prompt = context_compressor.compress_raw_prompt(full_prompt, max_tokens=max_tokens)
    return full_prompt


def normalize_qwen_parameter_tags(text: str) -> str:
    """Нормализует гибридные теги параметров Qwen (<parameter=key>...</parameter>) в валидный JSON."""
    if not text or "parameter" not in text:
        return text
    # 1. Замена перехода между параметрами: </parameter>\n<parameter=key> -> ", "key": 
    normalized = re.sub(r'\s*</parameter>\s*<parameter=([a-zA-Z0-9_\-]+)>\s*', r'", "\1": ', text)
    # 2. Замена одиночного </parameter> -> "
    normalized = re.sub(r'\s*</parameter>', r'"', normalized)
    # 3. Замена одиночного <parameter=key> -> "key": 
    normalized = re.sub(r'<parameter=([a-zA-Z0-9_\-]+)>\s*', r'"\1": ', normalized)
    return normalized


def _parse_broken_arguments(args_str: str) -> Dict[str, Any]:
    """
    Устойчивый парсер аргументов инструмента:
    - Восстанавливает JSON с неэкранированными внутренними кавычками (например, внутри shell-команд: echo "...", grep '...').
    - Извлекает ключи и значения через позиционный сплит пар.
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
    """Парсит все JSON-объекты (один или несколько параллельных) из блока tool_call."""
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

    # Fallback 1: стандартный json.loads
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

    # Fallback 2: экранируем сырые переносы строк
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

    # Fallback 3: парсим поврежденный JSON с неэкранированными внутренними кавычками
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
    Извлекает вызовы инструментов из ответа модели:
    - Поддерживает один или несколько JSON-объектов внутри одного тега <tool_call>...</tool_call>.
    - Поддерживает гибридные теги параметров Qwen (<parameter=key>...</parameter>).
    - Поддерживает теги <tool_call>...</tool_call> (включая <tool_call"> и с атрибутами).
    - Поддерживает markdown блоки ```tool_call...```.
    - Поддерживает нативный формат Qwen <function=name>...</function>.
    - Поддерживает многострочный код внутри аргументов (strict=False).
    - Поддерживает неформатированные вызовы Edit/Write с путем к файлу и кодом.
    - Выполняет дедупликацию идентичных вызовов.
    Возвращает (очищенный_текст, список_tool_calls).
    """
    tool_calls: List[OpenAIToolCall] = []
    seen_calls = set()
    clean_text = text

    # 0. Проверка формата DeepSeek DSML: <｜DSML｜tool_calls>...<｜DSML｜invoke name="...">...</｜DSML｜invoke>...</｜DSML｜tool_calls>
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

    # 1. Проверка формата Claude / Anthropic / DeepSeek: <invoke name="...">...</invoke>
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

    # Очищаем блоки invoke (включая если они обернуты в <tool_call>...<invoke>...</tool_calls>)
    clean_text = re.sub(r"<[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?[^>]*>\s*(?:<invoke\b.*?</invoke>\s*)+</[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?>", "", clean_text, flags=re.DOTALL)
    clean_text = re.sub(invoke_pat, "", clean_text, flags=re.DOTALL)

    # 2. Паттерны для поиска стандартных блоков JSON tool_call
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

    # 3. Проверка нативного формата Qwen: <function=name>args</function>
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

    # 4. Проверка "голого" JSON вызова инструмента без обрамляющих тегов (Naked JSON tool call)
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

    # Реверсивный порядок: {"arguments": ..., "name": "..."}
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

    # 5. Проверка неформатированного вызова Edit/Write файла без JSON:
    # <tool_call>\npath/to/file.py\ncode...\n</tool_call>
    raw_file_call_pat = re.compile(
        r'<[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?[^>]*>\s*([a-zA-Z]:[\\/][^ \r\n\t]+|[a-zA-Z0-9_\-\.\/]+\.[a-zA-Z0-9]+)\s+([\s\S]+?)(?:</[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?>|$)',
        re.DOTALL
    )
    for match in raw_file_call_pat.finditer(clean_text):
        fpath = match.group(1).strip()
        code_body = match.group(2).strip()

        # Пропускаем, если внутри уже есть валидный JSON
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

    # Окончательная подчистка оставшихся пустых тегов <tool_call>
    clean_text = re.sub(r'</?[｜\|]*\s*(?:DSML\s*[｜\|]*)?tool_calls?[^>]*>', '', clean_text)

    return clean_text.strip(), tool_calls
