import json
import pytest
from httpx import AsyncClient, ASGITransport
import httpx

from app.main import app
from app.providers.registry import provider_registry


@pytest.mark.asyncio
async def test_openai_chat_completions_model_routing():
    """测试不同模型前缀向提供商的正确路由。"""
    captured_providers = []

    orig_resolve = provider_registry.resolve_provider_for_model
    def mock_resolve(model_name: str):
        p = orig_resolve(model_name)
        captured_providers.append((model_name, p.provider_id))
        return p

    provider_registry.resolve_provider_for_model = mock_resolve

    deepseek_provider = provider_registry.resolve_provider_for_model("deepseek-v4-pro")
    assert deepseek_provider.provider_id == "deepseek"

    qwen_provider = provider_registry.resolve_provider_for_model("qwen3.7-plus")
    assert qwen_provider.provider_id == "qwen"

    qwen_coder = provider_registry.resolve_provider_for_model("qwen-3.8-coder")
    assert qwen_coder.provider_id == "qwen"

    provider_registry.resolve_provider_for_model = orig_resolve


def test_qwen_model_resolution():
    """测试 Qwen 模型别名映射。"""
    qwen_p = provider_registry.get_provider("qwen")
    assert qwen_p._resolve_qwen_model("qwen-3.8-coder") == "qwen3.8-max"
    assert qwen_p._resolve_qwen_model("qwen3.7-plus") == "qwen3.7-plus"
    assert qwen_p._resolve_qwen_model("qwen-3-max") == "qwen3.8-max"
    assert qwen_p._resolve_qwen_model("qwen-3-flash") == "qwen3.7-plus"


@pytest.mark.asyncio
async def test_list_models_contains_both_providers():
    """测试 /api/v1/models 与 /v1/models 包含各厂商模型。"""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        ids = [m["id"] for m in data["data"]]
        assert "deepseek-v4-pro" in ids
        assert "deepseek-reasoner" in ids
        assert "qwen3.7-plus" in ids
        assert "qwen-3.8-coder" in ids

        api_resp = await ac.get("/api/v1/models")
        assert api_resp.status_code == 200
        api_data = api_resp.json()
        api_ids = [m["id"] for m in api_data]
        assert "deepseek-v4-pro" in api_ids
        assert "qwen3.7-plus" in api_ids


@pytest.mark.asyncio
async def test_providers_switch_endpoint():
    """测试切换默认提供商接口。"""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/api/v1/providers/switch?provider_id=qwen")
        assert resp.status_code == 200
        assert provider_registry.default_provider_id == "qwen"

        # 还原回 deepseek
        await ac.post("/api/v1/providers/switch?provider_id=deepseek")
        assert provider_registry.default_provider_id == "deepseek"


@pytest.mark.asyncio
async def test_qwen_adaptive_context_compression():
    """验证 Qwen 的上下文自动受控于安全 WAF 限制内。"""
    from app.services.context_compressor import context_compressor, estimate_tokens
    from app.services.tool_parser import format_messages_to_prompt
    from app.schemas.openai import OpenAIChatMessage, OpenAITool, OpenAIToolFunction

    tools = [
        OpenAITool(
            type="function",
            function=OpenAIToolFunction(
                name=f"tool_{i}",
                description=f"Description of tool {i} for testing context size compression",
                parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
            )
        )
        for i in range(50)
    ]

    messages = [
        OpenAIChatMessage(role="system", content="You are a system assistant."),
        OpenAIChatMessage(role="user", content="Task instructions: " + ("Very long prompt content for testing " * 3000)),
    ]

    qwen_limit = context_compressor.get_limit_for_provider("qwen")
    assert qwen_limit == 20_000

    compiled = format_messages_to_prompt(messages, tools, max_tokens=qwen_limit)
    compiled_tokens = estimate_tokens(compiled)

    assert compiled_tokens <= qwen_limit * 1.5
    assert len(compiled.encode("utf-8")) < 80_000


@pytest.mark.asyncio
async def test_stream_error_sse_formatting(monkeypatch):
    """测试提供商错误在 SSE 流中以标准格式输出。"""
    from fastapi import HTTPException

    qwen_p = provider_registry.get_provider("qwen")

    async def mock_fail(*args, **kwargs):
        raise HTTPException(status_code=403, detail="WAF challenge error")
        yield

    monkeypatch.setattr(qwen_p, "stream_chat", mock_fail)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        req_payload = {
            "model": "qwen-3.8-coder",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        resp = await ac.post("/v1/chat/completions", json=req_payload)
        assert resp.status_code == 200
        text = resp.text
        assert "WAF challenge error" in text
        assert "data: [DONE]" in text


def test_robust_tool_call_extraction():
    """测试多行代码、畸变标签与重复去重的鲁棒工具调用解析。"""
    from app.services.tool_parser import extract_tool_calls

    raw_response = """
Here is the file:

<tool_call">
{"name": "write_to_file", "arguments": {"path": "calculator.py", "content": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"}}
</tool_call">

<tool_call>
{"name": "write_to_file", "arguments": {"path": "calculator.py", "content": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"}}
</tool_call>

<function=replace_in_file>
{"path": "main.py", "diff": "- old\n+ new"}
</function>
"""
    clean_text, calls = extract_tool_calls(raw_response)
    assert len(calls) == 2
    assert calls[0].function.name == "write_to_file"
    assert "calculator.py" in calls[0].function.arguments
    assert calls[1].function.name == "replace_in_file"
    assert "main.py" in calls[1].function.arguments
    assert "<tool_call" not in clean_text
    assert "<function=" not in clean_text


def test_multiple_json_in_single_tool_call_tag():
    """测试单个 <tool_call> 标签内包含多个 JSON 对象的并行调用解析。"""
    from app.services.tool_parser import extract_tool_calls

    raw_response = """Let me explore the directory.

<tool_call>
{"name": "Bash", "arguments": {"command": "find /tmp -type f | head -80", "description": "List files"}}
{"name": "Bash", "arguments": {"command": "ls -la /tmp", "description": "List root"}}
</tool_call>"""

    clean_text, calls = extract_tool_calls(raw_response)
    assert len(calls) == 2
    assert calls[0].function.name == "Bash"
    assert "find" in calls[0].function.arguments
    assert calls[1].function.name == "Bash"
    assert "ls -la" in calls[1].function.arguments
    assert clean_text == "Let me explore the directory."
    assert "<tool_call>" not in clean_text


def test_deepseek_claude_xml_invoke_extraction():
    """测试 Claude/DeepSeek XML 格式 (<invoke name=...>) 工具调用解析。"""
    from app.services.tool_parser import extract_tool_calls

    raw_response = """Let me start by exploring the project.

<tool_call>
<invoke name="Bash">
<parameter name="command">cd /workspace && git ls-files | head -200</parameter>
<parameter name="description">List tracked files</parameter>
</invoke>
</tool_calls>"""

    clean_text, calls = extract_tool_calls(raw_response)
    assert len(calls) == 1
    assert calls[0].function.name == "Bash"
    assert "cd /workspace" in calls[0].function.arguments
    assert clean_text == "Let me start by exploring the project."
    assert "<invoke" not in clean_text
    assert "<tool_call" not in clean_text


def test_deepseek_dsml_tool_calls_extraction():
    """测试 DeepSeek DSML 标签工具调用解析。"""
    from app.services.tool_parser import extract_tool_calls

    raw_response = """Let me explore the project.
<｜DSML｜tool_calls>
    <｜DSML｜invoke name="Bash">
        <｜DSML｜parameter name="command" string="true">git status</｜DSML｜parameter>
        <｜DSML｜parameter name="description" string="true">Check git status</｜DSML｜parameter>
    </｜DSML｜invoke>
</｜DSML｜tool_calls>"""

    clean_text, calls = extract_tool_calls(raw_response)
    assert len(calls) == 1
    assert calls[0].function.name == "Bash"
    assert "git status" in calls[0].function.arguments
    assert clean_text == "Let me explore the project."
    assert "DSML" not in clean_text
    assert "<｜" not in clean_text


def test_system_directive_after_tool_output():
    """测试工具执行结果返回后注入的自主 Agent 延续指令。"""
    from app.services.tool_parser import format_messages_to_prompt
    from app.schemas.openai import OpenAIChatMessage, OpenAITool, OpenAIToolFunction

    messages = [
        OpenAIChatMessage(role="user", content="Find files"),
        OpenAIChatMessage(role="assistant", content="Running search"),
        OpenAIChatMessage(role="tool", tool_call_id="call_1", content="file1.py\nfile2.py"),
    ]
    tools = [
        OpenAITool(
            type="function",
            function=OpenAIToolFunction(
                name="Bash",
                description="Run shell command",
                parameters={"type": "object", "properties": {"command": {"type": "string"}}},
            )
        )
    ]

    prompt = format_messages_to_prompt(messages, tools)
    assert "[Autonomous Directive:" in prompt
    assert "DO NOT stop halfway" in prompt


def test_intent_pattern_matching():
    """测试中英文行动意图正则表达式。"""
    from app.api.v1.endpoints.chat import INTENT_PAT

    sample1 = "我来看看当前文件夹下的文件结构。"
    sample2 = "Let me check the backend code to understand how endpoints are configured."
    sample3 = "Here is the completed output for your request. Have a nice day!"
    sample4 = "让我先检查一下 package.json 中的依赖配置。"

    assert INTENT_PAT.search(sample1) is not None
    assert INTENT_PAT.search(sample2) is not None
    assert INTENT_PAT.search(sample3) is None
    assert INTENT_PAT.search(sample4) is not None
