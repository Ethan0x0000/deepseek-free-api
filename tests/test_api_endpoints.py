import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.credentials import credentials_manager


@pytest.mark.asyncio
async def test_health_and_root():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "DeepSeek" in data["app"]

        health_resp = await ac.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_models_endpoints():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/models")
        assert resp.status_code == 200
        models = resp.json()
        model_ids = [m["id"] for m in models]
        assert "deepseek-chat" in model_ids
        assert "deepseek-reasoner" in model_ids
        assert "deepseek-search" in model_ids

        oai_resp = await ac.get("/v1/models")
        assert oai_resp.status_code == 200
        oai_data = oai_resp.json()
        assert oai_data["object"] == "list"
        model_names = [m["id"] for m in oai_data["data"]]
        assert "deepseek-flash" in model_names
        assert "deepseek-v4.1-flash" in model_names
        assert "deepseek-v4-pro" in model_names
        assert len(oai_data["data"]) >= 5


@pytest.mark.asyncio
async def test_tool_parser_extraction():
    from app.services.tool_parser import extract_tool_calls, format_messages_to_prompt
    from app.schemas.openai import OpenAIChatMessage, OpenAITool, OpenAIToolFunction

    mock_response = """Let me check the files.
<tool_call>
{"name": "list_files", "arguments": {"directory": "."}}
</tool_call>"""
    clean_text, tools = extract_tool_calls(mock_response)
    assert clean_text == "Let me check the files."
    assert len(tools) == 1
    assert tools[0].function.name == "list_files"
    assert '"directory": "."' in tools[0].function.arguments

    tools_def = [
        OpenAITool(
            type="function",
            function=OpenAIToolFunction(
                name="list_files",
                description="List files",
                parameters={"type": "object", "properties": {"directory": {"type": "string"}}},
            ),
        )
    ]
    msgs = [
        OpenAIChatMessage(role="system", content="You are a helpful assistant."),
        OpenAIChatMessage(role="user", content="List my files"),
    ]
    compiled = format_messages_to_prompt(msgs, tools_def)
    assert "Available Tools" in compiled
    assert "list_files" in compiled
    assert "User: List my files" in compiled


@pytest.mark.asyncio
async def test_native_dsml_extraction_suite():
    """全面测试 DeepSeek 最新基座模型的原生 DSML 工具调用解析能力。"""
    from app.services.tool_parser import extract_tool_calls

    # Case 1: 真实 OpenCode 会话中触发的原生全角 DSML 格式
    opencode_session_text = """我来为您查询明天南昌市红谷滩区的天气情况。
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="bash">
<｜｜DSML｜｜ parameter name="command" string="true">curl -s "https://wttr.in/Nanchang?format=j1" | head -c 4000</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>"""

    clean, tools = extract_tool_calls(opencode_session_text)
    assert clean == "我来为您查询明天南昌市红谷滩区的天气情况。"
    assert len(tools) == 1
    assert tools[0].function.name == "bash"
    assert "https://wttr.in/Nanchang" in tools[0].function.arguments

    # Case 2: 半角 DSML 格式且包含 CDATA
    cdata_dsml_text = """<|DSML| calls>
<|DSML| invoke name="edit">
<|DSML| parameter name="file_path">/app/test.py</|DSML| parameter>
<|DSML| parameter name="content"><![CDATA[print("hello\nworld")]]></|DSML| parameter>
</|DSML| invoke>
</|DSML| calls>"""

    clean2, tools2 = extract_tool_calls(cdata_dsml_text)
    assert clean2 == ""
    assert len(tools2) == 1
    assert tools2[0].function.name == "edit"
    assert "/app/test.py" in tools2[0].function.arguments
    assert "hello\\nworld" in tools2[0].function.arguments or "hello\nworld" in tools2[0].function.arguments

    # Case 3: 并行多工具调用
    multi_dsml_text = """<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="read_file">
<｜｜DSML｜｜ parameter name="file_path">a.txt</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ invoke name="read_file">
<｜｜DSML｜｜ parameter name="file_path">b.txt</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>"""

    clean3, tools3 = extract_tool_calls(multi_dsml_text)
    assert clean3 == ""
    assert len(tools3) == 2
    assert tools3[0].function.name == "read_file"
    assert tools3[1].function.name == "read_file"


@pytest.mark.asyncio
async def test_tool_type_coercion_and_error_recovery():
    """测试基于 Schema 的参数强类型矫正、DSML 原生数字解析与报错自愈提示。"""
    import json
    from app.services.tool_parser import extract_tool_calls, format_messages_to_prompt
    from app.schemas.openai import OpenAIChatMessage, OpenAITool, OpenAIToolFunction

    bash_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "number"},
            "background": {"type": "boolean"},
        },
        "required": ["command"],
    }
    tools_schemas = {"bash": bash_schema}

    # 1. 真实故障场景测试：JSON 包含未转义双引号且 timeout 为数字
    broken_json = """<tool_call>
{"name": "bash", "arguments": {"command": "git -c user.name=\"Ethan0x0000\" commit -m \"chore: build\"", "timeout": 300000}}
</tool_call>"""
    _, tools = extract_tool_calls(broken_json, tools_schemas=tools_schemas)
    assert len(tools) == 1
    args = json.loads(tools[0].function.arguments)
    assert args["timeout"] == 300000
    assert isinstance(args["timeout"], int)

    # 2. 真实故障场景测试：模型输出了字符串形式的数字，Schema 纠正自愈
    string_timeout_json = """<tool_call>
{"name": "bash", "arguments": {"command": "pnpm install", "timeout": "300000", "background": "true"}}
</tool_call>"""
    _, tools = extract_tool_calls(string_timeout_json, tools_schemas=tools_schemas)
    assert len(tools) == 1
    args = json.loads(tools[0].function.arguments)
    assert args["timeout"] == 300000
    assert isinstance(args["timeout"], int)
    assert args["background"] is True

    # 3. DSML 场景测试：DSML 中原生输出数字与布尔
    dsml_numeric = """<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="bash">
<｜｜DSML｜｜ parameter name="command">pnpm test</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="timeout">300000</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="background">false</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>"""
    _, tools = extract_tool_calls(dsml_numeric, tools_schemas=tools_schemas)
    assert len(tools) == 1
    args = json.loads(tools[0].function.arguments)
    assert args["timeout"] == 300000
    assert isinstance(args["timeout"], int)
    assert args["background"] is False

    # 4. 上条工具执行报错时的自愈引导提示
    tools_def = [
        OpenAITool(
            type="function",
            function=OpenAIToolFunction(
                name="bash",
                description="Run command",
                parameters=bash_schema,
            ),
        )
    ]
    error_msgs = [
        OpenAIChatMessage(role="user", content="run build"),
        OpenAIChatMessage(
            role="assistant",
            content="",
            tool_calls=tools,
        ),
        OpenAIChatMessage(
            role="tool",
            tool_call_id="call_12345",
            content='The bash tool was called with invalid arguments: SchemaError(Expected number, got "300000" at ["timeout"])',
        ),
    ]
    compiled = format_messages_to_prompt(error_msgs, tools_def)
    assert "ATTENTION - The previous tool call returned an ERROR" in compiled
    assert "SchemaError" in compiled
    assert "correct your argument values and types" in compiled


@pytest.mark.asyncio
async def test_auth_token_set(monkeypatch, tmp_path):
    fake_proj = tmp_path / "credentials.json"
    fake_user = tmp_path / "user_credentials.json"
    fake_env = tmp_path / ".env"
    monkeypatch.setattr(credentials_manager, "project_file", fake_proj)
    monkeypatch.setattr(credentials_manager, "user_file", fake_user)
    monkeypatch.setattr(credentials_manager, "env_file", fake_env)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        test_token = "test_temporary_token_1234567890"
        resp = await ac.post("/api/v1/auth/token", json={"token": test_token})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

        status_resp = await ac.get("/api/v1/auth/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["authenticated"] is True


@pytest.mark.asyncio
async def test_thinking_tool_call_recovery():
    """测试当模型把 tool_call 误输出在思考过程 (Thinking) 中时，服务端能够成功拯救并转为 tool_calls。"""
    from app.services.tool_parser import extract_tool_calls

    thinking_with_tool = """我们开始执行。先创建目录，然后安装。
我们执行：
<tool_call>
{"name": "shell", "arguments": {"command": "mkdir -p ~/.agents/skills", "workdir": "/Users/ethan"}}
</tool_call>"""

    clean_text, tools = extract_tool_calls(thinking_with_tool, allowed_tool_names={"shell"})
    assert len(tools) == 1
    assert tools[0].function.name == "shell"
    assert "mkdir -p ~/.agents/skills" in tools[0].function.arguments
    assert "<tool_call>" not in clean_text

