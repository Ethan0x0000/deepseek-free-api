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

