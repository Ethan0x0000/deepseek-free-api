import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.anthropic_converter import (
    convert_anthropic_request_to_deepseek,
    convert_deepseek_response_to_anthropic,
)
from app.schemas.anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessage,
    AnthropicTool,
    AnthropicThinkingConfig,
)
from app.schemas.chat import DeepSeekChatResponse


def test_anthropic_converter_basic():
    req = AnthropicMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        system="You are Claude, a helpful AI assistant.",
        messages=[
            AnthropicMessage(role="user", content="Hello world"),
        ],
        tools=[
            AnthropicTool(
                name="get_weather",
                description="Get weather for city",
                input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ],
        thinking=AnthropicThinkingConfig(type="enabled"),
    )

    deepseek_req, has_tools = convert_anthropic_request_to_deepseek(req)
    assert has_tools is True
    assert "System Instructions:" in deepseek_req.prompt
    assert "You are Claude" in deepseek_req.prompt
    assert "Available Tools" in deepseek_req.prompt
    assert "get_weather" in deepseek_req.prompt
    assert "User: Hello world" in deepseek_req.prompt
    assert deepseek_req.thinking_enabled is True


def test_anthropic_response_with_tools_and_thinking():
    mock_resp = DeepSeekChatResponse(
        content="Let me check the weather.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Moscow\"}}\n</tool_call>",
        thinking="User wants the weather for Moscow.",
        session_id="test-sess",
        message_id=10,
        token_usage=120,
    )

    anthropic_resp = convert_deepseek_response_to_anthropic(mock_resp, model="deepseek-v4-pro", has_tools=True)
    assert anthropic_resp.type == "message"
    assert anthropic_resp.stop_reason == "tool_use"
    assert anthropic_resp.usage.output_tokens == 120

    # 验证 content 内容块: 发生 tool_use 时应严格包含工具调用块
    types = [b.type for b in anthropic_resp.content]
    assert "tool_use" in types
    assert "thinking" not in types

    tool_block = next(b for b in anthropic_resp.content if b.type == "tool_use")
    assert tool_block.name == "get_weather"
    assert tool_block.input == {"city": "Moscow"}
    assert tool_block.id.startswith("toolu_")


def test_anthropic_response_with_thinking_no_tools():
    """测试不含工具调用的纯思考链与正文回答。"""
    mock_resp = DeepSeekChatResponse(
        content="The weather in Moscow is 15C.",
        thinking="Checking weather facts.",
        session_id="test-sess",
        message_id=10,
        token_usage=120,
    )
    anthropic_resp = convert_deepseek_response_to_anthropic(mock_resp, model="deepseek-v4-pro", has_tools=False)
    types = [b.type for b in anthropic_resp.content]
    assert "thinking" in types
    assert "text" in types
    assert anthropic_resp.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_anthropic_endpoint_validation():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 空请求校验
        resp = await ac.post("/v1/messages", json={"messages": []})
        assert resp.status_code == 400 or resp.status_code == 422
