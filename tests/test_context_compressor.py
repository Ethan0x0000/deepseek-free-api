import pytest
from app.schemas.openai import OpenAIChatMessage
from app.services.context_compressor import ContextCompressor, estimate_tokens, truncate_tool_output


def test_estimate_tokens():
    # ASCII text
    ascii_text = "Hello world! This is a test."
    tokens_ascii = estimate_tokens(ascii_text)
    assert tokens_ascii > 0

    # CJK / Multi-language text
    cjk_text = "你好世界！这是一段中文测试文本。"
    tokens_cjk = estimate_tokens(cjk_text)
    assert tokens_cjk > 0
    assert tokens_cjk > len(cjk_text) / 2.5


def test_truncate_tool_output():
    short_output = "Short output"
    assert truncate_tool_output(short_output, max_tokens=100) == short_output

    huge_output = "A" * 50_000
    truncated = truncate_tool_output(huge_output, max_tokens=500)
    assert len(truncated) < len(huge_output)
    assert "Context compressed" in truncated


def test_compress_openai_messages():
    compressor = ContextCompressor(max_context_tokens=300, retain_recent_count=2)

    system_msg = OpenAIChatMessage(role="system", content="You are a helpful coding assistant.")
    
    # 构建超长历史消息
    messages = [system_msg]
    for i in range(20):
        messages.append(OpenAIChatMessage(role="user", content=f"User message number {i} " + "content " * 20))
        messages.append(OpenAIChatMessage(role="assistant", content=f"Assistant response number {i} " + "answer " * 20))

    compressed = compressor.compress_openai_messages(messages, max_tokens=300)

    # 1. 系统提示词完整保留
    assert compressed[0].role == "system"
    assert compressed[0].content == system_msg.content

    # 2. 中间历史生成摘要
    assert any("Summary of previous conversation context" in (m.content or "") for m in compressed)

    # 3. 最近两条消息完整保留
    assert compressed[-1].content == messages[-1].content
    assert compressed[-2].content == messages[-2].content


def test_compress_raw_prompt():
    compressor = ContextCompressor(max_context_tokens=100)
    short_prompt = "Short prompt"
    assert compressor.compress_raw_prompt(short_prompt, max_tokens=100) == short_prompt

    huge_prompt = "Header instructions:\n" + ("Data line details\n" * 500) + "\nFinal task: summarize everything."
    compressed = compressor.compress_raw_prompt(huge_prompt, max_tokens=50)

    assert "Context compressed" in compressed
    assert "Header instructions" in compressed
    assert "Final task" in compressed


def test_compress_raw_prompt_preserves_tools_and_instructions():
    """验证压缩对话历史时，工具定义与系统指令始终保持不被裁剪。"""
    compressor = ContextCompressor()

    tools_block = "# Available Tools\n```json\n" + ("x" * 20_000) + "\n```\n\n# Tool Call Instructions\nCRITICAL REQUIREMENT: invoke tool using <tool_call>.\n"
    sys_block = "System Instructions:\nYou are ZCode assistant.\n"
    history_block = "\nConversation History:\nUser: Help me\nAssistant: Sure\n" + ("Long message with large file contents and logs " * 2000)

    full_prompt = tools_block + "\n\n" + sys_block + history_block

    compressed = compressor.compress_raw_prompt(full_prompt, max_tokens=20_000, max_bytes=70_000)

    # 1. 工具指令完整保留
    assert "# Tool Call Instructions" in compressed
    assert "CRITICAL REQUIREMENT: invoke tool using <tool_call>." in compressed
    assert "```\n\n# Tool Call Instructions" in compressed
    assert "System Instructions:\nYou are ZCode assistant." in compressed

    # 2. 压缩仅在对话历史区间发生
    assert "Conversation History:" in compressed
    assert "Context compressed" in compressed

    # 3. 最终体积在安全限制内 (< 70 KB)
    assert len(compressed.encode("utf-8")) <= 70_000
