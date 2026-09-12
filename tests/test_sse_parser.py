import pytest
from app.services.sse_parser import SSEParser, parse_sse_stream, parse_sse_lines


@pytest.mark.asyncio
async def test_sse_parser_user_example():
    """测试标准 DeepSeek SSE 流事件解析。"""
    raw_events = [
        b"event: ready\r\ndata: {\"request_message_id\":1,\"response_message_id\":2,\"model_type\":\"expert\"}\r\n\r\n",
        b"event: update_session\r\ndata: {\"updated_at\":1788251660.8037179}\r\n\r\n",
        b"data: {\"v\":{\"response\":{\"message_id\":2,\"parent_id\":1,\"model\":\"\",\"role\":\"ASSISTANT\",\"thinking_enabled\":false,\"ban_edit\":false,\"ban_regenerate\":false,\"status\":\"WIP\",\"incomplete_message\":null,\"accumulated_token_usage\":0,\"feedback\":null,\"inserted_at\":1788251660.7872858,\"search_enabled\":false,\"fragments\":[{\"id\":2,\"type\":\"RESPONSE\",\"content\":\"Hello\",\"references\":[],\"stage_id\":1}],\"conversation_mode\":\"DEFAULT\",\"has_pending_fragment\":false,\"auto_continue\":false,\"search_triggered\":false}}}\r\n\r\n",
        b"data: {\"p\":\"response/fragments/-1/content\",\"o\":\"APPEND\",\"v\":\", world\"}\r\n\r\n",
        b"data: {\"v\":\"!\"}\r\n\r\n",
        b"data: {\"p\":\"response\",\"o\":\"BATCH\",\"v\":[{\"p\":\"accumulated_token_usage\",\"v\":46},{\"p\":\"quasi_status\",\"v\":\"FINISHED\"}]}\r\n\r\n",
        b"data: {\"p\":\"response/status\",\"o\":\"SET\",\"v\":\"FINISHED\"}\r\n\r\n",
        b"event: title\r\ndata: {\"content\":\"Greeting\"}\r\n\r\n",
        b"event: close\r\ndata: {\"click_behavior\":\"none\",\"auto_resume\":false}\r\n\r\n",
    ]

    async def byte_generator():
        for b in raw_events:
            yield b

    collected_content = []
    collected_types = []
    token_usage = None

    async for chunk in parse_sse_stream(byte_generator(), session_id="test-session"):
        collected_types.append(chunk.type)
        if chunk.type == "content":
            collected_content.append(chunk.text)
        if chunk.token_usage:
            token_usage = chunk.token_usage

    full_text = "".join(collected_content)
    assert "Hello, world!" in full_text
    assert token_usage == 46
    assert "status" in collected_types
    assert "title" in collected_types


@pytest.mark.asyncio
async def test_sse_parser_thinking_and_response():
    """测试思考链与回答正文的清晰分离。"""
    raw_events = [
        b'event: ready\r\ndata: {"request_message_id":1,"response_message_id":2,"model_type":"expert"}\r\n\r\n',
        b'data: {"v":{"response":{"message_id":2,"parent_id":1,"model":"","role":"ASSISTANT","thinking_enabled":true,"fragments":[{"id":1,"type":"THINKING","content":"First thought. "}]}}}\r\n\r\n',
        b'data: {"p":"response/fragments/-1/content","o":"APPEND","v":"Second thought."}\r\n\r\n',
        b'data: {"p":"response/fragments","o":"APPEND","v":{"id":2,"type":"RESPONSE","content":"Final "}}\r\n\r\n',
        b'data: {"p":"response/fragments/-1/content","o":"APPEND","v":"answer."}\r\n\r\n',
        b'data: {"p":"response/status","o":"SET","v":"FINISHED"}\r\n\r\n',
        b"event: close\r\ndata: {}\r\n\r\n",
    ]

    async def gen():
        for b in raw_events:
            yield b

    thinking = []
    content = []
    async for chunk in parse_sse_stream(gen(), session_id="test-session"):
        if chunk.type == "thinking":
            thinking.append(chunk.text)
        elif chunk.type == "content":
            content.append(chunk.text)

    assert "".join(thinking) == "First thought. Second thought."
    assert "".join(content) == "Final answer."


@pytest.mark.asyncio
async def test_sse_parser_batch_fragment_switch():
    """测试通过 BATCH 操作从思考链切换为最终正文。"""
    raw_events = [
        b'event: ready\r\ndata: {"request_message_id":1,"response_message_id":2,"model_type":"expert"}\r\n\r\n',
        b'data: {"v":{"response":{"message_id":2,"parent_id":1,"model":"","role":"ASSISTANT","thinking_enabled":true,"fragments":[{"id":1,"type":"THINKING","content":"Thinking..."}]}}}\r\n\r\n',
        b'data: {"p":"response","o":"BATCH","v":[{"p":"fragments/0/status","o":"SET","v":"FINISHED"},{"p":"fragments","o":"APPEND","v":{"id":2,"type":"RESPONSE","content":"Hello world"}}]}\r\n\r\n',
        b'data: {"v":"! How can I help?"}\r\n\r\n',
        b'data: {"p":"response/status","o":"SET","v":"FINISHED"}\r\n\r\n',
        b"event: close\r\ndata: {}\r\n\r\n",
    ]

    async def gen():
        for b in raw_events:
            yield b

    thinking = []
    content = []
    async for chunk in parse_sse_stream(gen(), session_id="test-session"):
        if chunk.type == "thinking":
            thinking.append(chunk.text)
        elif chunk.type == "content":
            content.append(chunk.text)

    assert "".join(thinking) == "Thinking..."
    assert "".join(content) == "Hello world! How can I help?"


@pytest.mark.asyncio
async def test_sse_parser_real_deepseek_dump():
    """测试真实 DeepSeek-R1 响应数据结构解析。"""
    lines = [
        'event: ready',
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"expert"}',
        'data: {"v":{"response":{"message_id":2,"parent_id":1,"model":"","role":"ASSISTANT","thinking_enabled":true,"fragments":[{"id":1,"type":"THINKING","content":"We calculate 2+2"}]}}}',
        'data: {"p":"response/fragments/0/content","o":"APPEND","v":". The answer is 4."}',
        'data: {"p":"response","o":"BATCH","v":[{"p":"fragments/0/status","o":"SET","v":"FINISHED"},{"p":"fragments","o":"APPEND","v":{"id":2,"type":"RESPONSE","content":"4"}}]}',
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}',
        'event: close',
        'data: {}',
    ]

    async def gen_lines():
        for l in lines:
            yield l

    thinking = []
    content = []
    async for chunk in parse_sse_lines(gen_lines(), session_id="test-session"):
        if chunk.type == "thinking":
            thinking.append(chunk.text)
        elif chunk.type == "content":
            content.append(chunk.text)

    th = "".join(thinking)
    ct = "".join(content)
    assert "We calculate" in th
    assert ct == "4"


@pytest.mark.asyncio
async def test_sse_parser_hint_error():
    """测试服务端 hint 报错检测。"""
    lines = [
        'event: ready',
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"expert"}',
        'event: hint',
        'data: {"type":"error","content":"Text too long. Please shorten it.","clear_response":true,"finish_reason":"input_exceeds_limit"}',
        'event: close',
        'data: {"click_behavior":"none","auto_resume":false}',
    ]

    async def gen_lines():
        for l in lines:
            yield l

    chunks = []
    async for chunk in parse_sse_lines(gen_lines(), session_id="test-session"):
        chunks.append(chunk)

    error_chunks = [c for c in chunks if c.type == "error"]
    assert len(error_chunks) == 1
    assert "input_exceeds_limit" in error_chunks[0].text or "Text too long" in error_chunks[0].text


@pytest.mark.asyncio
async def test_sse_parser_batch_multipart_never_drops():
    """测试当单条 BATCH 操作中包含多个 content/thinking 补丁时，所有分块均被完整保留而不被截断吞没。"""
    lines = [
        'event: ready',
        'data: {"request_message_id":1,"response_message_id":2,"model_type":"expert"}',
        'data: {"p":"response","o":"BATCH","v":[{"p":"response/fragments/1/content","o":"APPEND","v":"git checkout --orphan main; git rm -r --cached . > "},{"p":"response/fragments/1/content","o":"APPEND","v":"/dev/null 2>&1; git add -A"}]}',
        'data: {"p":"response/status","o":"SET","v":"FINISHED"}',
        'event: close',
        'data: {}',
    ]

    async def gen_lines():
        for l in lines:
            yield l

    chunks = []
    async for chunk in parse_sse_lines(gen_lines(), session_id="test-session"):
        if chunk.type == "content":
            chunks.append(chunk.text)

    full_output = "".join(chunks)
    # 验证前序的 "git checkout... > " 没有被丢失，后序的 "/dev/null..." 紧密衔接
    assert "git checkout --orphan main; git rm -r --cached . > /dev/null 2>&1; git add -A" == full_output
    assert len(chunks) == 2
