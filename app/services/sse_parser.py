import json
import logging
from typing import AsyncGenerator, Optional, Tuple, Dict, Any
from app.schemas.chat import StreamChunk

logger = logging.getLogger(__name__)


def normalize_fragment_type(raw_type: Any, stage_id: Optional[int] = None) -> str:
    """标准化 DeepSeek 响应分块类型为 THINKING 或 RESPONSE。"""
    if raw_type:
        s = str(raw_type).strip().upper()
        if s in ["THINKING", "THINK", "THOUGHT", "REASONING", "PLANNING"]:
            return "THINKING"
        if s in ["RESPONSE", "ANSWER", "CONTENT", "TEXT"]:
            return "RESPONSE"
    if stage_id == 1:
        return "THINKING"
    return "RESPONSE"


class SSEParser:
    """DeepSeek 网页端 Server-Sent Events (SSE) 事件流解析器。"""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.message_id: Optional[int] = None
        self.parent_message_id: Optional[int] = None
        self.thinking_text: str = ""
        self.response_text: str = ""
        self.current_fragment_type: str = "THINKING"
        self.fragments: Dict[int, Dict[str, Any]] = {}
        self.token_usage: Optional[int] = None
        self.title: Optional[str] = None
        self.is_finished: bool = False

    def _parse_json(self, data_str: str) -> Any:
        if not data_str or data_str.strip() == "":
            return None
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            logger.debug(f"JSON 解析失败: {data_str}")
            return None

    def _apply_patch(self, path: str, op: str, val: Any) -> Optional[StreamChunk]:
        """递归解析 DeepSeek 的 JSON-Patch 流事件。"""
        # 1. 递归处理 BATCH 批量操作
        if op == "BATCH" and isinstance(val, list):
            last_chunk = None
            for item in val:
                if isinstance(item, dict):
                    sub_p = item.get("p", "") or ""
                    sub_o = item.get("o", "") or ""
                    sub_v = item.get("v")

                    if sub_p.startswith("response/"):
                        full_p = sub_p
                    elif sub_p:
                        full_p = f"{path}/{sub_p}".strip("/")
                    else:
                        full_p = path

                    chunk = self._apply_patch(full_p, sub_o, sub_v)
                    if chunk:
                        last_chunk = chunk
            return last_chunk

        # 2. 捕获官方 Token 计数
        if "accumulated_token_usage" in path and val is not None:
            try:
                self.token_usage = int(val)
            except (ValueError, TypeError):
                pass
            return None

        # 3. 思考链分块结束 (status: FINISHED 或 elapsed_secs)
        if ("fragments/0" in path or "fragments/-1" in path) and (path.endswith("/status") and val == "FINISHED" or path.endswith("/elapsed_secs")):
            self.current_fragment_type = "RESPONSE"
            return None

        # 4. 全局响应状态 FINISHED
        if path == "response/status" and val == "FINISHED":
            self.is_finished = True
            return StreamChunk(
                type="status",
                text="FINISHED",
                message_id=self.message_id,
                session_id=self.session_id,
                token_usage=self.token_usage,
            )

        # 5. 追加分块列表: response/fragments -> [{"id": 3, "type": "RESPONSE", "content": "..."}]
        if "fragments" in path and isinstance(val, list):
            for frag in val:
                if isinstance(frag, dict):
                    f_type = normalize_fragment_type(frag.get("type"), frag.get("stage_id"))
                    self.current_fragment_type = f_type
                    f_content = frag.get("content", "")
                    if f_content:
                        chunk_type = "thinking" if f_type == "THINKING" else "content"
                        if chunk_type == "thinking":
                            self.thinking_text += f_content
                        else:
                            self.response_text += f_content
                        return StreamChunk(
                            type=chunk_type,
                            text=f_content,
                            message_id=self.message_id,
                            session_id=self.session_id,
                        )
            return None

        # 6. 设置单一分块对象
        if "fragments" in path and isinstance(val, dict):
            f_type = normalize_fragment_type(val.get("type"), val.get("stage_id"))
            self.current_fragment_type = f_type
            f_content = val.get("content", "")
            if f_content:
                chunk_type = "thinking" if f_type == "THINKING" else "content"
                if chunk_type == "thinking":
                    self.thinking_text += f_content
                else:
                    self.response_text += f_content
                return StreamChunk(
                    type=chunk_type,
                    text=f_content,
                    message_id=self.message_id,
                    session_id=self.session_id,
                )
            return None

        # 7. 更新分块类型
        if "fragments" in path and path.endswith("/type") and isinstance(val, str):
            self.current_fragment_type = normalize_fragment_type(val)
            return None

        # 8. 更新 stage_id
        if "fragments" in path and path.endswith("/stage_id"):
            self.current_fragment_type = "THINKING" if val == 1 else "RESPONSE"
            return None

        # 9. 向分块追加文本: response/fragments/.../content
        if "fragments" in path and (op == "APPEND" or not op) and isinstance(val, str):
            text_piece = val
            if "<think>" in text_piece:
                self.current_fragment_type = "THINKING"
                text_piece = text_piece.replace("<think>", "")
            if "</think>" in text_piece:
                self.current_fragment_type = "RESPONSE"
                text_piece = text_piece.replace("</think>", "")

            if not text_piece:
                return None

            # 区分思考链和正文
            if "fragments/1" in path or "fragments/2" in path:
                self.current_fragment_type = "RESPONSE"
                chunk_type = "content"
                self.response_text += text_piece
            elif "fragments/0" in path and self.response_text:
                self.current_fragment_type = "RESPONSE"
                chunk_type = "content"
                self.response_text += text_piece
            else:
                if self.current_fragment_type == "THINKING":
                    chunk_type = "thinking"
                    self.thinking_text += text_piece
                else:
                    chunk_type = "content"
                    self.response_text += text_piece

            return StreamChunk(
                type=chunk_type,
                text=text_piece,
                message_id=self.message_id,
                session_id=self.session_id,
            )

        return None

    def process_event(self, event_type: str, data_str: str) -> Optional[StreamChunk]:
        """处理单条 SSE 事件。"""
        data = self._parse_json(data_str)

        # 1. event: ready
        if event_type == "ready" and isinstance(data, dict):
            self.message_id = data.get("response_message_id", self.message_id)
            self.parent_message_id = data.get("request_message_id", self.parent_message_id)
            return None

        # 2. event: title
        if event_type == "title":
            if isinstance(data, dict):
                self.title = data.get("title") or data.get("content", "")
            elif isinstance(data, str):
                self.title = data
            return StreamChunk(
                type="title",
                text=self.title or "",
                session_id=self.session_id
            )

        # 3. event: update_session
        if event_type == "update_session":
            return StreamChunk(
                type="session",
                text=str(data.get("updated_at", "")) if isinstance(data, dict) else "",
                session_id=self.session_id
            )

        # 3.5. event: hint (DeepSeek 校验报错或限流)
        if event_type == "hint":
            err_content = ""
            if isinstance(data, dict):
                content_str = data.get("content") or ""
                finish_reason = data.get("finish_reason") or ""
                if content_str and finish_reason and finish_reason != content_str:
                    err_content = f"{content_str} ({finish_reason})"
                else:
                    err_content = content_str or finish_reason or ""
            elif isinstance(data, str):
                err_content = data
            logger.warning(f"DeepSeek 返回 hint 事件: {err_content}")
            return StreamChunk(
                type="error",
                text=f"DeepSeek error: {err_content}",
                session_id=self.session_id,
            )

        # 4. event: close
        if event_type == "close":
            self.is_finished = True
            return StreamChunk(
                type="status",
                text="CLOSED",
                message_id=self.message_id,
                session_id=self.session_id,
                token_usage=self.token_usage,
            )

        # 5. 处理 data 数据
        if isinstance(data, dict):
            if "v" in data and isinstance(data["v"], dict) and isinstance(data["v"].get("response"), dict):
                resp = data["v"]["response"]
                self.message_id = resp.get("message_id", self.message_id)
                self.parent_message_id = resp.get("parent_id", self.parent_message_id)

                raw_fragments = resp.get("fragments") or []
                if isinstance(raw_fragments, list):
                    for idx, frag in enumerate(raw_fragments):
                        if isinstance(frag, dict):
                            f_type = normalize_fragment_type(frag.get("type"), frag.get("stage_id"))
                            f_content = frag.get("content", "")
                            self.fragments[idx] = {"type": f_type, "content": f_content}
                            self.current_fragment_type = f_type

                            if f_content:
                                if f_type == "THINKING":
                                    self.thinking_text += f_content
                                    return StreamChunk(type="thinking", text=f_content, message_id=self.message_id, session_id=self.session_id)
                                else:
                                    self.response_text += f_content
                                    return StreamChunk(type="content", text=f_content, message_id=self.message_id, session_id=self.session_id)
                return None

            # 简易文本分块: data: {"v": "..."}
            if "v" in data and isinstance(data["v"], str) and "p" not in data and "o" not in data:
                text_piece = data["v"]
                if "<think>" in text_piece:
                    self.current_fragment_type = "THINKING"
                    text_piece = text_piece.replace("<think>", "")
                if "</think>" in text_piece:
                    self.current_fragment_type = "RESPONSE"
                    text_piece = text_piece.replace("</think>", "")

                if not text_piece:
                    return None

                if self.current_fragment_type == "THINKING":
                    self.thinking_text += text_piece
                    return StreamChunk(type="thinking", text=text_piece, message_id=self.message_id, session_id=self.session_id)
                else:
                    self.response_text += text_piece
                    return StreamChunk(type="content", text=text_piece, message_id=self.message_id, session_id=self.session_id)

            # JSON-Patch 分块: 路径 "p" 和操作 "o"
            path = data.get("p", "") or ""
            op = data.get("o", "") or ""
            val = data.get("v")

            return self._apply_patch(path, op, val)

        return None


async def parse_sse_lines(line_stream: AsyncGenerator[str, None], session_id: Optional[str] = None) -> AsyncGenerator[StreamChunk, None]:
    """将 DeepSeek SSE 文本行流实时解析为 StreamChunk 对象的异步生成器。"""
    parser = SSEParser(session_id=session_id)
    current_event = "message"

    async for line in line_stream:
        line = line.rstrip("\r\n")
        if not line:
            current_event = "message"
            continue

        if line.startswith("event:"):
            current_event = line[6:].strip()
            continue

        if line.startswith("data:"):
            data_content = line[5:]
            if data_content.startswith(" "):
                data_content = data_content[1:]
            chunk = parser.process_event(current_event, data_content)
            current_event = "message"
            if chunk:
                yield chunk


async def parse_sse_stream(byte_stream: AsyncGenerator[bytes, None], session_id: Optional[str] = None) -> AsyncGenerator[StreamChunk, None]:
    """将 DeepSeek SSE 字节流实时解析为 StreamChunk 对象的异步生成器。"""
    parser = SSEParser(session_id=session_id)
    buffer = ""
    current_event = "message"

    async for chunk_bytes in byte_stream:
        chunk_str = chunk_bytes.decode("utf-8", errors="replace")
        buffer += chunk_str

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r\n")

            if not line:
                current_event = "message"
                continue

            if line.startswith("event:"):
                current_event = line[6:].strip()
                continue

            if line.startswith("data:"):
                data_content = line[5:]
                if data_content.startswith(" "):
                    data_content = data_content[1:]
                chunk = parser.process_event(current_event, data_content)
                current_event = "message"
                if chunk:
                    yield chunk
