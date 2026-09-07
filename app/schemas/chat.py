from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str = Field(..., min_length=1)


class DeepSeekChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="请求提示词文本")
    chat_session_id: Optional[str] = Field(default=None, description="会话 ID，如不传则使用当前活跃会话或创建新会话")
    parent_message_id: Optional[int] = Field(default=None, description="父消息 ID，用于在同会话内串接上下文")
    ref_file_ids: Optional[List[str]] = Field(default=None, description="已上传的附件/图像文件 ID 列表")
    model: str = Field(default="deepseek-chat", description="模型标识符: deepseek-chat, deepseek-reasoner, deepseek-v4-pro 等")
    thinking_enabled: Optional[bool] = Field(default=None, description="是否启用深度思考 (DeepSeek R1 / Thinking)")
    search_enabled: Optional[bool] = Field(default=None, description="是否开启联网搜索")
    stream: bool = Field(default=True, description="是否启用流式输出 (SSE)")
    active_token: Optional[str] = Field(default=None, description="本次请求锁定的认证 Token")


class StreamChunk(BaseModel):
    type: Literal["thinking", "content", "status", "session", "title", "error"]
    text: str = ""
    message_id: Optional[int] = None
    session_id: Optional[str] = None
    token_usage: Optional[int] = None


class DeepSeekChatResponse(BaseModel):
    session_id: str
    message_id: int
    parent_message_id: Optional[int] = None
    thinking: Optional[str] = None
    content: str
    token_usage: Optional[int] = None
    status: str = "FINISHED"


class SessionInfo(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    model_type: str
    supports_thinking: bool
    supports_search: bool
