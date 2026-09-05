from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str = Field(..., min_length=1)


class DeepSeekChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Текст запроса")
    chat_session_id: Optional[str] = Field(default=None, description="ID сессии. Если не передан, используется активная или создается новая")
    parent_message_id: Optional[int] = Field(default=None, description="ID родительского сообщения для продолжения контекста")
    ref_file_ids: Optional[List[str]] = Field(default=None, description="ID прикрепленных файлов/изображений")
    model: str = Field(default="deepseek-chat", description="Модель: deepseek-chat (expert), deepseek-reasoner (r1), deepseek-search")
    thinking_enabled: Optional[bool] = Field(default=None, description="Включить режим рассуждений (DeepSeek R1)")
    search_enabled: Optional[bool] = Field(default=None, description="Включить веб-поиск")
    stream: bool = Field(default=True, description="Стриминг ответа (SSE)")


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
