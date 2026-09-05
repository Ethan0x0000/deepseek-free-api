import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = ""
    input_schema: Dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class AnthropicContentBlock(BaseModel):
    type: Literal["text", "thinking", "redacted_thinking", "tool_use", "tool_result", "image"] = "text"
    text: Optional[str] = None
    thinking: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, List[Any]]] = None
    is_error: Optional[bool] = None
    source: Optional[Dict[str, Any]] = None


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[AnthropicContentBlock, Dict[str, Any]]]]


class AnthropicThinkingConfig(BaseModel):
    type: Literal["enabled", "disabled"] = "enabled"
    budget_tokens: Optional[int] = 1024


class AnthropicMessagesRequest(BaseModel):
    model: str = Field(default="deepseek-v4-pro")
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    max_tokens: Optional[int] = 4096
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    thinking: Optional[AnthropicThinkingConfig] = None
    chat_session_id: Optional[str] = None
    session_id: Optional[str] = None


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class AnthropicMessagesResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:20]}")
    type: str = "message"
    role: str = "assistant"
    model: str
    content: List[AnthropicContentBlock]
    stop_reason: Optional[str] = "end_turn"  # "end_turn", "tool_use", "max_tokens"
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)
