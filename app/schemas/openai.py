import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class OpenAIToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class OpenAITool(BaseModel):
    type: Literal["function"] = "function"
    function: OpenAIToolFunction


class OpenAIToolCallFunction(BaseModel):
    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: OpenAIToolCallFunction


class OpenAIChatMessage(BaseModel):
    role: str  # "system", "user", "assistant", "tool", "function"
    content: Optional[Union[str, List[Dict[str, Any]]]] = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None


class OpenAIChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}

    model: str = Field(default="deepseek-v4-pro")
    messages: List[OpenAIChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    chat_session_id: Optional[str] = None
    session_id: Optional[str] = None
    thinking_enabled: Optional[bool] = None
    thinking: Optional[Union[bool, Dict[str, Any]]] = None
    search_enabled: Optional[bool] = None
    stream_options: Optional[Dict[str, Any]] = None


class OpenAIChoiceMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = ""
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None


class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIChoiceMessage
    finish_reason: Optional[str] = "stop"  # "stop", "tool_calls", "length"


class OpenAIPromptTokensDetails(BaseModel):
    cached_tokens: int = 0
    audio_tokens: int = 0


class OpenAICompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: Optional[OpenAIPromptTokensDetails] = None
    completion_tokens_details: Optional[OpenAICompletionTokensDetails] = None


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[OpenAIChoice]
    usage: Optional[OpenAIUsage] = None


# --- Streaming Chunks ---

class OpenAIDeltaToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class OpenAIDeltaToolCall(BaseModel):
    index: int = 0
    id: Optional[str] = None
    type: Optional[str] = "function"
    function: Optional[OpenAIDeltaToolCallFunction] = None


class OpenAIDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[OpenAIDeltaToolCall]] = None


class OpenAIChunkChoice(BaseModel):
    index: int = 0
    delta: OpenAIDelta
    finish_reason: Optional[str] = None


class OpenAIChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[OpenAIChunkChoice] = Field(default_factory=list)
    usage: Optional[OpenAIUsage] = None
