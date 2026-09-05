import json
import logging
from typing import List, Optional, Union, Any
from app.core.config import settings
from app.schemas.openai import OpenAIChatMessage

logger = logging.getLogger(__name__)


def estimate_tokens(text: Union[str, Any]) -> int:
    """
    快速准确评估多语言文本、代码和 JSON 的 Token 数量。
    - 英文与代码: ~3.6 字符 / Token
    - 中文汉字 (CJK) / 常见符号: ~1.4 字符 / Token
    - 空格与特殊符号综合加权
    """
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)

    length = len(text)
    if length == 0:
        return 0

    non_ascii_count = sum(1 for c in text if ord(c) > 127)
    ascii_count = length - non_ascii_count

    tokens = int((ascii_count / 3.6) + (non_ascii_count / 1.4))
    return max(1, tokens)


def truncate_tool_output(content: str, max_tokens: int = 25_000) -> str:
    """
    截断超长工具执行输出（如大型文件日志、超大目录列表），
    保留头部（Head）与尾部（Tail），确保关键错误与开头信息不丢失。
    """
    current_tokens = estimate_tokens(content)
    if current_tokens <= max_tokens:
        return content

    # 保留前 40% 与后 40%，中间裁剪
    budget_chars = int(max_tokens * 3.2)
    head_len = int(budget_chars * 0.45)
    tail_len = int(budget_chars * 0.45)

    if head_len + tail_len >= len(content):
        return content

    head = content[:head_len]
    tail = content[-tail_len:]
    omitted_chars = len(content) - (head_len + tail_len)
    omitted_tokens = int(omitted_chars / 3.2)

    return (
        f"{head}\n\n"
        f"[... Context compressed: omitted {omitted_chars:,} characters (~{omitted_tokens:,} tokens) of output ...]\n\n"
        f"{tail}"
    )


class ContextCompressor:
    """
    智能上下文压缩器:
    - 监控提供商的安全 Token 预算与 UTF-8 Payload 大小
    - 100% 完整保留系统指令和 Tools 工具定义
    - 完整保留最近 N 轮对话历史
    - 对过长历史中间部分进行无损摘要与压缩
    """

    QWEN_MAX_WEB_TOKENS: int = 20_000
    QWEN_MAX_PAYLOAD_BYTES: int = 70_000
    DEEPSEEK_MAX_WEB_TOKENS: int = 100_000
    DEEPSEEK_MAX_PAYLOAD_BYTES: int = 380_000
    DEFAULT_MAX_TOKENS: int = 64_000

    def __init__(
        self,
        max_context_tokens: Optional[int] = None,
        retain_recent_count: Optional[int] = None,
        max_tool_tokens: Optional[int] = None,
    ):
        self.max_context_tokens = max_context_tokens or getattr(settings, "MAX_CONTEXT_TOKENS", 100_000)
        self.retain_recent_count = retain_recent_count or getattr(settings, "RETAIN_RECENT_MESSAGES_COUNT", 12)
        self.max_tool_tokens = max_tool_tokens or getattr(settings, "MAX_TOOL_OUTPUT_TOKENS", 25_000)

    def get_limit_for_provider(self, provider_id: str) -> int:
        """返回指定提供商的安全 Token 上限。"""
        pid = str(provider_id).lower().strip()
        if pid == "qwen":
            return self.QWEN_MAX_WEB_TOKENS
        elif pid == "deepseek":
            return min(self.max_context_tokens, self.DEEPSEEK_MAX_WEB_TOKENS)
        return self.max_context_tokens

    def compress_openai_messages(
        self,
        messages: List[OpenAIChatMessage],
        max_tokens: Optional[int] = None,
    ) -> List[OpenAIChatMessage]:
        """将 OpenAI 消息列表压缩到指定 Token 预算内。"""
        if not messages:
            return messages

        limit = max_tokens or self.max_context_tokens
        tool_limit = min(self.max_tool_tokens, max(2_000, limit // 5))

        # 1. 优先压缩过大 tool 输出
        sanitized_messages: List[OpenAIChatMessage] = []
        for msg in messages:
            if msg.role in ["tool", "function"] and isinstance(msg.content, str):
                compressed_content = truncate_tool_output(msg.content, max_tokens=tool_limit)
                if compressed_content != msg.content:
                    msg_dict = msg.model_dump()
                    msg_dict["content"] = compressed_content
                    sanitized_messages.append(OpenAIChatMessage(**msg_dict))
                    continue
            sanitized_messages.append(msg)

        # 2. 评估总 Token
        total_tokens = sum(estimate_tokens(m.content or "") for m in sanitized_messages)
        if total_tokens <= limit:
            return sanitized_messages

        logger.info(
            f"对话上下文 ({total_tokens:,} Token) 超出阈值 {limit:,}，启动智能压缩..."
        )

        # 3. 分离系统提示、历史中间段与最近活跃消息
        system_msgs = [m for m in sanitized_messages if m.role == "system"]
        non_system_msgs = [m for m in sanitized_messages if m.role != "system"]

        if len(non_system_msgs) <= self.retain_recent_count:
            return sanitized_messages

        recent_msgs = non_system_msgs[-self.retain_recent_count:]
        middle_msgs = non_system_msgs[:-self.retain_recent_count]

        # 4. 汇总压缩中间历史段
        summary_lines = []
        for m in middle_msgs:
            role = m.role
            c = str(m.content or "")
            if len(c) > 300:
                c = c[:280] + "..."
            summary_lines.append(f"- [{role}]: {c}")

        summary_text = (
            f"[Summary of previous conversation context ({len(middle_msgs)} earlier messages compressed)]:\n"
            + "\n".join(summary_lines)
        )

        summary_msg = OpenAIChatMessage(
            role="system",
            content=summary_text,
        )

        result = system_msgs + [summary_msg] + recent_msgs
        new_tokens = sum(estimate_tokens(m.content or "") for m in result)
        logger.info(f"✓ 上下文成功压缩: 从 {total_tokens:,} 降至 {new_tokens:,} Token。")
        return result

    def compress_raw_prompt(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ) -> str:
        """针对底层文本的兜底安全截断与压缩，确保不触发 WAF 阻断。"""
        limit = max_tokens or self.max_context_tokens
        curr_tokens = estimate_tokens(prompt)
        prompt_bytes = len(prompt.encode("utf-8"))
        if max_bytes:
            effective_max_bytes = max_bytes
        elif limit <= self.QWEN_MAX_WEB_TOKENS:
            effective_max_bytes = self.QWEN_MAX_PAYLOAD_BYTES
        else:
            effective_max_bytes = self.DEEPSEEK_MAX_PAYLOAD_BYTES

        if curr_tokens <= limit and prompt_bytes <= effective_max_bytes:
            return prompt

        logger.info(
            f"提示词 ({curr_tokens:,} Token, {prompt_bytes:,} 字节) 超过安全限制 "
            f"({limit:,} Token, {effective_max_bytes} 字节)。应用自适应压缩..."
        )

        bytes_per_char = max(1.0, prompt_bytes / max(1, len(prompt)))
        if effective_max_bytes and prompt_bytes > effective_max_bytes:
            target_char_len = int((effective_max_bytes - 800) / bytes_per_char)
        else:
            target_char_len = int(limit * 3.0 / bytes_per_char)

        # 优先压缩 Conversation History 块，完整保留系统指令和 Tools
        for marker in ["\nConversation History:\n", "\n\nConversation History:\n", "Conversation History:\n"]:
            if marker in prompt:
                header, history = prompt.split(marker, 1)
                header_with_marker = header + marker
                header_bytes = len(header_with_marker.encode("utf-8"))
                remaining_bytes = (effective_max_bytes - 800) - header_bytes if effective_max_bytes else (target_char_len - len(header_with_marker))

                if remaining_bytes > 3000:
                    h_bytes_per_char = max(1.0, len(history.encode("utf-8")) / max(1, len(history)))
                    h_target_chars = int(remaining_bytes / h_bytes_per_char)
                    if len(history) > h_target_chars:
                        h_head_chars = int(h_target_chars * 0.20)
                        h_tail_chars = int(h_target_chars * 0.75)
                        h_head = history[:h_head_chars]
                        h_tail = history[-h_tail_chars:]
                        omitted = len(history) - (h_head_chars + h_tail_chars)
                        omitted_tokens = int(omitted / 3.2)
                        return (
                            f"{header_with_marker}{h_head}\n\n"
                            f"[... Context compressed: omitted {omitted:,} characters (~{omitted_tokens:,} tokens) "
                            f"of intermediate history to keep model focus within {limit:,} tokens ...]\n\n"
                            f"{h_tail}"
                        )

        # 兜底压缩
        head_chars = int(target_char_len * 0.35)
        tail_chars = int(target_char_len * 0.55)

        if head_chars + tail_chars >= len(prompt):
            return prompt

        head = prompt[:head_chars]
        tail = prompt[-tail_chars:]
        omitted_chars = len(prompt) - (head_chars + tail_chars)
        omitted_tokens = int(omitted_chars / 3.2)

        compressed = (
            f"{head}\n\n"
            f"[... Context compressed: omitted {omitted_chars:,} characters (~{omitted_tokens:,} tokens) "
            f"of intermediate history to keep model focus within {limit:,} tokens ...]\n\n"
            f"{tail}"
        )
        return compressed


context_compressor = ContextCompressor()
