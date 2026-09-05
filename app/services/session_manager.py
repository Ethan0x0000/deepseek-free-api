import logging
from typing import Dict, Optional
import httpx
from app.core.config import settings
from app.core.credentials import credentials_manager

logger = logging.getLogger(__name__)


class SessionManager:
    """管理与 DeepSeek / 各厂商 Web 会话生命周期并追踪上下文 (parent_message_id)。

    在 multi 模式下为无状态 Agent 请求生成临时会话，并在完成后自动清理，防止网页端被垃圾会话污染。
    在 single 模式下维护单会话复用。
    """

    def __init__(self):
        self._current_session_id: Optional[str] = None
        # 每个提供商的会话ID: provider_id -> session_id
        self._provider_sessions: Dict[str, Optional[str]] = {}
        # session_id -> last_message_id (用于维持同会话内的消息链)
        self._last_message_ids: Dict[str, Optional[int]] = {}
        # 会话标题缓存
        self._session_titles: Dict[str, str] = {}
        # 会话模式: single (单会话累积) 或 multi (每请求独立临时会话)
        self.single_session_mode: bool = bool(
            settings.SINGLE_SESSION_MODE or (settings.PROXY_MODE.lower() == "single")
        )

    def set_single_session_mode(self, enabled: bool) -> None:
        """切换单会话复用模式或独立会话模式。"""
        self.single_session_mode = enabled
        logger.info(f"会话模式已切换为: {'单会话模式 (Single)' if enabled else '独立会话模式 (Multi)'}")

    def is_single_session_mode(self) -> bool:
        return self.single_session_mode

    # ── 提供商会话存储 ─────────────────────────────────────────

    def get_provider_session(self, provider_id: str) -> Optional[str]:
        """获取指定提供商的已保存 session_id。"""
        sid = self._provider_sessions.get(provider_id)
        if sid is None and provider_id == "deepseek":
            sid = self._current_session_id
        return sid

    def set_provider_session(self, provider_id: str, session_id: str) -> None:
        """保存指定提供商的 session_id。"""
        self._provider_sessions[provider_id] = session_id
        if provider_id == "deepseek":
            self._current_session_id = session_id
        if session_id not in self._last_message_ids:
            self._last_message_ids[session_id] = None
        logger.debug(f"已更新提供商 {provider_id} 的会话ID: {session_id}")

    def clear_provider_session(self, provider_id: str) -> None:
        """重置指定提供商的会话。"""
        old = self._provider_sessions.pop(provider_id, None)
        if provider_id == "deepseek":
            self._current_session_id = None
        if old:
            logger.info(f"提供商 {provider_id} 的会话已重置: {old}")

    # ── DeepSeek 会话管理与垃圾回收 ────────────────────────

    def invalidate_current_session(self) -> None:
        """当服务端报错或会话失效时，废弃当前会话。"""
        if self._current_session_id:
            logger.warning(f"废弃 DeepSeek 失效会话: {self._current_session_id}")
            self._last_message_ids.pop(self._current_session_id, None)
            self._provider_sessions.pop("deepseek", None)
            self._current_session_id = None

    def get_current_session_id(self) -> Optional[str]:
        return self._current_session_id

    def set_current_session_id(self, session_id: str) -> None:
        self._current_session_id = session_id
        self._provider_sessions["deepseek"] = session_id
        if session_id not in self._last_message_ids:
            self._last_message_ids[session_id] = None

    def get_parent_message_id(self, session_id: str) -> Optional[int]:
        return self._last_message_ids.get(session_id)

    def update_session_state(self, session_id: str, last_message_id: int, title: Optional[str] = None) -> None:
        self._last_message_ids[session_id] = last_message_id
        if title:
            self._session_titles[session_id] = title

    async def create_new_session(self, client: httpx.AsyncClient) -> str:
        """在 DeepSeek 网页端创建新的会话。"""
        url = f"{settings.DEEPSEEK_BASE_URL}/api/v0/chat_session/create"
        headers = {
            "accept": "*/*",
            "authorization": credentials_manager.auth_header,
            "content-type": "application/json",
            "x-client-bundle-id": settings.CLIENT_BUNDLE_ID,
            "x-client-locale": settings.CLIENT_LOCALE,
            "x-client-platform": settings.CLIENT_PLATFORM,
            "x-client-timezone-offset": settings.CLIENT_TIMEZONE_OFFSET,
            "x-client-version": settings.CLIENT_VERSION,
            "user-agent": settings.USER_AGENT,
        }

        response = await client.post(url, json={}, headers=headers)
        response.raise_for_status()

        result = response.json() or {}
        data = result.get("data") if isinstance(result, dict) else {}
        if data is None:
            data = {}
        biz_data = data.get("biz_data") if isinstance(data, dict) else {}
        if biz_data is None:
            biz_data = {}

        session_id = None
        if isinstance(biz_data, dict):
            if "chat_session" in biz_data and isinstance(biz_data["chat_session"], dict):
                session_id = biz_data["chat_session"].get("id")
            elif "id" in biz_data:
                session_id = biz_data.get("id")

        if not isinstance(result, dict) or result.get("code") != 0 or not session_id:
            raise ValueError(f"创建 DeepSeek 网页端会话失败: {result}")

        self._current_session_id = session_id
        self._provider_sessions["deepseek"] = session_id
        self._last_message_ids[session_id] = None
        logger.info(f"已创建 DeepSeek 网页会话: {session_id}")
        return session_id

    async def delete_session(self, client: httpx.AsyncClient, session_id: str) -> None:
        """
        在后台静默删除 DeepSeek 网页端的临时会话。
        防止 AI Agent 的大量代码测试与中间轮次把用户的网页左侧对话列表刷屏。
        """
        if not session_id:
            return
        try:
            url = f"{settings.DEEPSEEK_BASE_URL}/api/v0/chat_session/delete"
            headers = {
                "accept": "*/*",
                "authorization": credentials_manager.auth_header,
                "content-type": "application/json",
                "x-client-platform": settings.CLIENT_PLATFORM,
                "x-client-version": settings.CLIENT_VERSION,
                "user-agent": settings.USER_AGENT,
            }
            resp = await client.post(url, json={"chat_session_id": session_id}, headers=headers, timeout=10.0)
            if resp.status_code == 200:
                logger.debug(f"已自动清理网页端临时会话: {session_id}")
        except Exception as e:
            logger.debug(f"后台清理网页端会话失败 (非致命): {e}")

    async def get_or_create_session(self, client: httpx.AsyncClient, session_id: Optional[str] = None) -> str:
        """
        获取指定的 session_id，或在单会话模式下复用，或创建全新临时会话。
        """
        if session_id:
            if session_id not in self._last_message_ids:
                self._last_message_ids[session_id] = None
            self._current_session_id = session_id
            self._provider_sessions["deepseek"] = session_id
            return session_id

        # 单会话模式下复用保存的会话
        if self.single_session_mode:
            saved = self._provider_sessions.get("deepseek") or self._current_session_id
            if saved:
                logger.debug(f"复用当前 DeepSeek 单会话 (Single-Session): {saved}")
                self._current_session_id = saved
                return saved

        # 默认 multi 模式创建全新会话
        return await self.create_new_session(client)

    def reset_context(self) -> None:
        """重置当前活跃会话。"""
        self._current_session_id = None
        self._provider_sessions.pop("deepseek", None)


session_manager = SessionManager()
