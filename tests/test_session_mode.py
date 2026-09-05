import pytest
from unittest.mock import AsyncMock, patch
import httpx
from app.services.session_manager import SessionManager
from app.providers.qwen_provider import QwenProvider
from app.api.v1.endpoints.sessions import get_session_mode, set_session_mode


@pytest.mark.asyncio
async def test_session_manager_single_session_mode():
    mgr = SessionManager()
    mgr.set_single_session_mode(True)
    assert mgr.is_single_session_mode() is True

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    # 1. 首次调用创建新会话
    with patch.object(mgr, "create_new_session", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = "session-123"
        mgr._current_session_id = None

        sid1 = await mgr.get_or_create_session(mock_client)
        assert sid1 == "session-123"
        mgr._current_session_id = "session-123"
        assert mock_create.call_count == 1

    # 2. 单会话模式下第二次调用复用当前会话，不创建新会话
    with patch.object(mgr, "create_new_session", new_callable=AsyncMock) as mock_create:
        sid2 = await mgr.get_or_create_session(mock_client)
        assert sid2 == "session-123"
        mock_create.assert_not_called()

    # 3. 显式废弃会话
    mgr.invalidate_current_session()
    assert mgr.get_current_session_id() is None

    # 4. 废弃后重新创建新会话
    with patch.object(mgr, "create_new_session", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = "session-456"
        sid3 = await mgr.get_or_create_session(mock_client)
        assert sid3 == "session-456"
        assert mock_create.call_count == 1


@pytest.mark.asyncio
async def test_session_manager_multi_session_mode():
    mgr = SessionManager()
    mgr.set_single_session_mode(False)
    assert mgr.is_single_session_mode() is False

    mock_client = AsyncMock(spec=httpx.AsyncClient)

    with patch.object(mgr, "create_new_session", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = ["session-a", "session-b"]

        sid1 = await mgr.get_or_create_session(mock_client)
        sid2 = await mgr.get_or_create_session(mock_client)

        assert sid1 == "session-a"
        assert sid2 == "session-b"
        assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_qwen_provider_single_session_mode():
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    provider = QwenProvider(mock_client)

    from app.services.session_manager import session_manager
    session_manager.set_single_session_mode(True)

    provider._current_chat_id = "qwen-chat-123"
    with patch.object(provider, "_create_new_chat", new_callable=AsyncMock) as mock_create:
        cid = await provider.get_or_create_chat()
        assert cid == "qwen-chat-123"
        mock_create.assert_not_called()

    # multi 模式下创建新会话
    session_manager.set_single_session_mode(False)
    with patch.object(provider, "_create_new_chat", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = "qwen-chat-new"
        cid = await provider.get_or_create_chat()
        assert cid == "qwen-chat-new"
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_sessions_mode_api_endpoints():
    from app.services.session_manager import session_manager

    await set_session_mode(mode="single")
    assert session_manager.is_single_session_mode() is True

    res1 = await get_session_mode()
    assert res1["mode"] == "single"
    assert res1["single_session_mode"] is True

    await set_session_mode(mode="multi")
    assert session_manager.is_single_session_mode() is False

    res2 = await get_session_mode()
    assert res2["mode"] == "multi"
    assert res2["single_session_mode"] is False
