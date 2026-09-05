from typing import Annotated, Dict, Any, List, Optional
from fastapi import APIRouter, Depends, Query
import httpx
from app.api.deps import get_http_client
from app.providers.registry import provider_registry
from app.services.session_manager import session_manager

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])


@router.post("/new", summary="创建新会话 (重新开始对话)")
async def create_new_session(
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    provider: Optional[str] = Query(None, description="提供商 ID (deepseek, qwen, glm)"),
) -> Dict[str, Any]:
    """在指定提供商服务端创建新会话并重置本地上下文。"""
    target_provider = provider_registry.get_provider(provider)

    if target_provider.provider_id == "qwen":
        target_provider.reset_session()
        session_id = await target_provider.get_or_create_chat()
    else:
        session_id = await session_manager.create_new_session(client)

    return {
        "status": "success",
        "provider": target_provider.provider_id,
        "session_id": session_id,
        "message": f"新会话创建成功 ({target_provider.display_name})",
    }


@router.get("/current", summary="获取当前活跃会话 ID")
async def get_current_session(
    provider: Optional[str] = Query(None, description="提供商 ID (deepseek, qwen, glm)"),
) -> Dict[str, Any]:
    target_provider = provider_registry.get_provider(provider)
    current_id = target_provider.get_current_session_id()
    parent_id = session_manager.get_parent_message_id(current_id) if current_id and target_provider.provider_id == "deepseek" else None

    return {
        "provider": target_provider.provider_id,
        "session_id": current_id,
        "parent_message_id": parent_id,
        "active": current_id is not None,
    }


@router.get("/list", summary="获取提供商的可用历史会话列表")
async def list_provider_sessions(
    provider: Optional[str] = Query(None, description="提供商 ID (deepseek, qwen, glm)"),
) -> Dict[str, Any]:
    """返回提供商服务端的历史对话列表。"""
    target_provider = provider_registry.get_provider(provider)
    sessions = await target_provider.list_sessions()
    return {
        "provider": target_provider.provider_id,
        "sessions": sessions,
        "count": len(sessions),
    }


@router.post("/reset", summary="重置本地会话上下文")
async def reset_session_context(
    provider: Optional[str] = Query(None, description="提供商 ID (deepseek, qwen, glm)"),
) -> Dict[str, Any]:
    target_provider = provider_registry.get_provider(provider)
    target_provider.reset_session()
    return {
        "status": "success",
        "provider": target_provider.provider_id,
        "message": "当前上下文已重置，下一条请求将创建全新会话。",
    }


@router.get("/mode", summary="获取当前会话模式 (single 或 multi)")
async def get_session_mode() -> Dict[str, Any]:
    """返回代理当前会话模式: single (单会话复用) 或 multi (每请求独立临时会话)。"""
    return {
        "mode": "single" if session_manager.is_single_session_mode() else "multi",
        "single_session_mode": session_manager.is_single_session_mode(),
        "description": "单会话复用模式" if session_manager.is_single_session_mode() else "每请求独立隔离会话 (防止上下文膨胀)",
    }


@router.post("/mode", summary="切换会话模式")
async def set_session_mode(
    mode: str = Query(..., description="会话模式: 'single' (单会话) 或 'multi' (隔离会话)"),
) -> Dict[str, Any]:
    """切换会话模式: 'single' 或 'multi'。"""
    is_single = mode.strip().lower() in ["single", "1", "true", "s"]
    session_manager.set_single_session_mode(is_single)
    return {
        "status": "success",
        "mode": "single" if is_single else "multi",
        "single_session_mode": is_single,
        "message": f"会话模式已成功修改为: {'single (单会话复用)' if is_single else 'multi (独立隔离会话)'}",
    }
