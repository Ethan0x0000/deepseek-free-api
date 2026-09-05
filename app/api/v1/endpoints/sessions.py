from typing import Annotated, Dict, Any, List, Optional
from fastapi import APIRouter, Depends, Query
import httpx
from app.api.deps import get_http_client
from app.providers.registry import provider_registry
from app.services.session_manager import session_manager

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])


@router.post("/new", summary="Создать новую чат-сессию (начать диалог заново)")
async def create_new_session(
    client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    provider: Optional[str] = Query(None, description="ID провайдера (deepseek, qwen, glm)"),
) -> Dict[str, Any]:
    """Создает новую сессию на серверах выбранного провайдера и сбрасывает текущий контекст."""
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
        "message": f"Новая сессия успешно создана ({target_provider.display_name})",
    }


@router.get("/current", summary="Получить ID текущей активной сессии")
async def get_current_session(
    provider: Optional[str] = Query(None, description="ID провайдера (deepseek, qwen, glm)"),
) -> Dict[str, Any]:
    target_provider = provider_registry.get_provider(provider)
    current_id = target_provider.get_current_session_id()
    parent_id = session_manager.get_parent_message_id(current_id) if current_id and target_provider.provider_id == "deepseek" else None

    return {
        "provider": target_provider.provider_id,
        "session_id": current_id,
        "parent_message_id": parent_id,
        "has_active_session": bool(current_id),
    }


@router.get("/list", summary="Список доступных сессий/диалогов провайдера")
async def list_sessions(
    provider: Optional[str] = Query(None, description="ID провайдера (deepseek, qwen, glm)"),
) -> Dict[str, Any]:
    """Возвращает список существующих чатов с серверов провайдера."""
    target_provider = provider_registry.get_provider(provider)
    sessions = await target_provider.list_sessions()
    return {
        "provider": target_provider.provider_id,
        "count": len(sessions),
        "sessions": sessions,
    }


@router.post("/reset", summary="Сбросить локальный контекст сессии")
async def reset_session(
    provider: Optional[str] = Query(None, description="ID провайдера (deepseek, qwen, glm)"),
) -> Dict[str, Any]:
    target_provider = provider_registry.get_provider(provider)
    target_provider.reset_session()
    return {
        "status": "success",
        "provider": target_provider.provider_id,
        "message": "Текущий контекст сброшен. Следующий запрос создаст новую сессию.",
    }


@router.get("/mode", summary="Получить текущий режим сессий (single или multi)")
async def get_session_mode() -> Dict[str, Any]:
    """Возвращает информацию о текущем режиме прокси: single (единая сессия) или multi (новый чат на запрос)."""
    return {
        "mode": "single" if session_manager.is_single_session_mode() else "multi",
        "single_session_mode": session_manager.is_single_session_mode(),
        "description": "Единая сессия без создания новых чатов" if session_manager.is_single_session_mode() else "Изолированные чаты на каждый запрос",
    }


@router.post("/mode", summary="Переключить режим сессий")
async def set_session_mode(
    mode: str = Query(..., description="Режим сессий: 'single' (единая сессия) или 'multi' (изолированные чаты)"),
) -> Dict[str, Any]:
    """Переключает режим сессий: 'single' (без создания новых чатов) или 'multi'."""
    is_single = mode.strip().lower() in ["single", "1", "true", "s"]
    session_manager.set_single_session_mode(is_single)
    return {
        "status": "success",
        "mode": "single" if is_single else "multi",
        "single_session_mode": is_single,
        "message": f"Режим успешно изменен на: {'single (единая сессия)' if is_single else 'multi (изолированные чаты)'}",
    }

