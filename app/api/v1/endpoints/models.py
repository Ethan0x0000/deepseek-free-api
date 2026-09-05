import time
from typing import Any, Dict, List
from fastapi import APIRouter
from app.providers.registry import provider_registry
from app.schemas.chat import ModelInfo

router = APIRouter(tags=["Models"])


@router.get("/api/v1/models", response_model=List[ModelInfo], summary="获取所有提供商的可用模型列表")
async def list_models() -> List[ModelInfo]:
    return provider_registry.get_all_models()


@router.get("/v1/models", summary="OpenAI 兼容的模型列表接口")
async def openai_list_models() -> Dict[str, Any]:
    all_models = provider_registry.get_all_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m.id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek-free-api",
                "permission": [],
                "root": m.id,
                "parent": None,
            }
            for m in all_models
        ]
    }


@router.get("/api/v1/providers", summary="获取所有 LLM 提供商列表及其认证状态")
async def list_providers() -> List[Dict[str, Any]]:
    return provider_registry.list_providers()


@router.post("/api/v1/providers/switch", summary="切换默认活跃提供商")
async def switch_default_provider(provider_id: str) -> Dict[str, Any]:
    provider_registry.set_default_provider(provider_id)
    return {
        "status": "success",
        "default_provider": provider_registry.default_provider_id,
        "message": f"默认提供商已成功切换为 {provider_registry.default_provider_id}",
    }
