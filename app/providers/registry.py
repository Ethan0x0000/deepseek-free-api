import logging
from typing import Dict, List, Optional
import httpx
from app.providers.base import BaseLLMProvider
from app.providers.deepseek_provider import DeepSeekProvider
from app.providers.qwen_provider import QwenProvider
from app.schemas.chat import ModelInfo

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """LLM 提供商中央注册与调度分发器。"""

    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider_id: str = "deepseek"
        # 预注册基础提供商实例 (使用默认 Client，FastAPI lifespan 启动后会用共享 Client 重新覆盖)
        self.init_providers(httpx.AsyncClient())

    def init_providers(self, http_client: httpx.AsyncClient) -> None:
        """初始化可用提供商及其共享 HTTP 客户端。"""
        self._providers["deepseek"] = DeepSeekProvider(http_client)
        self._providers["qwen"] = QwenProvider(http_client)

    def get_provider(self, provider_id: Optional[str] = None) -> BaseLLMProvider:
        """按 ID 返回提供商或返回默认提供商。"""
        pid = (provider_id or self.default_provider_id).lower().strip()
        if pid not in self._providers:
            raise KeyError(f"未知的提供商 '{pid}'。可用提供商: {list(self._providers.keys())}")
        return self._providers[pid]

    def set_default_provider(self, provider_id: str) -> None:
        pid = provider_id.lower().strip()
        if pid not in self._providers:
            raise ValueError(f"未知的提供商 '{pid}'。可用提供商: {list(self._providers.keys())}")
        self.default_provider_id = pid
        logger.info(f"当前默认提供商已变更为: {pid}")

    def resolve_provider_for_model(self, model_name: str) -> BaseLLMProvider:
        """
        根据模型名称自动推断对应的提供商:
        - qwen-... -> Qwen
        - glm-... -> GLM
        - deepseek-..., r1, chat, search 等 -> DeepSeek
        """
        m = (model_name or "").lower().strip()

        if m.startswith("qwen") or "tongyi" in m:
            if "qwen" in self._providers:
                return self._providers["qwen"]
        elif m.startswith("glm") or "zhipu" in m:
            if "glm" in self._providers:
                return self._providers["glm"]

        if "deepseek" in self._providers:
            return self._providers["deepseek"]

        return self.get_provider(self.default_provider_id)

    def get_all_models(self) -> List[ModelInfo]:
        """返回所有提供商的模型聚合列表。"""
        all_models = []
        for p in self._providers.values():
            all_models.extend(p.get_models())
        return all_models

    def list_providers(self) -> List[Dict[str, str]]:
        return [
            {
                "id": p.provider_id,
                "name": p.display_name,
                "is_default": p.provider_id == self.default_provider_id,
                "authenticated": p.is_authenticated(),
            }
            for p in self._providers.values()
        ]


provider_registry = ProviderRegistry()
