import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx

from app.api.v1.api_router import api_router
from app.core.config import settings
from app.core.credentials import credentials_manager
from app.providers.registry import provider_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"启动 {settings.APP_NAME} v{settings.APP_VERSION}")
    app.state.http_client = httpx.AsyncClient(
        timeout=settings.REQUEST_TIMEOUT,
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    )

    provider_registry.init_providers(app.state.http_client)

    active_providers = [p["name"] for p in provider_registry.list_providers() if p["authenticated"]]
    if active_providers:
        logger.info(f"已就绪的提供商 (已配置 Token): {', '.join(active_providers)}")
    else:
        logger.warning(
            "提示: 未检测到提供商认证 Token。请通过 /api/v1/auth/token 接口或 credentials.json 配置。"
        )

    yield

    logger.info("关闭服务，释放网络连接池...")
    await app.state.http_client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/", tags=["General"])
    async def root():
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "default_provider": provider_registry.default_provider_id,
            "providers": provider_registry.list_providers(),
        }

    @app.get("/health", tags=["General"])
    async def health():
        return {
            "status": "ok",
            "default_provider": provider_registry.default_provider_id,
            "providers": provider_registry.list_providers(),
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
