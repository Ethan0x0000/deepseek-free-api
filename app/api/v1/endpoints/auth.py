from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.credentials import credentials_manager
from app.services.browser_auth import browser_auth_service

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


class TokenRequest(BaseModel):
    token: str = Field(..., min_length=5, description="从浏览器提取的 Bearer Token 或 API Key")
    provider: Optional[str] = Field(default="deepseek", description="提供商: deepseek, qwen")


@router.post("/token", summary="保存指定提供商的认证 Token")
async def set_auth_token(req: TokenRequest) -> Dict[str, Any]:
    prov = (req.provider or "deepseek").lower().strip()
    try:
        credentials_manager.save_token(req.token, provider=prov)
        return {
            "status": "success",
            "provider": prov,
            "message": f"提供商 '{prov}' 的 Token 保存并激活成功",
            "token_preview": f"{req.token[:6]}...{req.token[-4:]}" if len(req.token) > 10 else "***",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存提供商 {prov} 的 Token 失败: {str(e)}"
        )


@router.post("/browser-login", summary="启动系统浏览器并自动捕获 Token")
async def browser_login(provider: Optional[str] = "deepseek") -> Dict[str, Any]:
    prov = (provider or "deepseek").lower().strip()

    if prov not in ["deepseek", "qwen"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前仅支持 'deepseek' 与 'qwen' 提供商"
        )

    try:
        token = await browser_auth_service.login_and_capture_token(provider=prov)
        if token:
            credentials_manager.save_token(token, provider=prov)
            return {
                "status": "success",
                "provider": prov,
                "message": f"通过浏览器成功获取并保存了 {prov} 的 Token！",
                "token_preview": f"{token[:6]}...{token[-4:]}",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail=f"获取 {prov} Token 失败 (超时或浏览器窗口已关闭)"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/status", summary="查看所有提供商的认证状态")
async def get_auth_status() -> Dict[str, Any]:
    return {
        "authenticated": credentials_manager.is_authenticated(),
        "providers": credentials_manager.get_all_status(),
    }
