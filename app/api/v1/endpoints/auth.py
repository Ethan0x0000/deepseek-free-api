from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.credentials import credentials_manager
from app.services.browser_auth import browser_auth_service

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


class TokenRequest(BaseModel):
    token: Optional[str] = Field(default=None, description="单个 Bearer Token 或 API Key")
    tokens: Optional[List[str]] = Field(default=None, description="多个 Bearer Token 列表")
    provider: Optional[str] = Field(default="deepseek", description="提供商: deepseek, qwen")
    action: Optional[str] = Field(default="append", description="操作模式: 'append' (追加到Token池) 或 'replace' (覆盖Token池)")


@router.post("/token", summary="保存指定提供商的认证 Token (支持单Token追加或批量配置)")
async def set_auth_token(req: TokenRequest) -> Dict[str, Any]:
    prov = (req.provider or "deepseek").lower().strip()
    target: Union[str, List[str]]
    if req.tokens:
        target = [t.strip() for t in req.tokens if t and t.strip()]
        if not target:
            raise HTTPException(status_code=400, detail="tokens 列表不能为空")
    elif req.token:
        target = req.token.strip()
        if len(target) < 5:
            raise HTTPException(status_code=400, detail="token 长度至少为 5 个字符")
    else:
        raise HTTPException(status_code=400, detail="必须提供 'token' 或 'tokens' 字段")

    append_mode = (req.action or "append").lower().strip() == "append"

    try:
        credentials_manager.save(target, provider=prov, append=append_mode)
        pool_info = credentials_manager.get_pool_status(prov)
        return {
            "status": "success",
            "provider": prov,
            "action": "append" if append_mode else "replace",
            "message": f"提供商 '{prov}' 的 Token 保存并激活成功",
            "total_tokens": pool_info["total"],
            "healthy_tokens": pool_info["healthy"],
            "cooling_tokens": pool_info["cooling"],
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
            credentials_manager.save(token, provider=prov, append=True)
            return {
                "status": "success",
                "provider": prov,
                "message": f"通过浏览器成功获取并追加了 {prov} 的 Token！",
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


@router.get("/status", summary="查看所有提供商的认证状态与 Token 池统计")
async def get_auth_status() -> Dict[str, Any]:
    return {
        "authenticated": credentials_manager.is_authenticated(),
        "providers": credentials_manager.get_all_status(),
        "pools": {
            "deepseek": credentials_manager.get_pool_status("deepseek"),
            "qwen": credentials_manager.get_pool_status("qwen"),
        },
    }
