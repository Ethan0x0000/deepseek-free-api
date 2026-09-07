import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import httpx
from app.core.config import settings
from app.core.credentials import credentials_manager

logger = logging.getLogger(__name__)

WASM_WORKER_PATH = Path(__file__).parent.parent / "wasm" / "pow_worker.cjs"
DEEPSEEK_WASM_FALLBACK_URL = "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"


class PoWSolver:
    def __init__(self, worker_path: Path = WASM_WORKER_PATH):
        self.worker_path = worker_path
        if not self.worker_path.exists():
            raise FileNotFoundError(f"未找到 WASM Worker 脚本: {self.worker_path}")
        self._ensure_wasm_file()

    def _ensure_wasm_file(self):
        wasm_file = self.worker_path.parent / "sha3_wasm_bg.wasm"
        if not wasm_file.exists():
            try:
                logger.info("正在下载 sha3_wasm_bg.wasm 文件...")
                with httpx.Client(timeout=15.0) as client:
                    resp = client.get(DEEPSEEK_WASM_FALLBACK_URL)
                    if resp.status_code == 200 and resp.content[:4] == b"\x00asm":
                        wasm_file.write_bytes(resp.content)
                        logger.info("sha3_wasm_bg.wasm 下载完成。")
            except Exception as e:
                logger.warning(f"自动下载 sha3_wasm_bg.wasm 失败: {e}")

    async def get_challenge(
        self,
        client: httpx.AsyncClient,
        target_path: str = "/api/v0/chat/completion",
        token: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{settings.DEEPSEEK_BASE_URL}/api/v0/chat/create_pow_challenge"
        auth_val = f"Bearer {token}" if token else credentials_manager.auth_header
        headers = {
            "accept": "*/*",
            "authorization": auth_val,
            "content-type": "application/json",
            "x-client-bundle-id": settings.CLIENT_BUNDLE_ID,
            "x-client-locale": settings.CLIENT_LOCALE,
            "x-client-platform": settings.CLIENT_PLATFORM,
            "x-client-timezone-offset": settings.CLIENT_TIMEZONE_OFFSET,
            "x-client-version": settings.CLIENT_VERSION,
            "user-agent": settings.USER_AGENT,
        }

        response = await client.post(url, json={"target_path": target_path}, headers=headers)
        response.raise_for_status()
        
        result = response.json() or {}
        data = result.get("data") if isinstance(result, dict) else {}
        if data is None:
            data = {}
        biz_data = data.get("biz_data") if isinstance(data, dict) else {}
        if biz_data is None:
            biz_data = {}

        challenge = None
        if isinstance(biz_data, dict):
            challenge = biz_data.get("challenge") or (biz_data if "algorithm" in biz_data else None)

        if not isinstance(result, dict) or result.get("code") != 0 or not challenge:
            raise ValueError(f"获取 PoW 挑战失败: {result}")

        return challenge

    async def solve_challenge(self, challenge_data: Dict[str, Any]) -> str:
        algorithm = challenge_data["algorithm"]
        challenge = challenge_data["challenge"]
        salt = challenge_data["salt"]
        difficulty = challenge_data["difficulty"]
        expire_at = challenge_data.get("expire_at", 0)
        signature = challenge_data["signature"]
        target_path = challenge_data.get("target_path", "/api/v0/chat/completion")

        prefix = f"{salt}_{expire_at}_"

        proc = await asyncio.create_subprocess_exec(
            "node",
            str(self.worker_path),
            challenge,
            prefix,
            str(difficulty),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"WASM Worker 计算 PoW 失败: {err_msg}")

        answer_str = stdout.decode().strip()
        answer = int(float(answer_str))

        if answer < 0:
            raise RuntimeError(f"未找到难度 {difficulty} 的有效 PoW 求解")

        pow_response = {
            "algorithm": algorithm,
            "challenge": challenge,
            "salt": salt,
            "answer": answer,
            "signature": signature,
            "target_path": target_path,
        }

        json_bytes = json.dumps(pow_response, separators=(",", ":")).encode("utf-8")
        return base64.b64encode(json_bytes).decode("utf-8")

    async def get_pow_header(
        self,
        client: httpx.AsyncClient,
        target_path: str = "/api/v0/chat/completion",
        token: Optional[str] = None,
    ) -> str:
        challenge_data = await self.get_challenge(client, target_path, token=token)
        return await self.solve_challenge(challenge_data)


pow_solver = PoWSolver()
