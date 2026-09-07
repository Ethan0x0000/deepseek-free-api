import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from app.core.config import settings

logger = logging.getLogger(__name__)


class CredentialsManager:
    """管理各厂商多 Token 凭证池、轮询调度 (Round-Robin)、健康监测与熔断冷却机制。"""

    def __init__(self):
        self.project_file = settings.CREDENTIALS_PATH
        self.user_file = settings.USER_CREDENTIALS_PATH
        self.env_file = settings.PROJECT_ROOT / ".env"
        self._lock = threading.RLock()

        # 凭证池: provider -> List[clean_token]
        self._token_pools: Dict[str, List[str]] = {}
        # 轮询游标: provider -> int
        self._token_indices: Dict[str, int] = {}
        # Token 健康状态跟踪: provider -> token -> dict
        # {cooldown_until: float, failure_count: int, success_count: int, last_error: str, last_used: float}
        self._token_status: Dict[str, Dict[str, Dict[str, Any]]] = {}

        self.load()

    @staticmethod
    def _clean_token_str(token: str) -> str:
        """清洗 Token 字符串，去除 Bearer 前缀及多余空白符与引号。"""
        if not token:
            return ""
        t = token.strip().strip("\"'")
        if t.startswith("Bearer "):
            t = t[7:].strip().strip("\"'")
        return t

    def _ensure_token_status(self, provider: str, token: str) -> Dict[str, Any]:
        """初始化或获取 Token 状态对象。"""
        if provider not in self._token_status:
            self._token_status[provider] = {}
        if token not in self._token_status[provider]:
            self._token_status[provider][token] = {
                "cooldown_until": 0.0,
                "failure_count": 0,
                "success_count": 0,
                "last_error": None,
                "last_used": 0.0,
            }
        return self._token_status[provider][token]

    def _read_from_file(self, path: Path) -> Dict[str, List[str]]:
        """从 JSON 文件中读取凭证，兼容单个字符串和字符串数组格式。"""
        tokens: Dict[str, List[str]] = {}
        if not path.exists():
            return tokens

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    return tokens

                for k, v in data.items():
                    prov = k.lower().strip()
                    if isinstance(v, list):
                        cleaned_list = [self._clean_token_str(x) for x in v if isinstance(x, str)]
                        cleaned_list = [x for x in cleaned_list if x]
                        if cleaned_list:
                            tokens[prov] = cleaned_list
                    elif isinstance(v, str) and v.strip():
                        clean_v = self._clean_token_str(v)
                        if clean_v:
                            tokens[prov] = [clean_v]

                # 兼容旧版本单一 key: token / authorization / ds_session_id
                single_token = data.get("token") or data.get("authorization") or data.get("ds_session_id")
                if single_token and "deepseek" not in tokens:
                    st = self._clean_token_str(single_token)
                    if st:
                        tokens["deepseek"] = [st]
        except Exception as e:
            logger.debug(f"读取凭证文件失败 {path}: {e}")

        return tokens

    def load(self) -> Dict[str, List[str]]:
        """从各存储介质加载全部 Token 并构建凭证池。"""
        with self._lock:
            self._token_pools = {}

            proj_tokens = self._read_from_file(self.project_file)
            for prov, t_list in proj_tokens.items():
                self._token_pools[prov] = list(t_list)

            user_tokens = self._read_from_file(self.user_file)
            for prov, t_list in user_tokens.items():
                if prov not in self._token_pools:
                    self._token_pools[prov] = []
                for t in t_list:
                    if t not in self._token_pools[prov]:
                        self._token_pools[prov].append(t)

            # 环境变量支持：支持逗号、分号或换行分隔的多 Token
            if settings.DEEPSEEK_BEARER_TOKEN:
                raw_env = settings.DEEPSEEK_BEARER_TOKEN.strip()
                if raw_env:
                    split_tokens = [
                        self._clean_token_str(x)
                        for part in raw_env.replace(";", ",").replace("\n", ",").split(",")
                        if (x := part.strip())
                    ]
                    split_tokens = [x for x in split_tokens if x]
                    if split_tokens:
                        if "deepseek" not in self._token_pools:
                            self._token_pools["deepseek"] = []
                        for t in split_tokens:
                            if t not in self._token_pools["deepseek"]:
                                self._token_pools["deepseek"].append(t)

            # 初始化状态字典
            for prov, t_list in self._token_pools.items():
                for t in t_list:
                    self._ensure_token_status(prov, t)

            return self._token_pools

    def get_token(self, provider: str = "deepseek", rotate: bool = True) -> Optional[str]:
        """获取指定提供商的有效 Token。

        支持 Round-Robin 轮询调度并自动跳过处于熔断冷却期的 Token。
        若所有 Token 均处于冷却期，则兜底返回最早解封的 Token。
        """
        prov = provider.lower().strip()
        with self._lock:
            tokens = self._token_pools.get(prov, [])
            if not tokens:
                self.load()
                tokens = self._token_pools.get(prov, [])

            if not tokens:
                return None

            now = time.time()
            # 筛选健康未冷却的 Token
            available = [
                t for t in tokens
                if self._ensure_token_status(prov, t)["cooldown_until"] <= now
            ]

            # 熔断兜底：如果全部被封禁/限流冷却，选取离解封时间最近的那个尝试
            if not available:
                available = sorted(
                    tokens,
                    key=lambda t: self._ensure_token_status(prov, t)["cooldown_until"]
                )
                earliest_sec = max(0, int(self._ensure_token_status(prov, available[0])["cooldown_until"] - now))
                logger.warning(
                    f"提供商 [{prov}] 所有 Token 均在冷却中！选取最近解封账号兜底 (距解封约 {earliest_sec}s)"
                )

            if rotate:
                idx = self._token_indices.get(prov, 0)
                selected = available[idx % len(available)]
                self._token_indices[prov] = (idx + 1) % len(available)
            else:
                selected = available[0]

            st = self._ensure_token_status(prov, selected)
            st["last_used"] = now
            return selected

    def get_all_tokens(self, provider: str = "deepseek") -> List[str]:
        """获取指定提供商配置的所有 Token 列表。"""
        prov = provider.lower().strip()
        with self._lock:
            tokens = self._token_pools.get(prov, [])
            if not tokens:
                self.load()
                tokens = self._token_pools.get(prov, [])
            return list(tokens)

    def mark_token_status(
        self,
        provider: str,
        token: str,
        cooldown_seconds: float = 0,
        error: Optional[str] = None,
        is_success: bool = False,
    ) -> None:
        """记录 Token 的请求结果。支持设置冷却倒计时 (用于限流 429 或封禁 403 隔离) 及成功重置。"""
        prov = provider.lower().strip()
        clean_t = self._clean_token_str(token)
        if not clean_t:
            return

        with self._lock:
            st = self._ensure_token_status(prov, clean_t)
            now = time.time()
            if is_success:
                st["success_count"] += 1
                st["cooldown_until"] = 0.0
                st["last_error"] = None
            else:
                st["failure_count"] += 1
                if cooldown_seconds > 0:
                    st["cooldown_until"] = now + cooldown_seconds
                if error:
                    st["last_error"] = str(error)

    def get_pool_status(self, provider: str = "deepseek") -> Dict[str, Any]:
        """获取指定提供商 Token 池的详细健康统计数据。"""
        prov = provider.lower().strip()
        with self._lock:
            tokens = self._token_pools.get(prov, [])
            now = time.time()
            total = len(tokens)
            healthy = 0
            cooling = 0
            token_details = []

            for t in tokens:
                st = self._ensure_token_status(prov, t)
                is_cooling = st["cooldown_until"] > now
                if is_cooling:
                    cooling += 1
                else:
                    healthy += 1

                masked = f"{t[:6]}...{t[-6:]}" if len(t) > 12 else "***"
                token_details.append({
                    "token_masked": masked,
                    "status": "cooling_down" if is_cooling else "healthy",
                    "cooldown_remaining_sec": max(0, int(st["cooldown_until"] - now)) if is_cooling else 0,
                    "success_count": st["success_count"],
                    "failure_count": st["failure_count"],
                    "last_error": st["last_error"],
                    "last_used": st["last_used"],
                })

            return {
                "provider": prov,
                "total": total,
                "healthy": healthy,
                "cooling": cooling,
                "tokens": token_details,
            }

    def save(
        self,
        token: Union[str, List[str]],
        provider: str = "deepseek",
        append: bool = False,
    ) -> None:
        """保存 Token。支持单个字符串、列表追加 (append) 或全量覆盖 (replace)。"""
        prov = provider.lower().strip()

        with self._lock:
            if isinstance(token, list):
                incoming = [self._clean_token_str(t) for t in token if isinstance(t, str)]
                incoming = [t for t in incoming if t]
            else:
                clean_t = self._clean_token_str(token)
                incoming = [clean_t] if clean_t else []

            if not incoming:
                raise ValueError("提供的 Token 不能为空")

            if append:
                existing = self._token_pools.get(prov, [])
                for t in incoming:
                    if t not in existing:
                        existing.append(t)
                self._token_pools[prov] = existing
            else:
                self._token_pools[prov] = incoming

            # 准备写入磁盘的 JSON 结构
            payload: Dict[str, Any] = {}
            for p, t_list in self._token_pools.items():
                if len(t_list) == 1:
                    payload[p] = t_list[0]
                else:
                    payload[p] = list(t_list)

            # 向后兼容写入单一 token 键
            if "deepseek" in self._token_pools and self._token_pools["deepseek"]:
                payload["token"] = self._token_pools["deepseek"][0]

            try:
                self.project_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.project_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                logger.info(f"Token [{prov}] ({len(self._token_pools[prov])} 个) 已保存至 {self.project_file}")
            except Exception as e:
                logger.warning(f"保存至 {self.project_file} 失败: {e}")

            try:
                self.user_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.user_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                logger.info(f"Token [{prov}] ({len(self._token_pools[prov])} 个) 已保存至 {self.user_file}")
            except Exception as e:
                logger.warning(f"保存至 {self.user_file} 失败: {e}")

            if prov == "deepseek":
                try:
                    env_lines = []
                    token_written = False
                    env_val = ",".join(self._token_pools["deepseek"])
                    if self.env_file.exists():
                        with open(self.env_file, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith("DEEPSEEK_BEARER_TOKEN="):
                                    env_lines.append(f'DEEPSEEK_BEARER_TOKEN="{env_val}"\n')
                                    token_written = True
                                else:
                                    env_lines.append(line)

                    if not token_written:
                        env_lines.append(f'DEEPSEEK_BEARER_TOKEN="{env_val}"\n')

                    with open(self.env_file, "w", encoding="utf-8") as f:
                        f.writelines(env_lines)
                except Exception as e:
                    logger.warning(f"更新 {self.env_file} 失败: {e}")

    save_token = save

    @property
    def token(self) -> Optional[str]:
        """获取 DeepSeek 当前默认首选 Token (不触发轮询推进)。"""
        return self.get_token("deepseek", rotate=False)

    @property
    def auth_header(self) -> str:
        """获取全局默认 Authorization 头。"""
        t = self.token
        if not t:
            raise ValueError(
                "未配置 DeepSeek 认证凭证！请通过 credentials.json、.env 或 /api/v1/auth/token 配置。"
            )
        return f"Bearer {t}"

    def auth_header_for(self, token: Optional[str] = None) -> str:
        """为指定 Token 生成 Authorization 头。"""
        t = token or self.token
        if not t:
            raise ValueError(
                "未配置 DeepSeek 认证凭证！请通过 credentials.json、.env 或 /api/v1/auth/token 配置。"
            )
        return f"Bearer {t}"

    def is_authenticated(self, provider: str = "deepseek") -> bool:
        """检查指定提供商是否已有可用凭证。"""
        prov = provider.lower().strip()
        with self._lock:
            tokens = self._token_pools.get(prov, [])
            if not tokens:
                self.load()
                tokens = self._token_pools.get(prov, [])
            return len(tokens) > 0

    def get_all_status(self) -> Dict[str, bool]:
        """获取各提供商的基本认证布尔状态字典。"""
        return {
            "deepseek": self.is_authenticated("deepseek"),
            "qwen": self.is_authenticated("qwen"),
        }


credentials_manager = CredentialsManager()
