import json
import logging
from pathlib import Path
from typing import Dict, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class CredentialsManager:
    def __init__(self):
        self.project_file = settings.CREDENTIALS_PATH
        self.user_file = settings.USER_CREDENTIALS_PATH
        self.env_file = settings.PROJECT_ROOT / ".env"
        self._tokens: Dict[str, str] = {}
        self.load()

    def _read_from_file(self, path: Path) -> Dict[str, str]:
        tokens = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, str) and v.strip():
                                clean_v = v.strip()
                                if clean_v.startswith("Bearer "):
                                    clean_v = clean_v[7:].strip()
                                tokens[k.lower()] = clean_v

                        single_token = data.get("token") or data.get("authorization") or data.get("ds_session_id")
                        if single_token and "deepseek" not in tokens:
                            st = single_token.strip()
                            if st.startswith("Bearer "):
                                st = st[7:].strip()
                            tokens["deepseek"] = st
            except Exception as e:
                logger.debug(f"读取凭证文件失败 {path}: {e}")
        return tokens

    def load(self) -> Dict[str, str]:
        self._tokens = {}

        proj_tokens = self._read_from_file(self.project_file)
        self._tokens.update(proj_tokens)

        user_tokens = self._read_from_file(self.user_file)
        for k, v in user_tokens.items():
            if k not in self._tokens:
                self._tokens[k] = v

        if settings.DEEPSEEK_BEARER_TOKEN and "deepseek" not in self._tokens:
            t = settings.DEEPSEEK_BEARER_TOKEN.strip()
            if t.startswith("Bearer "):
                t = t[7:].strip()
            self._tokens["deepseek"] = t

        return self._tokens

    def get_token(self, provider: str = "deepseek") -> Optional[str]:
        if not self._tokens:
            self.load()
        return self._tokens.get(provider.lower().strip())

    def save(self, token: str, provider: str = "deepseek") -> None:
        prov = provider.lower().strip()
        clean_token = token.strip()
        if clean_token.startswith("Bearer "):
            clean_token = clean_token[7:].strip()

        self._tokens[prov] = clean_token

        payload = dict(self._tokens)
        if prov == "deepseek":
            payload["token"] = clean_token

        try:
            self.project_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.project_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"Token [{prov}] 已保存至 {self.project_file}")
        except Exception as e:
            logger.warning(f"保存至 {self.project_file} 失败: {e}")

        try:
            self.user_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.user_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"Token [{prov}] 已保存至 {self.user_file}")
        except Exception as e:
            logger.warning(f"保存至 {self.user_file} 失败: {e}")

        if prov == "deepseek":
            try:
                env_lines = []
                token_written = False
                if self.env_file.exists():
                    with open(self.env_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("DEEPSEEK_BEARER_TOKEN="):
                                env_lines.append(f'DEEPSEEK_BEARER_TOKEN="{clean_token}"\n')
                                token_written = True
                            else:
                                env_lines.append(line)

                if not token_written:
                    env_lines.append(f'DEEPSEEK_BEARER_TOKEN="{clean_token}"\n')

                with open(self.env_file, "w", encoding="utf-8") as f:
                    f.writelines(env_lines)
            except Exception as e:
                logger.warning(f"更新 {self.env_file} 失败: {e}")

    @property
    def token(self) -> Optional[str]:
        return self.get_token("deepseek")

    @property
    def auth_header(self) -> str:
        t = self.token
        if not t:
            raise ValueError(
                "未配置 DeepSeek 认证凭证！请通过 credentials.json、.env 或 /api/v1/auth/token 配置。"
            )
        return f"Bearer {t}"

    def is_authenticated(self, provider: str = "deepseek") -> bool:
        return bool(self.get_token(provider))


credentials_manager = CredentialsManager()
