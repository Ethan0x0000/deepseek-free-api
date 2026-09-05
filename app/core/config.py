from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "DeepSeek Web API Proxy"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8317

    # Project paths
    PROJECT_ROOT: Path = BASE_DIR
    CREDENTIALS_PATH: Path = BASE_DIR / "credentials.json"
    USER_CREDENTIALS_PATH: Path = Path.home() / ".deepseek" / "credentials.json"

    # DeepSeek Web API settings
    DEEPSEEK_BASE_URL: str = "https://chat.deepseek.com"
    DEEPSEEK_BEARER_TOKEN: str = Field(default="", description="Bearer token without 'Bearer ' prefix")
    
    # Client emulation headers
    CLIENT_BUNDLE_ID: str = "com.deepseek.chat"
    CLIENT_VERSION: str = "2.4.0"
    CLIENT_LOCALE: str = "zh-CN"
    CLIENT_PLATFORM: str = "web"
    CLIENT_TIMEZONE_OFFSET: str = "28800"  # UTC+8 (Asia/Shanghai)
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    
    # Request timeouts
    REQUEST_TIMEOUT: float = 180.0

    # 上下文长度与智能压缩设置 (1,000,000 Token 窗口 -> 300,000 阈值)
    MAX_CONTEXT_TOKENS: int = Field(default=300_000, description="上下文压缩阈值 (Token 数量)")
    CONTEXT_COMPRESSION_ENABLED: bool = Field(default=True, description="是否启用智能上下文压缩")
    RETAIN_RECENT_MESSAGES_COUNT: int = Field(default=12, description="保留最近不被压缩的消息轮数")
    MAX_TOOL_OUTPUT_TOKENS: int = Field(default=25_000, description="单个工具执行结果的最大允许 Token 数量")

    # 会话模式 ('single' 单会话 或 'multi' 隔离会话)
    PROXY_MODE: str = Field(
        default="multi",
        description="会话工作模式: 'single' (单会话复用) 或 'multi' (每请求独立临时会话，防止上下文爆炸)"
    )
    SINGLE_SESSION_MODE: bool = Field(
        default=False,
        description="是否在单会话内持续累积消息 (非 Agent 编程场景使用)"
    )
    AUTO_CLEAN_WEB_SESSIONS: bool = Field(
        default=True,
        description="请求结束后是否在后台自动删除网页端的临时会话，保持网页左侧对话列表干净整洁"
    )


settings = Settings()
