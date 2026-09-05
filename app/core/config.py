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

    # Long-Context & Compression Settings (1,000,000 token window -> 300,000 threshold)
    MAX_CONTEXT_TOKENS: int = Field(default=300_000, description="Порог сжатия контекста (токенов)")
    CONTEXT_COMPRESSION_ENABLED: bool = Field(default=True, description="Включить интеллектуальное сжатие контекста")
    RETAIN_RECENT_MESSAGES_COUNT: int = Field(default=12, description="Количество последних сообщений без сжатия")
    MAX_TOOL_OUTPUT_TOKENS: int = Field(default=25_000, description="Максимальный размер отдельного вывода инструмента")

    # Proxy Session Mode ('single' or 'multi')
    PROXY_MODE: str = Field(
        default="multi",
        description="Режим работы сессий: 'single' (единая сессия без создания новых чатов) или 'multi' (новый чат на запрос)"
    )
    SINGLE_SESSION_MODE: bool = Field(
        default=False,
        description="Работать в рамках одной постоянной сессии, предотвращая частые вызовы chat_session/create и ошибку 429"
    )


settings = Settings()
