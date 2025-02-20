from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra='ignore'
    )

    APP_ADDRESS: str = "0.0.0.0"
    APP_PORT: int = 8080

    APP_NAME: str
    APP_VERSION: str = "0.0.1"
    LOGGING_LEVEL: str = "DEBUG"

    ENABLE_DEBUG: bool = True

    API_PREFIX: str = "/api"

    OPENAI_API_KEY: str
    OPENAI_ENDPOINT: str
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"

    SYSTEM_INSTRUCTION: str = "Представь, что ты профессионал по поиску информации. Тебе необходимо помогать находить ответы на вопросы по университету ИТМО."

settings = Settings()