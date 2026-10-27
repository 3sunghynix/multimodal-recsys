"""환경 설정. 값은 환경 변수 또는 .env에서 읽는다 (루트 .env.example 참고)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 로컬 개발 기본값은 SQLite. docker-compose에서는 PostgreSQL 주소로 덮어쓴다
    database_url: str = "sqlite:///./local.db"
    default_k: int = 20


settings = Settings()
