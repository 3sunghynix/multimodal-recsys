"""환경 설정. 값은 환경 변수 또는 .env에서 읽는다 (루트 .env.example 참고)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 로컬 개발 기본값은 SQLite. docker-compose에서는 PostgreSQL 주소로 덮어쓴다
    database_url: str = "sqlite:///./local.db"
    default_k: int = 20

    # 이미지 임베딩. OpenAI CLIP ViT-B/32는 512차원. 모델을 바꾸면 embedding_dim도 함께 변경한다.
    image_encoder_model: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    # 서버 시작 시 CLIP 모델을 미리 로드할지. 테스트에서는 False(기본은 True in docker)
    load_image_encoder: bool = True
    # /search/image 업로드 크기 상한 (bytes). 기본 10MB
    max_image_bytes: int = 10 * 1024 * 1024


settings = Settings()
