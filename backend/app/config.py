"""환경 설정. 값은 환경 변수 또는 .env에서 읽는다 (루트 .env.example 참고)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 로컬 개발 기본값은 SQLite. docker-compose에서는 PostgreSQL 주소로 덮어쓴다
    database_url: str = "sqlite:///./local.db"
    default_k: int = 20

    # 이미지 임베딩. Marqo-FashionSigLIP(open_clip) → L2 → PCA 256 → L2.
    # 상품(items.image_embedding)은 drive의 embeddings/v2가 같은 파이프라인으로 생성됐다.
    # 모델을 바꾸면 embedding_dim과 pca_image_path를 함께 조정한다.
    image_encoder_model: str = "hf-hub:Marqo/marqo-fashionSigLIP"
    embedding_dim: int = 256
    # 쿼리를 상품과 같은 공간으로 맞추기 위한 PCA 행렬 (mean · components). npz 포맷.
    # 드라이브: embeddings/v2/pca_image.npz. docker-compose의 ./data:/data 마운트로 접근.
    # 파일이 없으면 encoder는 로드되지만 /search/image는 503(PCA 미설치)으로 응답.
    pca_image_path: str = "/data/embeddings/v2/pca_image.npz"
    # 서버 시작 시 모델을 미리 로드할지. 테스트에서는 False(기본은 True in docker)
    load_image_encoder: bool = True
    # /search/image 업로드 크기 상한 (bytes). 기본 10MB
    max_image_bytes: int = 10 * 1024 * 1024


settings = Settings()
