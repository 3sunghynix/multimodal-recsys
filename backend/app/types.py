"""커스텀 SQLAlchemy 타입.

EmbeddingVector: PostgreSQL에서는 pgvector의 Vector(dim), 그 외 방언에서는 JSON으로 저장한다.
운영은 PostgreSQL + pgvector이고 테스트는 SQLite라 두 환경이 같은 모델 정의를 공유하도록 분기한다.
"""
from sqlalchemy.types import JSON, TypeDecorator


class EmbeddingVector(TypeDecorator):
    """상품/고객 임베딩용 벡터 컬럼. 코사인 유사도 ANN 검색은 PostgreSQL + pgvector에서만 동작한다."""

    impl = JSON
    cache_ok = True

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(JSON())
