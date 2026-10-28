"""커스텀 SQLAlchemy 타입.

EmbeddingVector: PostgreSQL에서는 pgvector의 Vector(dim), 그 외 방언에서는 JSON으로 저장한다.
운영은 PostgreSQL + pgvector이고 테스트는 SQLite라 두 환경이 같은 모델 정의를 공유하도록 분기한다.

입력(bind): numpy ndarray와 list[float] 둘 다 받는다. numpy는 .tolist()로 변환해
JSON serializer(SQLite 경로)와 pgvector adapter(PostgreSQL 경로) 모두 안전하게 처리한다.
출력(result): 방언에 따라 pgvector는 numpy, JSON은 list를 돌려주므로 **list로 통일**한다.
(pydantic 응답·JSON 직렬화 호환이 쉬움)
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

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if hasattr(value, "tolist"):   # numpy ndarray 등
            return value.tolist()
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if hasattr(value, "tolist"):   # PostgreSQL + pgvector 경로는 numpy로 돌아옴
            return value.tolist()
        return value
