from sqlalchemy.orm import Session

REGISTRY: dict[str, type["Recommender"]] = {}


class Recommender:
    """모든 추천 모델의 공통 인터페이스.

    name: 레지스트리 이름 (model_versions.name과 같아야 한다)
    recommend: 고객 한 명에게 상품 id k개를 점수 순으로 돌려준다. params는 model_versions.params.
    """

    name: str = ""

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def recommend(self, db: Session, user_id: int, k: int) -> list[int]:
        raise NotImplementedError


def register(cls):
    assert cls.name and cls.name not in REGISTRY, f"이름이 비었거나 중복: {cls.name}"
    REGISTRY[cls.name] = cls
    return cls


def get(name: str, params: dict | None = None) -> Recommender:
    if name not in REGISTRY:
        raise KeyError(f"등록되지 않은 모델: {name} (등록된 모델: {sorted(REGISTRY)})")
    return REGISTRY[name](params)
