from sqlalchemy.orm import Session

REGISTRY: dict[str, type["Recommender"]] = {}


class Recommender:
    """모든 추천 모델의 공통 인터페이스.

    name: 레지스트리 이름 (model_versions.name과 같아야 한다)
    recommend: 고객 한 명에게 상품 id k개를 점수 순으로 돌려준다. params는 model_versions.params.
    recommend_scored: (id, score) 쌍 — 응답의 recommendations[].score 로 노출된다.
        모델이 실제 점수를 돌려주면 override 하고 scored_native=True 로 표시한다.
        모델이 점수를 돌려주지 않으면 base 가 순위 기반 placeholder (1.0 → 0.0 감소) 를 돌려주고,
        응답에서는 score_kind="rank" 로 라벨된다.
    """

    name: str = ""
    scored_native: bool = False       # True = 모델이 실제 점수를 돌려줌, False = base 가 순위 기반 폴백

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def recommend(self, db: Session, user_id: int, k: int) -> list[int]:
        raise NotImplementedError

    def recommend_scored(self, db: Session, user_id: int, k: int) -> list[tuple[int, float]]:
        """기본: recommend() 로 id 를 받고 순위 기반 placeholder 점수 부여.
        실제 모델 점수가 있는 recommender 는 이 메서드를 override 한다."""
        ids = self.recommend(db, user_id, k)
        n = len(ids)
        return [(i, round(1 - (idx / max(n, 1)), 4)) for idx, i in enumerate(ids)]


def register(cls):
    assert cls.name and cls.name not in REGISTRY, f"이름이 비었거나 중복: {cls.name}"
    REGISTRY[cls.name] = cls
    return cls


def get(name: str, params: dict | None = None) -> Recommender:
    if name not in REGISTRY:
        raise KeyError(f"등록되지 않은 모델: {name} (등록된 모델: {sorted(REGISTRY)})")
    return REGISTRY[name](params)
