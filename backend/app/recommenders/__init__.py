"""추천 모델 레지스트리. 새 모델은 Recommender를 상속하고 @register로 등록하면 /models · A/B 테스트에서 바로 쓸 수 있다.

예정: UME(텍스트 · 이미지 Two-Tower 점수 합산, 논문 paper/msrs_paper 이식), 페르소나 · 3단계 모델.
"""
from .base import REGISTRY, Recommender, get, register  # noqa: F401
from . import popularity  # noqa: F401  기본 모델 등록
