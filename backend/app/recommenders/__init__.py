"""추천 모델 레지스트리. 새 모델은 Recommender를 상속하고 @register로 등록하면 /models · A/B 테스트에서 바로 쓸 수 있다.

ume: 텍스트 · 이미지 Two-Tower 점수 합산 + 인기 · 브랜드 가산 (recommenders/ume.py, 학습은 recsys/ume.py).
"""
from .base import REGISTRY, Recommender, get, register  # noqa: F401
from . import popularity, ume  # noqa: F401  모델 등록
