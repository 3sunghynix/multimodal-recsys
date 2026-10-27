"""A/B 테스트 할당. 같은 고객 · 같은 실험이면 항상 같은 그룹 (해시 기반, 상태 저장 없음)."""
import hashlib


def assign_variant(user_id: int, experiment_id: int, variants: list[dict]) -> dict:
    total = sum(v["weight"] for v in variants)
    h = int(hashlib.sha256(f"{experiment_id}:{user_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for v in variants:
        acc += v["weight"] / total
        if h <= acc:
            return v
    return variants[-1]
