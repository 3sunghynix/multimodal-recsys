"""색상 변형 묶기 (다음작업 C1).

H&M `source_item_id`는 article_id(10자리)이고 앞 7자리가 product_code다 — 같은 옷의 색상 · 무늬 변형은 product_code가 같다
(예: 0372860001 · 0372860002 · 0372860069). 추천 목록에서는 같은 옷을 한 번만 보여 주고, 나머지 색상은
`variant_ids`로 함께 돌려줘 화면이 "다른 색상 N개"로 보여 준다.

근거 (11-18, 전체 카탈로그 서비스 로그 leave-last-out, H&M 평가 고객 5,000명, 운영 가중치):
목록 12개 중 평균 4.6개가 같은 옷의 변형이었다. 1개로 묶으면 정확 일치 MAP@12 0.0425 → 0.0340 (−20%)이지만
같은 옷을 맞힌 비율(코드 단위 Recall@12)은 0.162 → 0.198 (+22%). 변형은 `variant_ids`로 고를 수 있으므로 코드 단위를 택했다.
Amazon 등 다른 데이터셋은 변형 정보가 없어 묶지 않는다.
"""


def variant_key(source: str | None, source_item_id: str | None) -> str:
    """같은 옷이면 같은 키, 묶지 않을 상품은 ""."""
    if source == "hm" and source_item_id and len(source_item_id) == 10 and source_item_id.isdigit():
        return f"hm:{source_item_id[:7]}"
    return ""


def first_per_variant(ids, keys, k: int, per_variant: int = 1) -> list:
    """점수 순 ids에서 같은 키는 per_variant개까지만 골라 k개. 모자라면 건너뛴 것으로 순서대로 채운다
    (결과는 항상 min(k, len(ids))개). keys[i] == ""이면 제한 없음."""
    picked, dup, count = [], [], {}
    for i, key in zip(ids, keys):
        if key and count.get(key, 0) >= per_variant:
            dup.append(i)
            continue
        picked.append(i)
        if key:
            count[key] = count.get(key, 0) + 1
        if len(picked) == k:
            return picked
    return picked + dup[:k - len(picked)]
