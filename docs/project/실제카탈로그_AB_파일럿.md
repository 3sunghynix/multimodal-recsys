# 실제 카탈로그 A/B 파일럿 — 김광석

> 코드: `simulator/catalog_pilot.py`, 테스트: `simulator/tests/test_catalog_pilot.py`, 화면: 대시보드 A/B 테스트 → 합성 A/B 탭
> 관련: [시뮬레이터_실행기.md](시뮬레이터_실행기.md), [페르소나_v0.3.md](페르소나_v0.3.md), [오프라인결과.md](오프라인결과.md)

## 무엇을 하는가

전체 카탈로그(상품 106,825개)가 적재된 DB와 실행 중인 API로 **인기(A) vs UME(B)** 를 비교한다.

1. 페르소나 v0.3 비율로 모의 고객을 만들고, 페르소나 안에서 무작위로 A · B에 반씩 배정한다 (seed 42).
2. 고객마다 DB에 사용자와 초기 구매 이력(`is_setup=True`)을 넣고, 서비스 추천 모델(`app.recommenders`)로 추천 10개를 받는다. B는 active UME 버전, A는 최신 popularity 버전.
3. 고객마다 검색 1회: 75% 텍스트(페르소나별 한 · 영 질의, #39), 25% 이미지(다른 데이터셋 같은 소분류 상품 사진). 실제 `POST /search` 호출.
4. 끝나면 DB 변경을 **rollback** 한다 (실험 고객 · 초기 이력이 서비스 DB에 남지 않음).
5. 추천 목록을 기존 행동 엔진(`simulator.engine.simulate`)에 재생해 view · click · add_to_cart · purchase를 만든다.
6. 주 지표: **첫 세션에 한 번 이상 클릭한 고객 비율**, 두 비율 Z-test · Newcombe 95% 구간 (`simulator/ab.py`). 페르소나별 결과도 같은 방식 (다중 비교 미보정).

## 실행 (Mac, Docker)

전제: 전체 카탈로그 적재 완료 ([로컬실행.md](로컬실행.md)), UME 버전 active, `docker-compose.override.yml`로 카탈로그 폴더를 `/data/full_catalog`에 연결.

```yaml
# docker-compose.override.yml (개인 설정, 커밋하지 않음)
services:
  api:
    volumes:
      - "/Users/<me>/Downloads/full_catalog:/data/full_catalog"
    environment:
      IMAGES_ROOT: /data/full_catalog/images
```

```bash
docker compose up --build -d
curl -s localhost:8000/items/version          # {"count":106825,...}
docker compose run --rm --no-deps \
  -v "$PWD/simulator:/app/simulator" \
  -v "$PWD/recsys:/app/recsys:ro" \
  api python -m simulator.catalog_pilot \
    --customers 10000 --seed 42 \
    --out /app/simulator/runs/catalog_pilot_10k \
    --api http://api:8000
```

- 이미지는 `full_catalog/images/{source}/{item_id}.jpg`로 **압축을 푼 상태**여야 한다 (업로드에 파일을 직접 연다).
- 출력 폴더가 이미 있으면 실행하지 않는다. 결과는 `simulator/runs/<이름>/`, 대시보드가 바로 읽는다.
- 실행 시간: 고객당 약 0.45초 (M 시리즈 MacBook, CPU). 1,000명 7분 25초, 10,000명 약 75분.

### C 그룹 (MAB 탐색 슬롯)

`--arms ABC --batches 10 --new-items 200`을 주면 C = UME + MAB 탐색 슬롯이 추가되고 묶음마다 탐색이 학습된다. 설정 · 해석: [MAB_탐색.md](MAB_탐색.md)

## 출력

| 파일 | 내용 |
|---|---|
| `catalog_ab_summary.json` | 전체 · 페르소나별 검정, 그룹별 행동 수, 신규 고객 · UME 대체 수, 추천된 서로 다른 상품 수, 실행 시간 |
| `customers.jsonl` | 모의 고객 (페르소나 · 선호 · 초기 이력) |
| `recommendations.jsonl` | 고객별 추천 목록, 모델 · 버전 · 대체 사유 |
| `events.jsonl` | 검색 + 행동 이벤트 |
| `decisions.jsonl` | 노출별 특성 · 선호 점수 · 확률 · 난수 |

## 결과 읽을 때 주의

- **모의 행동**이다. 클릭 확률은 페르소나 규칙(카테고리 · 브랜드 · 가격 · 인기 · 신규성 적합도)으로 정해지며 실제 사용자 반응을 추정한 값이 아니다.
- 눈팅형은 설계상 초기 이력이 0개인 고객이 약 절반이다 (`size_choices: [0, 1, 2]`). 이들은 B에서도 UME 대신 인기 목록을 받는다 (명세의 신규 고객 대체). `ume_fallback_count`는 버그가 아니라 이 경우다.
- 페르소나별 차이는 다중 비교 미보정이며 표본이 작으면 p-value를 내지 않는다 (그룹당 50명 미만 또는 기대 빈도 5 미만).
- 서버 `/experiments` 배정 · 집계를 거치지 않는다. 실험 고객은 rollback 되므로 `GET /experiments/{id}/metrics`에는 나타나지 않는다.
- 추천은 HTTP `GET /recommend`가 아니라 같은 추천 모델 코드를 직접 호출한다 (DB에 넣은 초기 이력을 rollback 전에 쓰기 위해). 그래서 이 스크립트는 API 컨테이너 안에서 실행한다.

## 10,000명 결과 (2026-11-10, `catalog_pilot_10k`)

고객 10,000명 (A 5,000 · B 5,000), seed 42, 실행 71.6분 (고객당 0.43초). 요약 원본: [catalog_pilot_10k_summary.json](catalog_pilot_10k_summary.json) (`simulator/runs/`는 git에 올리지 않으므로 요약만 복사).

**전체: 인기 34.6% vs UME 36.2%, 차이 +1.6%p [−0.2, +3.5], p = 0.086 → 유의하지 않음 (α = 0.05).**
전체 세션 CTR 인기 4.76% vs UME 4.94%. 추천된 서로 다른 상품 수 인기 10 · UME 4,818.

| 페르소나 | 고객 A/B | 인기 | UME | 차이 [95% 구간] | p |
|---|---:|---:|---:|---:|---:|
| 브랜드충성 | 600/600 | 0.202 | 0.333 | **+0.132** [+0.082, +0.181] | <0.001 |
| 실용주의자 | 900/900 | 0.146 | 0.206 | **+0.060** [+0.025, +0.095] | 0.0008 |
| 충동구매 | 400/400 | 0.772 | 0.782 | +0.010 [−0.048, +0.068] | 0.73 |
| 가성비추구 | 900/900 | 0.216 | 0.214 | −0.001 [−0.039, +0.037] | 0.95 |
| 신중탐색 | 850/850 | 0.402 | 0.396 | −0.006 [−0.052, +0.041] | 0.80 |
| 눈팅형 | 750/750 | 0.305 | 0.272 | −0.033 [−0.079, +0.013] | 0.15 |
| 트렌드세터 | 600/600 | 0.672 | 0.632 | −0.040 [−0.094, +0.014] | 0.15 |

- 눈팅형 이력 0개 고객 785명 중 B에 배정된 396명은 UME 대신 인기 목록을 받았다 (`ume_fallback_count` 396 / 5,000).
- 페르소나별 p는 다중 비교 미보정이다. 7개 비교에 Bonferroni(α = 0.05/7 ≈ 0.007)를 적용해도 브랜드충성 · 실용주의자의 개선은 유의하다.

**해석.** UME는 고객의 선호 카테고리 · 브랜드에 맞는 상품을 보여 주고(1,000명 파일럿 기준 카테고리 적합도 평균 0.44 vs 0.06), 그 적합도를 중시하는 브랜드충성 · 실용주의자에서 클릭이 뚜렷이 늘었다. 인기 · 신규성을 중시하는 트렌드세터 · 눈팅형에서는 작게 줄어 전체 효과가 희석됐다. 오프라인에서 UME가 MAP 2 ~ 3배인 것과 달리 모의 온라인 전체 효과는 작다 — 오프라인 지표는 "다음에 산 상품"을 맞히는지만 보지만, 모의 고객의 클릭은 페르소나마다 다른 이유(적합도 vs 인기 · 신규성)로 일어나기 때문이다. 이력이 1 ~ 2개뿐인 눈팅형에게 UME 개인화가 약한 점은 명세의 신규 고객 규칙(이력 5개 미만 → 인기 + 트렌딩)과 Re-ranking · MAB 탐색 슬롯 작업으로 이어진다.

1,000명 파일럿(`catalog_pilot_1k`)의 전체 결과는 37.6% vs 36.8%, p = 0.79로 같은 방향의 페르소나별 패턴(브랜드충성 · 실용주의자 +, 눈팅형 −)을 보였다.
