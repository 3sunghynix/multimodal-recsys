# recsys — 오프라인 학습 · 평가 (담당: 김유환 · 김광석)

서비스(API)는 `backend/app/recommenders/`의 `Recommender` 인터페이스로 모델을 불러 쓴다. 여기서는 모델을 학습하고 산출물(가중치 · 벡터)을 만든다.

## 지금 있는 것

- **평가 모듈** `recsys/eval/`: 평가 프로토콜(고정 분할 · 정답 · 평가 고객)과 지표(Recall · HitRate · NDCG · MAP@k, Coverage, BrandDiversity, IntraListDiversity, PopularityPct, BrandAlignment, UserBrandLoyalty). 정의: [docs/project/지표정의.md](../docs/project/지표정의.md)
- **기준 모델** `recsys/baselines.py`: 인기 기반 (데이터셋별)
- **실행** `python -m recsys.run_eval --data <processed/hm> --source hm [--embeddings <embeddings/v2>] [--out results.json]`
- **테스트** `python -m pytest recsys` (저장소 루트, `pip install -r recsys/requirements.txt`)

## 예정

- **UME 기준 모델:** 논문 코드(`paper/msrs_paper/two_tower.py`, `analysis_yh8.fuse`)를 이식. 텍스트(MiniLM) · 이미지(Marqo-FashionSigLIP) Two-Tower를 따로 학습해 고객별 표준화 점수를 1:1 합산
- **3단계 학습:** 후보 생성(UME) → 순위 모델 → 재정렬(다양성 · 브랜드 충성도) — 정의 문서는 김유환
- **페르소나 6종:** 시뮬레이터 사용자 유형과 같은 정의를 공유 (`simulator/personas.yaml`)
- **평가 모듈 (김광석):** Recall · NDCG · MAP@k + 다양성 · 커버리지 · 브랜드 충성도/다양성 지표. 결과는 API `metric_snapshots` 테이블로 기록해 대시보드 그래프에 쓴다
- **재학습 자동화:** 새 로그가 일정량 쌓이면 학습 → 평가 → 후보 모델 등록 (`/models`)
