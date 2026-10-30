# recsys — 오프라인 학습 · 평가 (담당: 김유환 · 김광석)

서비스(API)는 `backend/app/recommenders/`의 `Recommender` 인터페이스로 모델을 불러 쓴다. 여기서는 모델을 학습하고 산출물(가중치 · 벡터)을 만든다.

## 지금 있는 것

- **평가 모듈** `recsys/eval/`: 평가 프로토콜(고정 분할 · 정답 · 평가 고객)과 지표(Recall · HitRate · NDCG · MAP@k, Coverage, BrandDiversity, IntraListDiversity, PopularityPct, BrandAlignment, UserBrandLoyalty). 정의: [docs/project/지표정의.md](../docs/project/지표정의.md)
- **기준 모델** `recsys/baselines.py`: 인기 기반 (데이터셋별)
- **실행** `python -m recsys.run_eval --data <processed/hm> --source hm [--embeddings <embeddings/v2>] [--out results.json]`
- **테스트** `python -m pytest recsys` (저장소 루트, `pip install -r recsys/requirements.txt`)

## 기준 구현과 온라인 지표 (PR #22)

- **기준 구현** `recsys/eval/reference.py`: 읽기 쉬운 독립 공식으로 빠른 평가 구현을 교차 검사하는 용도다. 모델 실험·운영 평가는 기존 `recsys/eval/metrics.py`와 `python -m recsys.run_eval`을 사용한다.
- **온라인 브랜드 비중** `recsys/eval/online.py`: 이벤트 유형·기간별 고객의 최다 브랜드 비중 평균. DB/API 연동과 중복 제거는 호출자 책임이다.
- **교차 검사** `recsys/tests/test_reference_compatibility.py`: 같은 고객·정답·카탈로그·정규화 벡터에서 정확도·커버리지·다양성 일치 확인. 경계 조건의 정책 차이는 지표 문서에 명시한다.
- `recsys/evaluation` 패키지와 해당 CLI는 제거했다. 예제 JSON `recsys/examples/evaluation_toy.json`은 기준 구현의 가상 검증 자료이며 운영 결과가 아니다.
- [관리자 화면·운영 정책·API 초안](../docs/project/관리자페이지_요구사항_초안.md), [개발·발표 학습 기록](../docs/project/광석_개발학습기록.md)

## 예정

- **UME 기준 모델:** 논문 코드(`paper/msrs_paper/two_tower.py`, `analysis_yh8.fuse`)를 이식. 텍스트(MiniLM) · 이미지(Marqo-FashionSigLIP) Two-Tower를 따로 학습해 고객별 표준화 점수를 1:1 합산
- **3단계 학습:** 후보 생성(UME) → 순위 모델 → 재정렬(다양성 · 브랜드 충성도) — 정의 문서는 김유환
- **페르소나:** 6종 설계 중 4종 우선 파일럿. 시뮬레이터 사용자 유형과 같은 정의를 공유 (`simulator/personas.yaml`)
- **평가·운영 연동 (김광석):** 기존 오프라인 평가와 기준 구현의 일치를 유지하고 온라인 지표를 연결한다. API `metric_snapshots` 결과 저장·조회와 관리자 화면은 후속 작업이다
- **재학습 자동화:** 새 로그가 일정량 쌓이면 학습 → 평가 → 후보 모델 등록 (`/models`)
