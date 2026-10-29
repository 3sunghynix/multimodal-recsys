# recsys — 오프라인 학습 · 평가 (담당: 김유환 · 김광석)

서비스(API)는 `backend/app/recommenders/`의 `Recommender` 인터페이스로 모델을 불러 쓴다. 여기서는 모델을 학습하고 산출물(가중치 · 벡터)을 만든다.

## 독립 평가 모듈 v0.1

DB·모델·추가 패키지 없이 실행하는 계산 모듈을 `evaluation/`에 둔다.
Recall/NDCG/MAP@K, 카탈로그 커버리지, 고정 콘텐츠 벡터의 목록 내 다양성,
이벤트 유형별 고객의 최다 브랜드 비중을 계산한다.

저장소 루트에서:

```bash
python -m unittest discover -s recsys/tests -v
python -m recsys.evaluation recsys/examples/evaluation_toy.json
python -m recsys.evaluation recsys/examples/evaluation_toy.json --output /tmp/msrs-evaluation-new.json
```

출력 파일이 이미 있으면 덮어쓰지 않는다. 예제는 손으로 만든 데이터이며 실제 모델 성능이 아니다.
실제 split·긍정 feedback·warm-start 고객 선택은 입력 준비 단계에서 명시해야 한다.
아직 parquet 어댑터·API·DB 기록·신뢰구간은 포함하지 않았다.

- [지표 정의서](../docs/project/지표정의서.md)
- [관리자 화면·운영 정책·API 초안](../docs/project/관리자페이지_요구사항_초안.md)
- [개발·발표 학습 기록](../docs/project/광석_개발학습기록.md)

## 예정

- **UME 기준 모델:** 논문 코드(`paper/msrs_paper/two_tower.py`, `analysis_yh8.fuse`)를 이식. 텍스트(MiniLM) · 이미지(Marqo-FashionSigLIP) Two-Tower를 따로 학습해 고객별 표준화 점수를 1:1 합산
- **3단계 학습:** 후보 생성(UME) → 순위 모델 → 재정렬(다양성 · 브랜드 충성도) — 정의 문서는 김유환
- **페르소나 6종:** 시뮬레이터 사용자 유형과 같은 정의를 공유 (`simulator/personas.yaml`)
- **평가 모듈 (김광석):** 순수 계산 함수·테스트는 v0.1 구현. 실제 데이터 어댑터와 API `metric_snapshots` 결과 저장을 연결해 대시보드 그래프에 쓰는 작업은 예정
- **재학습 자동화:** 새 로그가 일정량 쌓이면 학습 → 평가 → 후보 모델 등록 (`/models`)
