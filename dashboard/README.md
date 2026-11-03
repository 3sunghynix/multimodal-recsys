# 관리자 대시보드 v0.3

메뉴는 **A/B 테스트 / 시뮬레이터 / 모니터링 페이지** 3개다. 개요·모델 실험실·데이터 현황·알림/재학습 독립 메뉴는 삭제했다. 기존 오프라인 모델 비교 기능은 A/B 테스트의 추천 평가 탭으로 옮겼다.

- A/B 테스트: 추천 평가(JSON의 MAP/NDCG/Recall·커버리지·다양성·브랜드 일치도), 합성 A/B 결과(p-value·95% 구간), 검색 품질(JSON의 HitRate@50·NDCG@50·MRR@10) 탭. 합성 A/B 검정은 첫 세션에 클릭한 고객 비율을 쓰고, 전체 세션 CTR은 별도 표에 표시한다. 결과를 읽어 시각화하며 여기서 모델을 다시 평가하지 않는다.
- 시뮬레이터: 기존 파일럿 요약과 scale_summary.json의 상품·고객·로그 규모, 3단계 카테고리 상태를 표시한다. 과거 4종 로그는 이전 유형으로 표시하며 현재 7종 로그와 섞지 않는다.
- 모니터링 페이지: monitor_state.json에서 CTR·HitRate·경보·재학습 요청·후보 버전을 표시한다. 실제 학습기·서버 집계·모델 활성화는 아직 연결하지 않았다.

```bash
mkdir -p simulator/runs results/ume
docker compose up --build -d dashboard
```

브라우저에서 http://127.0.0.1:8501 을 연다. simulator/runs/*/scale_summary.json 또는 monitor_state.json을 읽는다. 화면에서 JSON 업로드도 가능하다. 규모 실험을 실행한 뒤 화면을 새로고침한다.

원본 오프라인 결과는 results/ume/hm_test_pca.json, amazon_test_pca.json을 우선 사용한다. 없을 때만 문서 스냅샷으로 표시하고 출처·반올림을 알린다. 서로 다른 source/split/K/고객 집합을 하나의 비교로 합치지 않는다. 브랜드 일치도와 고객별 최다 브랜드 비중은 다른 지표다.

```bash
python -m pip install -r dashboard/requirements-dev.txt
python -m pytest -q dashboard/tests
```

데이터·가중치·원본 결과·생성 로그는 PR에 포함하지 않는다. 화면은 읽기 전용이다. 서비스 이벤트 전송·모델 교체·재학습 실행 버튼은 추가하지 않았다.
