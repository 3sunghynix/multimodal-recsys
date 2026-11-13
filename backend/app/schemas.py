from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal["view", "click", "add_to_cart", "purchase", "search"]
SearchType = Literal["text", "image", "hybrid"]


class ItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    source: str
    brand_id: int | None
    brand_name: str | None = None          # brands.name JOIN 결과 — 카드에 "BRAND #10" 대신 실제 브랜드명 노출
    title: str
    category: str
    price: float | None
    image_path: str | None


class ItemDetailOut(ItemOut):
    """상품 상세 페이지용. 카드용 ItemOut 에 설명 · 외부 상품 id 를 더한다.
    배치 조회(GET /items)는 응답 가벼운 ItemOut 유지, 단건 조회(GET /items/{id})만 이 스키마로 바꾼다."""
    description: str = ""
    source_item_id: str


class SearchHit(ItemOut):
    score: float                      # 결합 점수 (merge 방식에 따라 척도가 다르다. 순서만 의미 있음)


class SearchResultOut(BaseModel):
    """프로젝트 명세 2-5 "검색 API 필수 응답 필드" 의 results[] 아이템 규격."""
    product_id: str                   # 서버 내부 int id 를 문자열로 노출 (명세 샘플 "P12345")
    name: str                         # 명세 요구명 — 내부적으로는 Item.title 과 동일
    score: float
    price: float | None


class SearchOut(BaseModel):
    # ───── 명세(2-5) 필수 응답 필드 ─────
    search_type: SearchType           # text · image · hybrid 중 무엇으로 돌았는지
    results: list[SearchResultOut]    # 명세 필수: product_id · name · score · price 만
    latency_ms: float                 # 핸들러 시작부터 응답 직전까지 벽시계 시간
    total_count: int                  # 반환된 결과 수 (페이지네이션 없음)

    # ───── 데모 프론트/디버깅용 확장 필드 (명세 외) ─────
    query: str | None                 # 받은 검색어
    query_en: str | None              # 인코더에 넣은 영어 검색어 (한국어면 사전 번역 결과)
    untranslated: list[str]           # 사전에 없어 번역하지 못한 한국어 단어 (화면 안내용)
    weights: dict[str, float]         # text (MiniLM → 상품 텍스트) · text_image (Marqo 텍스트 → 상품 이미지) · image (업로드 이미지)
    merge: str
    items: list[SearchHit]            # 프론트가 쓰는 전체 상품 상세 (brand/image_path 등)


class EventIn(BaseModel):
    user_id: int
    type: EventType
    item_id: int | None = None
    query: str | None = None
    model_version_id: int | None = None
    experiment_id: int | None = None
    variant: str | None = None
    # 클라이언트가 생성한 재전송 멱등 키. 네트워크 재시도 · 시뮬레이터 재실행에 쓴다. 생략 시 서버가 UUID4 생성.
    event_id: UUID | None = None
    # 신규 고객의 초기 이력 주입. True이면 A/B · CTR 집계에서 제외.
    is_setup: bool = False
    # 클라이언트 세션 id (선택). 있으면 현재 세션을 이 값으로 구분 (없으면 서버 시각 30분 규칙). 이벤트연결규약 2.4
    session_id: str | None = Field(None, max_length=64)


class EventOut(EventIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    event_id: UUID
    created_at: datetime


class UserIn(BaseModel):
    # (source, source_user_id)가 같으면 같은 user를 돌려주는 멱등 생성. source_user_id가 None이면 익명 신규.
    source: str = "sim"
    source_user_id: str | None = None
    persona: str | None = None


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    source: str
    source_user_id: str | None
    persona: str | None


class RecommendationOut(BaseModel):
    """프로젝트 명세 2-5 "추천 API 필수 응답 필드" 의 recommendations[] 아이템 규격."""
    product_id: str                   # 서버 내부 int id 를 문자열로 노출
    score: float                      # 모델이 돌려준 점수 (score_kind="model") 또는 순위 placeholder (score_kind="rank")
    score_kind: Literal["model", "rank"]  # 자동 채점·리뷰에서 "모델 점수"로 오해되지 않도록 출처 명시
    # 점수 단위. 같은 score_kind="model" 이라도 경로마다 단위가 달라 값끼리 비교하면 안 된다 (recommenders.base 참고)
    score_basis: Literal["event_weight_sum", "ume_zscore", "rank_percentile", "rank_placeholder"]
    reason: str                       # 모델별 간단 라벨 (popular · brand_match · personalized)
    is_exploration: bool              # MAB 탐색 슬롯 여부 (model_versions.params["explore"] 설정 시 1 ~ 2개, services/explore.py)


class PipelineLatency(BaseModel):
    """명세 요구 pipeline_latency. UME: candidate = 타워 점수 계산, ranking = 인기 · 브랜드 · 데이터셋 결합,
    reranking = 다양성 재정렬. 단계를 나누지 않는 모델(인기 등)은 candidate_ms 에 전체 시간, 나머지 0.
    total_ms 는 요청 처리 전체(고객 확인 · 모델 선택 · 세션 컨텍스트의 DB · Redis 조회 포함)라 세 단계 합보다 크다."""
    candidate_ms: float
    ranking_ms: float
    reranking_ms: float
    total_ms: float


class SessionContext(BaseModel):
    """명세 요구 session_context. recent_clicks: 최근 클릭 상품 id (세션과 무관하게 최근 5개).
    session_interest: 현재 세션(30분 안) 클릭 상품의 가장 많은 소분류 (예: "니트·스웨터"), 세션이 없으면 None."""
    recent_clicks: list[str]
    session_interest: str | None


class RecommendOut(BaseModel):
    user_id: int
    model: str
    model_version_id: int | None
    experiment_id: int | None = None
    variant: str | None = None

    # ───── 명세(2-5) 필수 응답 필드 ─────
    recommendations: list[RecommendationOut]
    pipeline_latency: PipelineLatency
    session_context: SessionContext

    # ───── 하위호환 확장 필드 (시뮬레이터 bridge · 데모 프론트가 읽음) ─────
    items: list[int]                  # recommendations[].product_id 와 같은 순서의 int id 목록


class ModelVersionIn(BaseModel):
    name: str
    version: str = "v1"
    params: dict = Field(default_factory=dict)


class ModelVersionOut(ModelVersionIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    status: str


class Variant(BaseModel):
    name: str
    model_version_id: int
    weight: float = Field(gt=0)


class ExperimentIn(BaseModel):
    name: str
    variants: list[Variant] = Field(min_length=2)


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    status: str
    variants: list[Variant]


class VariantMetrics(BaseModel):
    variant: str
    users: int
    impressions: int
    clicks: int
    purchases: int
    ctr: float
    ctr_ci_95: tuple[float, float] = (0.0, 1.0)   # Wilson score 95% CI (CTR 자체의 신뢰구간)


class ABComparison(BaseModel):
    """두 변량의 지표 비교 (CTR · CVR). baseline 대비 treatment 의 차이와 통계 검정 결과."""
    metric: Literal["ctr", "cvr"]
    baseline: str
    treatment: str
    baseline_rate: float
    treatment_rate: float
    diff: float                                   # treatment_rate - baseline_rate
    lift: float | None                            # diff / baseline_rate (baseline=0 이면 None)
    diff_ci_95: tuple[float, float]               # 차이의 Wald 95% CI
    p_value: float                                # 양측 두 비율 Z-test
    test: Literal["two_proportion_z"]
    significant: bool                             # p_value < 0.05


class ExperimentStats(BaseModel):
    """실험 전체 요약: 그룹별 지표 + 그룹 간 통계 검정 결과."""
    experiment_id: int
    experiment_name: str
    variants: list[VariantMetrics]
    comparisons: list[ABComparison]


# ───── Continuous Training (명세 4.6) 지원 ─────

class MetricSnapshotIn(BaseModel):
    """오프라인 평가 결과 · 모델 교체 전후 비교 결과 저장. 모델 재학습 pipeline 과 대시보드가 쓴다."""
    model_version_id: int | None = None
    experiment_id: int | None = None
    variant: str | None = None
    metric: str                                   # "recall@50", "hit_rate", "ctr", "ndcg@10" 등 자유 라벨
    value: float
    n: int = 0                                    # 지표 산출에 쓰인 샘플 수 (있으면 CI 계산 등에 활용)


class MetricSnapshotOut(MetricSnapshotIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime


class MetricWindowOut(BaseModel):
    """monitoring/runner.py observe() 가 요구하는 window JSON 포맷 (광석 PR #37).
    서버가 지정 시간 구간에서 집계해서 돌려주면 CT runner 가 cron 으로 받아 임계값 체크 · 재학습 트리거.
    - setup_excluded 는 항상 True (PR #30 규칙 일관).
    - views · clicks · evaluated_users · hit_users 는 window 안의 non-setup 이벤트에서 집계.
    - cumulative_new_logs 는 전체 non-setup 이벤트 누적 (window_end 시점까지).
    """
    window_id: str
    window_start: str
    window_end: str
    model_version: str                            # 호출자가 지정한 라벨 (e.g. "popularity-v1")
    source: str
    cohort: str
    metric_definition_version: str
    setup_excluded: bool
    views: int
    clicks: int
    evaluated_users: int
    hit_users: int
    cumulative_new_logs: int


class LogDeltaOut(BaseModel):
    """since_event_id 이후로 쌓인 non-setup 이벤트 수. CT runner 가 "재학습 임계값(예: 10000)" 비교에 쓴다.
    서버는 수치만 제공하고 임계값 판단은 runner 가 한다 (policy 는 runner 책임)."""
    since_event_id: int
    max_event_id: int
    delta: int                                    # 지정 id 초과 ~ max_event_id 구간의 non-setup 이벤트 수
    reset_detected: bool                          # max_event_id < since_event_id → --reset 재적재 가능성, runner 가 기준점 재설정
