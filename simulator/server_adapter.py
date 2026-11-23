"""서버 연결 API adapter (다음작업 A1 · 이벤트연결규약).

이전 `simulator/server.py` 는 1단계 **read-only 진단** (openapi 체크, 추천만 GET) 이었다. 이 모듈은
그 다음 2단계 — 고객 생성 → 추천 받기 → 행동 이벤트 전송 — 를 서버와 실제로 왕복한다.

전제 (이벤트연결규약 PR #54 ~ #59):
    · POST /users 멱등 (source, source_user_id)
    · GET /recommend/{user_id}?k=K&session_id=S 가 request_id · position · variant 포함 응답
    · POST /events/batch 가 request_id 로 model_version_id · experiment_id · variant 자동 채움,
      event_id 로 재전송 멱등 (같은 내용 duplicate, 다른 내용 409 error)
    · is_setup=True 이벤트는 초기 이력 전용 (집계 제외)

본 adapter 는 결정론 보장:
    · source_user_id · session_id · event_id 모두 (run_id, 인덱스, 종류) 로 uuid5 생성
    · 같은 run_id 로 재실행하면 **같은 고객 · 같은 이벤트가 멱등으로 보존**
    · run_id 바꾸면 완전 분리된 코호트 (A/B 격리)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# 결정론적 event_id · source_user_id 생성용 네임스페이스. 교체해도 안 되지만 호환 위해 고정.
# (uuid5 는 namespace + name → 안정적 UUID; 서버는 이 값을 단순히 UNIQUE 키로 쓴다)
_NS = uuid.UUID("c79e8c37-0bda-4a04-9dcc-27f5c91d1b9e")


@dataclass
class RecommendResponse:
    """GET /recommend 응답의 핵심 필드만 뽑은 요약. 시뮬레이터가 다음 이벤트를 만들 때 쓴다."""
    request_id: str
    model: str
    model_version_id: int | None
    experiment_id: int | None
    variant: str | None
    # (position, item_id) 쌍 — position 은 1-based, 추천 목록 순서 그대로
    impressions: list[tuple[int, int]]


class ServerClient:
    """최소 HTTP wrapper. stdlib urllib 로 외부 의존성 추가 없이 동작한다 (시뮬레이터 컨테이너 가볍게)."""

    def __init__(self, base_url: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(self.base_url + path, data=data, method=method, headers=headers)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except HTTPError as e:
            # 서버 응답 본문을 에러에 포함 — 디버깅 어려움 완화
            detail = e.read().decode("utf-8", errors="replace")[:500] if e.fp else str(e)
            raise RuntimeError(f"{method} {path} failed: HTTP {e.code} · {detail}") from e
        return json.loads(raw) if raw else {}

    # ───── 고객 생성 ─────

    def ensure_customer(self, run_id: str, idx: int) -> int:
        """`source_user_id = uuid5(run_id, "customer", idx)` 로 멱등 생성.
        같은 run_id · idx 재호출 시 서버가 같은 user.id 돌려줌 (POST /users 는 (source, source_user_id) 멱등).

        persona 는 드라이버가 아직 전달하지 않으므로 argument 에서도 뺐다 — 페르소나 엔진을 붙이는
        후속 PR 에서 함께 추가한다 (PR #71 리뷰 반영)."""
        source_user_id = str(uuid.uuid5(_NS, f"{run_id}:customer:{idx}"))
        body = {"source": "sim", "source_user_id": source_user_id}
        row = self._request("POST", "/users", body)
        if "id" not in row:
            raise RuntimeError(f"POST /users returned no id: {row!r}")
        return int(row["id"])

    # ───── 추천 받기 ─────

    def recommend(self, user_id: int, session_id: str, k: int = 10) -> RecommendResponse:
        """세션 단위 추천 — ?session_id= 를 넘겨 그 세션의 클릭이 반영되도록 (이벤트연결규약 §2.4)."""
        path = f"/recommend/{user_id}?k={k}&session_id={session_id}"
        r = self._request("GET", path)
        if r.get("user_id") != user_id or not r.get("request_id"):
            raise RuntimeError(f"Invalid recommend response: user_id={r.get('user_id')} req={r.get('request_id')!r}")
        recs = r.get("recommendations") or []
        impressions = [(int(x["position"]), int(x["product_id"])) for x in recs]
        if sorted(p for p, _ in impressions) != list(range(1, len(impressions) + 1)):
            raise RuntimeError("Recommendation positions must be 1..k without gaps")
        return RecommendResponse(
            request_id=r["request_id"],
            model=str(r.get("model", "")),
            model_version_id=r.get("model_version_id"),
            experiment_id=r.get("experiment_id"),
            variant=r.get("variant"),
            impressions=impressions,
        )

    # ───── 이벤트 전송 ─────

    def event_id(self, run_id: str, customer_idx: int, session_idx: int, impression_key: str) -> str:
        """결정론적 event_id — 같은 (run_id, 고객, 세션, 상품위치, 종류) 재전송하면 서버가 멱등 duplicate 로 처리."""
        name = f"{run_id}:c{customer_idx}:s{session_idx}:{impression_key}"
        return str(uuid.uuid5(_NS, name))

    def send_events_batch(self, events: list[dict]) -> dict:
        """최대 1,000건. 건별 결과 ({created, duplicate, error}) 를 돌려준다. 호출자가 집계."""
        if not events:
            return {"results": [], "created": 0, "duplicate": 0, "error": 0}
        if len(events) > 1000:
            raise ValueError("send_events_batch: 최대 1,000건 (서버 /events/batch 제한)")
        return self._request("POST", "/events/batch", {"events": events})

    # ───── A/B 실험 (선택) ─────

    def ensure_experiment(self, name: str, variants: list[dict]) -> dict:
        """실험 등록 (멱등 아님 — 이미 있으면 서버가 어떻게 처리할지에 따라 다름).
        시뮬레이터는 보통 `POST /experiments` 를 미리 한 번 돌리고 그 experiment_id 로 고객이 자동 배정되게 둔다.
        variants: [{"name": "A", "model_version_id": 1, "weight": 1}, ...] (min 2)."""
        return self._request("POST", "/experiments", {"name": name, "variants": variants})
