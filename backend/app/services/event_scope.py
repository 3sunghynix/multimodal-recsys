"""이벤트를 어디에 세는가 — 서비스 통계(인기)와 재학습 카운터가 같은 규칙을 쓰도록 한 곳에 모은다.

- 노출(impression): 추천 목록에 보인 상품 = `type="view"` + `request_id` 있음 (이벤트연결규약 결정 3).
  "보여 줬다"는 기록이지 고객의 관심이 아니므로 인기 집계에 넣지 않는다 — 넣으면 많이 보여 준 상품이
  더 인기 상품이 되는 되먹임이 생긴다. 상세 페이지 조회(`request_id` 없는 view)는 센다.
- 시뮬레이터 고객(`users.source = "sim"`): 재학습 · A/B 실험용 가상 행동이다. 서비스 인기 통계에서는 뺀다
  (`settings.popularity_include_sim`으로 켤 수 있음). 시뮬레이터 구매는 현재 시각이라 H&M "최근 7일" 인기 창을
  통째로 차지하기 때문이다. 재학습 데이터 · 카운터에는 그대로 들어간다.
- 재학습 카운터: 학습 정답으로 쓰는 종류(TRAINING_TYPES = recsys.retrain.WEIGHTS)만 센다 — trainer는 new_log_count,
  CT 루프는 /metrics/log-delta?types=click,add_to_cart,purchase (#70). 노출 view만으로 "새 로그 1만 건"이 차서
  학습할 것 없는 재학습이 돌지 않게 한다.
"""
from sqlalchemy import or_
from sqlalchemy.sql import Select

from ..config import settings
from ..models import Event, User

TRAINING_TYPES = ("purchase", "add_to_cart", "click")


def not_impression():
    """노출 view(추천 목록 표시)가 아닌 이벤트."""
    return or_(Event.type != "view", Event.request_id.is_(None))


def service_users(q: Select) -> Select:
    """시뮬레이터 고객 이벤트를 뺀다 (settings.popularity_include_sim이면 그대로). q는 Event를 FROM에 둔 select."""
    if settings.popularity_include_sim:
        return q
    return q.join(User, User.id == Event.user_id).where(User.source != "sim")
