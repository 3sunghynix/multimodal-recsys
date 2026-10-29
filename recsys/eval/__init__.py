"""오프라인 평가. 지표 정의는 docs/project/지표정의.md 와 같다 (관리자 화면 숫자와 이 코드의 숫자가 같아야 한다)."""
from .metrics import accuracy_at_k, brand_alignment_at_k, brand_diversity_at_k, coverage_at_k, intra_list_diversity_at_k, \
    popularity_percentile_at_k, user_brand_loyalty  # noqa: F401
from .protocol import EvalSet, build_eval_set  # noqa: F401

# Online concentration over caller-selected event windows; distinct from offline dataset summaries.
from .online import dominant_brand_share  # noqa: F401
