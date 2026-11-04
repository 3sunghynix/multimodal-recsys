"""검색어 생성: 모든 한국어 검색어가 검색 서버 사전으로 번역되는지, 이미지 쿼리는 다른 데이터셋에서 고르는지."""
import numpy as np
import pandas as pd
import pytest

from recsys.translate import translate_ko
from simulator.queries import ITEM_WORDS, STYLE, image_query_item, text_query


def test_every_korean_query_is_fully_translated():
    rng = np.random.default_rng(0)
    seen = 0
    for leaf in ITEM_WORDS:
        for persona in STYLE:
            for _ in range(40):
                q = text_query(rng, persona, leaf, brand="Levi's")
                if q["lang"] == "ko":
                    en, untranslated = translate_ko(q["query"])
                    assert untranslated == [], (q, en)
                    seen += 1
    assert seen > 1000


def test_persona_style_and_language_mix():
    rng = np.random.default_rng(1)
    qs = [text_query(rng, "window_shopper", "원피스") for _ in range(300)]
    assert np.mean([len(q["query"].split()) for q in qs]) < 1.3            # 눈팅형: 짧은 일반 검색어
    assert 0.5 < np.mean([q["lang"] == "ko" for q in qs]) < 0.7             # 한국어 약 60%
    brand = [text_query(rng, "brand_loyal", "청바지", brand="Levi's")["query"] for _ in range(300)]
    assert np.mean(["Levi's" in q for q in brand]) > 0.7                    # 브랜드충성: 브랜드명 검색
    assert all(text_query(rng, "trendsetter", "헤어 액세서리")["lang"] == "en" for _ in range(20))   # 한국어 단어 없는 품목
    with pytest.raises(ValueError):
        text_query(rng, "trendsetter", "없는 품목")


def test_image_query_prefers_other_dataset():
    items = pd.DataFrame({"source": ["hm", "hm", "amazon", "amazon"], "item_id": [1, 2, 3, 4],
                          "category_l3": ["원피스", "원피스", "원피스", "티셔츠"]})
    rng = np.random.default_rng(2)
    picks = {image_query_item(rng, items, "원피스", "hm") for _ in range(20)}
    assert picks == {("amazon", 3)}                                         # H&M 고객 → Amazon 원피스 사진
    assert image_query_item(rng, items, "티셔츠", "amazon") == ("amazon", 4)  # 다른 데이터셋에 없으면 같은 데이터셋
    assert image_query_item(rng, items, "수영복", "hm") is None


def test_leaves_match_taxonomy_when_available():
    taxonomy = pytest.importorskip("recsys.taxonomy")                       # #36 머지 후 확인
    leaves = {leaf for _, _, leaf in taxonomy.RULES}
    assert leaves == set(ITEM_WORDS)


def test_modifiers_fit_the_item():
    from simulator.queries import MATERIALS, TRENDY
    rng = np.random.default_rng(3)
    for leaf in ("가방", "원피스", "시계"):
        for _ in range(300):
            words = text_query(rng, "trendsetter", leaf)["query"].split()
            for ko, en, ok in MATERIALS + TRENDY:
                if ko in words or en in " ".join(words):
                    assert leaf in ok, (leaf, words)
