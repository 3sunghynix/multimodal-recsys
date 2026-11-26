"""D4 리뷰 키워드 배지 (Amazon): 사 본 사람만 아는 정보(핏 · 착용감 · 주의 · 분위기 · 상황)를 상품 카드의 짧은 키워드로.

실행 (trainer, 리뷰 본문 = Drive msrs_project/exports/amazon_reviews_catalog.parquet, 429만 건):
  docker compose --profile trainer run --rm -v C:/msrs_data/reviews:/reviews:ro trainer \
      python -m recsys.review_badges --reviews /reviews/amazon_reviews_catalog.parquet --out /data/badges/amazon_review_badges.parquet
  → 상품별 배지 (parent_asin, badge, kind, n_reviews, n_mention, share, lift) + 같은 폴더 samples.json (배지별 근거 문장 예시)
적재: docker compose exec api python -m scripts.load_badges --file /data/badges/amazon_review_badges.parquet

규칙 (11월 Colab 집계에서 정한 것, docs/project/리뷰배지.md)
  1. 리뷰 문장을 배지별 정규식으로 센다 (상품 사진 · 설명으로 알 수 있는 정보 — 크롭 · 소재명 등 — 는 넣지 않는다).
     방향을 나눈다: 작게 / 크게 / 정사이즈, 따뜻해요 / 시원해요. 부정 표현("don't size up", "didn't shrink")은 뺀다.
  2. 상품의 언급 비율(언급 리뷰 / 리뷰)이 **같은 카테고리 평균의 1.5배 이상**이고 언급 리뷰 3건 · 비율 5% 이상이면 배지 후보
     (흔한 배지 — "착용감 좋음" — 가 모든 상품에 붙지 않게). 카테고리는 3단계, 상품 30개 미만이면 2단계로.
  3. 맞서는 배지는 많은 쪽 하나만 (작게 · 크게 · 정사이즈 중 1개, 따뜻해요 · 시원해요 중 1개).
  4. 핏 · 신축성 · 보온 · 비침 · 세탁 배지는 액세서리 · 가방 · 지갑에는 붙이지 않는다.
  5. 상품당 최대 3개: 주의 배지 먼저, 그다음 lift(카테고리 대비 배수) 순.
"""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# (배지, 종류, 긍정 패턴, 제외 패턴). 종류: fit · good · caution · vibe · occasion
BADGES = [
    ("정사이즈", "fit", r"true to size|\btts\b|fits? (perfectly|as expected)|perfect fit", None),
    ("작게 나와요", "caution",
     r"runs? (very |really |a (little|bit|tad) )?small|too (small|tight)|size up|sized up|order(ed)? (a |one )?size (up|bigger|larger)|smaller than (expected|usual)",
     r"(don't|do not|didn't|no need to|not) (need to )?size up|not too (small|tight|snug)|(often|usually|always) too small|writing .{0,15}too small"),
    ("크게 나와요", "caution",
     r"runs? (very |really |a (little|bit|tad) )?(large|big)|too (big|large|loose|baggy)|size down|sized down|bigger than (expected|usual)|larger than (expected|usual)",
     r"(don't|do not|didn't|no need to|not) (need to )?size down|not too (big|large|loose)|(might|would) be too (big|large)"),
    ("착용감 좋음", "good", r"\b(comfortable|comfy)\b|so soft|super soft|very soft", r"not (very |that |so )?(comfortable|comfy)"),
    ("품질 좋음", "good", r"(good|great|high|nice|excellent) quality|well made",
     r"not (super |very |that |so |a )?(good|great|high|nice) quality|cheap quality|n't (good|great|high) quality"),
    ("신축성 좋음", "good", r"\bstretchy\b|good stretch|nice stretch|stretches (well|nicely)", r"(not|n't) (very |at all |that )?stretchy|no stretch"),
    ("따뜻해요", "good", r"\bwarm\b|keeps? me warm|\bcozy\b", r"warm weather|not (very )?warm|too warm|wanted .{0,20}warm"),
    ("시원해요", "good", r"breathable|keeps? (me|you) cool|cool (in|for) (the )?(summer|heat)|cool and comfortable",
     r"(not|n't) (very |at all )?breathable|not breathe|doesn't breathe"),
    ("가성비 좋음", "good",
     r"great (price|value|deal)|good value|worth (it|the (money|price))|(great|good|nice|perfect|excellent|awesome|amazing) for the price|can'?t beat the price|bang for",
     r"not worth|(ok|okay|fair|fine|alright) for the price"),
    ("재구매 많음", "good",
     r"(bought|ordered|purchased|buying|getting|order|buy|get) (another|a second)( one| pair)?\b|(bought|ordered|buying|order|buy) more\b|(buy|order|get|purchase)\w* (it |them |these |this |one )?in (every|several|other|more|multiple|different) colou?rs|every colou?r",
     r"won't be (buying|ordering)|not (be )?(buying|ordering)|wouldn't (buy|order)|never (buy|order)"),
    ("칭찬 많이 받음", "good", r"\bcompliments?\b", r"no compliments"),
    ("비침 주의", "caution", r"see[- ]?through|\bsheer\b|transparent",
     r"(isn't|not|wasn't|aren't|never|nothing) (quite |at all |too |very |even |remotely )?(see[- ]?through|sheer|transparent)|sheer aesthetic|sheer (lace|sleeves?|overlay|panel)|paper thin/transparent"),
    ("줄어듦·물빠짐 주의", "caution",
     r"\bshr[au]nk\b|\bshrank\b|\bshrinks\b|did shrink|will shrink|\bfaded\b|fades (after|quickly|fast)|colou?r (bled|ran|bleeds)|\bbled\b",
     r"(didn't|did not|doesn't|does not|no|won't|not|never|hasn't|haven't) (shrink|shrunk|fade|faded|bleed|bled)|hoping .{0,25}shrink|to shrink|if (they|it) (will |would )?shrink|avoid (any )?shrink|pre-?shrunk|wait and see"),
    ("쉽게 해져요", "caution",
     r"fell apart|falling apart|came apart|started (to )?pill|\bpilled\b|pills? (after|easily|quickly|badly)|pilling (after|already|badly|everywhere|within)|(ripped|tore) (after|the first|when|at the|in the|within)|holes? (after|already|within)|hole in (it|them|the (seam|toe|heel|knee|crotch|butt))|unravel",
     r"no pilling|without pilling|avoid (any )?pilling|ripped off|ripped jeans|(not|without) (fall|falling|coming) apart|old .{0,25}(falling|fell) apart"),
    ("귀여운", "vibe", r"\b(cute|adorable)\b", None),
    ("체형 보정", "vibe", r"\bflattering\b|slimming|hides? (my )?(tummy|belly|stomach|flaws)|\bforgiving\b",
     r"(not|n't) (find it |think it was |think it's |really |find them )?(very |as |that |so |too )?flattering|unflattering"),
    ("섹시한", "vibe", r"\bsexy\b", r"than (the )?sexy|not (like |very |that )?(cute )?sexy"),
    ("트렌디", "vibe", r"\b(trendy|stylish|fashionable)\b", r"not (very |that )?(trendy|stylish|fashionable)"),
    ("우아한", "vibe", r"\b(elegant|classy|sophisticated)\b", r"not (as |very |that )?(classy|elegant)|more elegant|would have been .{0,20}elegant"),
    ("출근룩", "occasion", r"\bfor work\b|\bat work\b|wear(ing)? (it |them |this |these )?to work|\boffice\b|business casual|work (outfit|wear|pants|shoes|clothes|attire)",
     r"(wouldn't|won't|can't|couldn't) wear (it |them |this |these )?to work|travel .{0,10}for work|ready for work"),
    ("하객·행사룩", "occasion", r"\bwedding\b|formal (event|occasion|wear|dinner|affair)|\bprom\b|cocktail|date night|\bgala\b|special occasion",
     r"wedding (ring|band)|wedding party gift"),
    ("운동용", "occasion", r"\bgym\b|workouts?\b|work(ing)? out\b|for running|while running|running shoes?|\bjogging\b|\byoga\b|\bhiking\b|exercis", None),
    ("여행용", "occasion", r"\btravel|vacation|\bcruise\b", r"travel size"),
    ("홈웨어", "occasion", r"pajamas?|pyjamas?|sleepwear|lounging|lounge ?wear|(wear|wearing) (it |them |these )?around the house|sleep in", None),
    ("선물 추천", "occasion", r"\bgift\b|as a present|for christmas|birthday present", r"gift ?cards?|gift box|gift wrap|gift[- ]ready"),
]
OPPOSE = [("정사이즈", "작게 나와요", "크게 나와요"), ("따뜻해요", "시원해요")]
NO_BODY = {"정사이즈", "작게 나와요", "크게 나와요", "신축성 좋음", "따뜻해요", "시원해요", "비침 주의", "줄어듦·물빠짐 주의"}
NO_BODY_CATS = ("액세서리", "가방·지갑")
MIN_MENTION, MIN_SHARE, MIN_LIFT, MAX_BADGES, MIN_CAT_ITEMS = 3, 0.05, 1.5, 3, 30
SAMPLES = 20


warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")   # 그룹 있는 패턴 경고 (결과는 같다)


def count_mentions(reviews: Path, batch_rows: int = 500_000, log=print):
    """(상품별 리뷰 수 Series, 상품 × 배지 언급 리뷰 수 DataFrame, 배지별 근거 문장 예시)."""
    import pyarrow.parquet as pq
    pats = [(b, re.compile(p, re.I), re.compile(x, re.I) if x else None) for b, _, p, x in BADGES]
    n_parts, c_parts = [], []
    samples = {b: [] for b, _, _, _ in BADGES}
    f = pq.ParquetFile(reviews)
    done = 0
    for batch in f.iter_batches(batch_size=batch_rows, columns=["parent_asin", "title", "text"]):
        df = batch.to_pandas()
        text = (df["title"].fillna("") + " " + df["text"].fillna("")).str.replace("<br />", " ", regex=False) \
            .str.replace("\u2019", "'", regex=False)                      # 둥근 따옴표 → ' (부정 표현 "didn’t shrink" 제외용)
        hit = {}
        for b, p, x in pats:
            m = text.str.contains(p, regex=True)
            if x is not None:
                m &= ~text.str.contains(x, regex=True)
            hit[b] = m.values
            if m.any():                                           # 정확도 점검용: 배치마다 무작위 3건 → 끝에서 SAMPLES건
                for t in text[m].sample(min(3, int(m.sum())), random_state=len(samples[b])).tolist():
                    mm = p.search(t)
                    s = max(0, mm.start() - 80)
                    samples[b].append(t[s:mm.end() + 80])
        h = pd.DataFrame(hit)
        h["parent_asin"] = df["parent_asin"].values
        c_parts.append(h.groupby("parent_asin").sum())
        n_parts.append(df.groupby("parent_asin").size())
        done += len(df)
        log(f"리뷰 {done:,} / {f.metadata.num_rows:,}")
    rng = np.random.default_rng(0)
    samples = {b: [v[i] for i in sorted(rng.choice(len(v), min(SAMPLES, len(v)), replace=False))] if v else []
               for b, v in samples.items()}
    n = pd.concat(n_parts).groupby(level=0).sum()
    c = pd.concat(c_parts).groupby(level=0).sum().reindex(n.index).fillna(0).astype(int)
    return n, c, samples


def assign(n: pd.Series, c: pd.DataFrame, cats: pd.Series) -> pd.DataFrame:
    """규칙 2 ~ 5 → 배지 표 (parent_asin, badge, kind, n_reviews, n_mention, share, lift, rank)."""
    kind = {b: k for b, k, _, _ in BADGES}
    cats = cats.reindex(n.index).fillna("")
    lvl3 = cats
    lvl2 = cats.str.split(" > ").str[:2].str.join(" > ")
    size3 = lvl3.map(lvl3.value_counts())
    group = np.where(size3 >= MIN_CAT_ITEMS, lvl3, lvl2)
    share = c.div(n, axis=0)
    base = share.groupby(group).transform("mean")
    lift = share / base.replace(0, np.nan)
    ok = (c >= MIN_MENTION) & (share >= MIN_SHARE) & (lift >= MIN_LIFT)
    body_less = cats.str.split(" > ").str[1].isin(NO_BODY_CATS)
    for b in NO_BODY:
        ok.loc[body_less, b] = False
    for grp in OPPOSE:                                       # 맞서는 배지: 언급이 가장 많은 쪽만
        g = list(grp)
        top = c[g].idxmax(axis=1)
        for b in g:
            ok[b] &= top == b
    rows = ok.stack()
    rows = rows[rows].index.to_frame(index=False)
    rows.columns = ["parent_asin", "badge"]
    rows["kind"] = rows.badge.map(kind)
    rows["n_reviews"] = n.reindex(rows.parent_asin).values
    rows["n_mention"] = [int(c.at[a, b]) for a, b in zip(rows.parent_asin, rows.badge)]
    rows["share"] = (rows.n_mention / rows.n_reviews).round(4)
    rows["lift"] = [round(float(lift.at[a, b]), 3) for a, b in zip(rows.parent_asin, rows.badge)]
    rows["caution"] = rows.kind == "caution"
    rows = rows.sort_values(["parent_asin", "caution", "lift"], ascending=[True, False, False])
    rows["rank"] = rows.groupby("parent_asin").cumcount()
    return rows[rows["rank"] < MAX_BADGES].drop(columns="caution").reset_index(drop=True)


def catalog_categories() -> pd.Series:
    """DB의 Amazon 상품 원본 id(parent_asin) → 3단계 카테고리."""
    sys.path.insert(0, "/app")
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import Item
    with SessionLocal() as db:
        rows = db.execute(select(Item.source_item_id, Item.category).where(Item.source == "amazon")).all()
    return pd.Series({a: cat or "" for a, cat in rows})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviews", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/data/badges/amazon_review_badges.parquet"))
    a = ap.parse_args()
    n, c, samples = count_mentions(a.reviews)
    cats = catalog_categories()
    n, c = n[n.index.isin(cats.index)], c[c.index.isin(cats.index)]
    badges = assign(n, c, cats)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    badges.to_parquet(a.out, index=False)
    (a.out.parent / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
    per = badges.groupby("parent_asin").size().reindex(cats.index).fillna(0)
    summary = {"items": int(len(cats)), "items_with_reviews": int(len(n)), "reviews": int(n.sum()),
               "items_with_badge": round(float((per > 0).mean()), 4), "mean_badges_when_any": round(float(per[per > 0].mean()), 2),
               "badge_items": badges.badge.value_counts().to_dict(),
               "mention_rate": {b: round(float(c[b].sum() / n.sum()), 4) for b in c.columns}}
    (a.out.parent / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
