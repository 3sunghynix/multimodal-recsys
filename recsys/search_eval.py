"""텍스트 검색 방식 비교 (Colab). 정답 라벨이 없으므로 질의별 키워드 규칙으로 '맞는 상품'을 판정한다.

비교 대상
  인코더  text  : MiniLM 질의 → 상품 텍스트 임베딩 (같은 모델)
          image : Marqo-FashionSigLIP 텍스트 질의 → 상품 이미지 임베딩 (다른 모달리티)
          mix   : 두 점수를 표준화 후 가중 합 (화면의 텍스트 · 이미지 우선순위)
  결합    global (원래 점수) vs per_source_z (데이터셋별 표준화)
  언어    영어 질의 vs 한국어 질의 → 번역(opus-mt-ko-en) → 같은 경로
지표 (상위 20)
  precision : 상품 제목 · 설명 · 카테고리가 질의 키워드 규칙을 만족하는 비율 (예: "black dress" → dress 그리고 black)
  hm_share  : 결과 중 H&M 비율 (카탈로그 비율 약 19%. 한쪽 독차지 여부)
  both      : 결과에 두 데이터셋이 모두 나온 질의 비율
한계: 키워드 판정이라 제목에 단어가 그대로 있는 상품에 유리하다(MiniLM 쪽). 이미지만 맞고 제목이 다른 상품은 틀림으로 센다.

실행 (저장소 루트, Colab):
    python -m recsys.search_eval --root /content/drive/MyDrive/msrs_project --out results/search
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recsys import search as S

K = 20
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MARQO = "hf-hub:Marqo/marqo-fashionSigLIP"
MT_MODEL = "Helsinki-NLP/opus-mt-ko-en"

# (영어, 한국어, 규칙: 그룹마다 하나 이상 일치해야 하고 모든 그룹을 만족해야 정답)
QUERIES = [
    ("black dress", "검은색 원피스", [r"dress", r"black"]),
    ("summer floral dress", "여름 꽃무늬 원피스", [r"dress", r"floral|flower"]),
    ("white t-shirt", "흰색 티셔츠", [r"t-?shirt|\btee", r"white"]),
    ("denim jacket", "청자켓", [r"jacket", r"denim|jean"]),
    ("high waist jeans", "하이웨이스트 청바지", [r"jean|denim"]),
    ("men's running shoes", "남성 러닝화", [r"run|sneaker|trainer"]),
    ("ankle boots", "앵클 부츠", [r"boot"]),
    ("sandals", "샌들", [r"sandal"]),
    ("wool sweater", "울 스웨터", [r"sweater|jumper|pullover|knit"]),
    ("cardigan", "가디건", [r"cardigan"]),
    ("hoodie", "후드티", [r"hood"]),
    ("winter coat", "겨울 코트", [r"coat|parka|puffer"]),
    ("blazer", "블레이저", [r"blazer"]),
    ("linen shirt", "린넨 셔츠", [r"shirt|blouse", r"linen"]),
    ("skirt", "치마", [r"skirt"]),
    ("shorts", "반바지", [r"short"]),
    ("yoga leggings", "요가 레깅스", [r"legging|tights"]),
    ("sports bra", "스포츠 브라", [r"bra\b|bralette"]),
    ("swimsuit", "수영복", [r"swim|bikini"]),
    ("pajamas", "잠옷", [r"paja|pyja|sleep|night"]),
    ("socks", "양말", [r"sock"]),
    ("leather belt", "가죽 벨트", [r"belt", r"leather"]),
    ("baseball cap", "야구 모자", [r"cap\b|hat\b|baseball"]),
    ("silk scarf", "실크 스카프", [r"scarf|scarves"]),
    ("handbag", "핸드백", [r"bag|purse|tote"]),
    ("kids t-shirt", "아동 티셔츠", [r"t-?shirt|\btee|top", r"kid|child|boy|girl|baby|toddler"]),
    ("striped shirt", "스트라이프 셔츠", [r"shirt|top|blouse", r"stripe"]),
    ("red dress", "빨간 원피스", [r"dress", r"\bred\b"]),
]
DOC_COLS = ["title", "cat_target", "cat_type", "categories", "product_type_name", "description", "features"]


def load_catalog(root: Path):
    emb = root / "embeddings" / "v2"
    cat = pd.read_parquet(emb / "catalog.parquet").sort_values("row").reset_index(drop=True)
    items = {"text": np.load(emb / "text_pca256.npy").astype(np.float32),
             "image": np.load(emb / "image_pca256.npy").astype(np.float32)}
    pca = {m: dict(np.load(emb / f"pca_{m}.npz")) for m in ("text", "image")}
    docs = []
    for src, sub in (("amazon", "amazon_v2"), ("hm", "hm")):
        it = pd.read_parquet(root / "processed" / sub / "items.parquet")
        cols = [c for c in DOC_COLS if c in it.columns]
        doc = it[cols].astype(str).agg(" ".join, axis=1).str.lower()
        docs.append(pd.DataFrame({"source": src, "item_id": it.item_id.values, "doc": doc.values, "title": it.title.astype(str).values}))
    docs = cat.merge(pd.concat(docs), on=["source", "item_id"], how="left")
    assert len(docs) == len(cat) and docs.doc.notna().all(), "catalog와 items가 맞지 않습니다"
    return docs, items, pca


class Encoders:
    def __init__(self):
        import open_clip
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.minilm = SentenceTransformer(TEXT_MODEL, device=self.dev)
        self.marqo, _, _ = open_clip.create_model_and_transforms(MARQO)
        self.marqo = self.marqo.to(self.dev).eval()
        self.tok = open_clip.get_tokenizer(MARQO)
        self._mt = None

    def text(self, qs):
        return self.minilm.encode(qs, normalize_embeddings=True, convert_to_numpy=True)

    def marqo_text(self, qs):
        with self.torch.no_grad():
            return self.marqo.encode_text(self.tok(qs).to(self.dev)).float().cpu().numpy()

    def translate(self, qs):
        # transformers 5에서 translation 파이프라인이 없어져 모델을 직접 호출한다
        if self._mt is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._mt = (AutoTokenizer.from_pretrained(MT_MODEL),
                        AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL).to(self.dev).eval())
        tok, model = self._mt
        batch = tok(qs, return_tensors="pt", padding=True).to(self.dev)
        with self.torch.no_grad():
            out = model.generate(**batch, max_new_tokens=32, num_beams=4)
        return tok.batch_decode(out, skip_special_tokens=True)


def relevant(doc: str, rule) -> bool:
    return all(re.search(p, doc) for p in rule)


def evaluate(docs, idx, rules):
    src = docs.source.values
    rows = []
    for q, (row, rule) in enumerate(zip(idx, rules)):
        hit = [relevant(docs.doc.values[i], rule) for i in row]
        s = src[row]
        rows.append({"precision": float(np.mean(hit)), "hm_share": float((s == "hm").mean()),
                     "both": float(len(set(s)) == 2)})
    return pd.DataFrame(rows)


METHODS = {   # 이름: (가중치, image 질의 평균 빼기)
    "text": ({"text": 1.0}, True),
    "image": ({"image": 1.0}, False),
    "image_centered": ({"image": 1.0}, True),
    "mix_5_5": ({"text": 0.5, "image": 0.5}, False),
    "mix_7_3": ({"text": 0.7, "image": 0.3}, False),
}


def run(root: Path, out: Path | None):
    docs, items, pca = load_catalog(root)
    groups = docs.source.values
    enc = Encoders()
    en = [q[0] for q in QUERIES]
    ko = [q[1] for q in QUERIES]
    rules = [q[2] for q in QUERIES]
    t = time.time()
    ko_en = enc.translate(ko)
    langs = {"en": en, "ko": ko_en}
    print("번역:", *[f"{a} → {b}" for a, b in zip(ko, ko_en)], sep="\n  ")

    table, per_query, examples = [], [], {}
    for lang, qs in langs.items():
        raw = {"text": enc.text(qs), "image": enc.marqo_text(qs)}
        for name, (w, center_img) in METHODS.items():
            q = {"text": S.project(raw["text"], pca["text"]["mean"], pca["text"]["components"]),
                 "image": S.project(raw["image"], pca["image"]["mean"], pca["image"]["components"], center=center_img)}
            for merge in S.MERGES:
                idx, _ = S.search(q, items, w, groups, K, merge)
                ev = evaluate(docs, idx, rules)
                table.append({"lang": lang, "method": name, "merge": merge, **ev.mean().round(4).to_dict()})
                per_query.append(ev.assign(lang=lang, method=name, merge=merge, query=qs))
                if lang == "en" and merge == "per_source_z":
                    examples[name] = {qs[i]: docs.title.values[idx[i, :5]].tolist() for i in (0, 5, 10)}
    table = pd.DataFrame(table)
    print(f"\n상위 {K} 기준 ({time.time() - t:.0f}s)")
    print(table.to_markdown(index=False))
    if out:
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "search_compare.csv", index=False)
        pd.concat(per_query).to_csv(out / "search_per_query.csv", index=False)
        (out / "search_examples.json").write_text(json.dumps({"translations": dict(zip(ko, ko_en)), "top5": examples},
                                                             ensure_ascii=False, indent=2), encoding="utf-8")
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="드라이브 msrs_project 경로")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    run(a.root, a.out)


if __name__ == "__main__":
    main()
