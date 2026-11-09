"""시뮬레이터 검색어 생성 (텍스트 · 이미지). 페르소나와 선호 소분류(3단계 분류표, docs/project/카테고리분류.md)로 만든다.

텍스트 검색어
- 품목 단어(소분류) + 수식어(색 · 소재 · 핏 · 브랜드)를 페르소나 성격에 맞게 조합한다.
- 한국어 · 영어 섞기 (기본 한국어 60%). 한국어 단어는 모두 검색 서버의 한국어 사전(recsys/translate.py)에 있는 것만 쓴다
  → 시뮬레이터 검색이 번역 실패로 엉뚱한 결과를 내지 않는다 (테스트로 확인).
- 사전에 한국어가 없는 품목(헤어 액세서리 · 케이스 등)은 영어로만 검색한다.
이미지 검색
- 고객과 **다른 데이터셋**의 같은 소분류 상품 사진을 쿼리로 쓴다 ("본 옷과 비슷한 것" 상황). 쿼리로 쓴 상품은 호출자가 결과에서 뺀다.
- 사진이 실제로 있는 상품만 고른다 (업로드 실패 방지). 사진 목록은 image_keys(데이터 폴더) → mark_has_image.

페르소나 id는 simulator/personas.yaml(v0.3)과 같다.
"""
import numpy as np

# 소분류 → (한국어 품목 단어, 영어 품목 단어)
ITEM_WORDS = {
    "티셔츠": (["티셔츠", "반팔티"], ["t-shirt", "tee"]),
    "셔츠·블라우스": (["셔츠", "블라우스"], ["shirt", "blouse"]),
    "니트·스웨터": (["니트", "스웨터"], ["sweater", "knit sweater"]),
    "후드·맨투맨": (["후드티", "맨투맨"], ["hoodie", "sweatshirt"]),
    "민소매·탱크톱": (["민소매", "탱크탑", "나시"], ["tank top", "camisole"]),
    "카디건": (["가디건"], ["cardigan"]),
    "폴로": (["폴로"], ["polo shirt"]),
    "기타 상의": (["탑", "크롭티"], ["top", "crop top"]),
    "바지": (["바지", "슬랙스", "조거팬츠"], ["pants", "trousers", "joggers"]),
    "청바지": (["청바지"], ["jeans"]),
    "반바지": (["반바지"], ["shorts"]),
    "스커트": (["치마", "스커트"], ["skirt"]),
    "레깅스": (["레깅스"], ["leggings"]),
    "원피스": (["원피스", "드레스"], ["dress", "maxi dress"]),
    "점프수트": (["점프수트", "멜빵"], ["jumpsuit", "overalls"]),
    "세트": (["트레이닝복"], ["tracksuit", "matching set"]),
    "재킷": (["자켓", "재킷", "바람막이"], ["jacket"]),
    "코트·패딩": (["코트", "패딩", "트렌치코트"], ["coat", "puffer jacket"]),
    "블레이저·정장": (["블레이저", "정장"], ["blazer", "suit"]),
    "조끼": (["조끼", "베스트"], ["vest"]),
    "속옷": (["속옷", "브라"], ["underwear", "bra"]),
    "잠옷·라운지웨어": (["잠옷", "파자마"], ["pajamas", "loungewear"]),
    "양말·스타킹": (["양말", "스타킹"], ["socks", "tights"]),
    "수영복": (["수영복", "비키니"], ["swimsuit", "bikini"]),
    "운동화": (["운동화", "스니커즈", "러닝화"], ["sneakers", "running shoes"]),
    "부츠": (["부츠", "앵클 부츠"], ["boots", "ankle boots"]),
    "샌들": (["샌들"], ["sandals"]),
    "플랫·로퍼": (["로퍼", "플랫슈즈"], ["loafers", "flats"]),
    "슬리퍼": (["슬리퍼"], ["slippers"]),
    "구두·힐": (["구두", "힐"], ["heels", "pumps"]),
    "기타 신발": (["신발"], ["shoes"]),
    "가방": (["가방", "백팩", "토트백", "크로스백"], ["bag", "backpack", "tote bag"]),
    "지갑": (["지갑"], ["wallet"]),
    "귀걸이": (["귀걸이"], ["earrings"]),
    "목걸이": (["목걸이"], ["necklace"]),
    "팔찌": (["팔찌"], ["bracelet"]),
    "반지": (["반지"], ["ring"]),
    "시계": (["시계"], ["watch"]),
    "모자": (["모자", "야구모자", "비니", "버킷햇"], ["hat", "baseball cap", "beanie"]),
    "선글라스": (["선글라스"], ["sunglasses"]),
    "벨트": (["벨트"], ["belt"]),
    "스카프": (["스카프", "머플러"], ["scarf"]),
    "장갑": (["장갑"], ["gloves"]),
    "헤어 액세서리": ([], ["hair clip", "headband", "scrunchie"]),
    "기타 주얼리": ([], ["jewelry", "brooch"]),
    "기타 액세서리": ([], ["accessories", "keychain"]),
    "케이스·소품": ([], ["glasses case"]),
}

COLORS = (["검은색", "흰색", "베이지", "네이비", "회색", "빨간", "파란", "분홍"],
          ["black", "white", "beige", "navy", "gray", "red", "blue", "pink"])
# 수식어 (한국어, 영어, 어울리는 소분류) — 어울리는 품목에만 붙인다 ("플리츠 백팩", "와이드 드레스" 방지)
TOPS = {"티셔츠", "셔츠·블라우스", "니트·스웨터", "후드·맨투맨", "민소매·탱크톱", "카디건", "기타 상의"}
OUTER = {"재킷", "코트·패딩", "블레이저·정장", "조끼"}
MATERIALS = [
    ("린넨", "linen", {"셔츠·블라우스", "바지", "원피스", "반바지", "블레이저·정장"}),
    ("데님", "denim", {"재킷", "스커트", "셔츠·블라우스", "반바지", "원피스", "점프수트"}),
    ("가죽", "leather", {"재킷", "가방", "지갑", "벨트", "부츠", "플랫·로퍼", "스커트", "장갑"}),
    ("울", "wool", {"니트·스웨터", "코트·패딩", "카디건", "모자", "스카프", "양말·스타킹", "장갑"}),
    ("실크", "silk", {"셔츠·블라우스", "원피스", "스카프", "잠옷·라운지웨어", "스커트"}),
    ("코튼", "cotton", {"티셔츠", "양말·스타킹", "속옷", "잠옷·라운지웨어", "후드·맨투맨", "반바지"}),
]
TRENDY = [
    ("오버사이즈", "oversized", TOPS | OUTER),
    ("크롭", "cropped", {"티셔츠", "니트·스웨터", "카디건", "후드·맨투맨", "재킷", "기타 상의", "민소매·탱크톱"}),
    ("와이드", "wide leg", {"바지", "청바지"}),
    ("플리츠", "pleated", {"스커트", "원피스", "바지"}),
    ("빈티지", "vintage", TOPS | OUTER | {"원피스", "청바지", "가방", "부츠", "선글라스"}),
]
NO_COLOR = {"시계", "케이스·소품", "기타 액세서리"}

# 페르소나별 수식어 확률 (색, 소재, 유행 수식어, 브랜드명)
STYLE = {
    "trendsetter":      {"color": 0.3, "material": 0.1, "trendy": 0.6, "brand": 0.0},
    "practical":        {"color": 0.6, "material": 0.2, "trendy": 0.0, "brand": 0.1},
    "value_seeker":     {"color": 0.3, "material": 0.1, "trendy": 0.0, "brand": 0.0},
    "brand_loyal":      {"color": 0.2, "material": 0.0, "trendy": 0.0, "brand": 0.8},
    "impulse_buyer":    {"color": 0.3, "material": 0.0, "trendy": 0.4, "brand": 0.0},
    "careful_explorer": {"color": 0.5, "material": 0.4, "trendy": 0.1, "brand": 0.1},
    "window_shopper":   {"color": 0.1, "material": 0.0, "trendy": 0.0, "brand": 0.0},   # 짧은 일반 검색어
}


def text_query(rng: np.random.Generator, persona: str, leaf: str, brand: str | None = None, ko_share: float = 0.6) -> dict:
    """{"query", "lang", "leaf"}. leaf: 선호 소분류, brand: 고객 선호 브랜드명(브랜드충성 등)."""
    if leaf not in ITEM_WORDS:
        raise ValueError(f"모르는 소분류: {leaf}")
    if persona not in STYLE:
        raise ValueError(f"모르는 페르소나: {persona}")
    ko_words, en_words = ITEM_WORDS[leaf]
    lang = "ko" if ko_words and rng.random() < ko_share else "en"
    k = 0 if lang == "ko" else 1
    words = [str(rng.choice((ko_words, en_words)[k]))]
    style = STYLE[persona]
    fits = lambda table: [w for w in table if leaf in w[2]]               # noqa: E731
    if rng.random() < style["material"] and fits(MATERIALS):
        words.insert(0, fits(MATERIALS)[int(rng.integers(len(fits(MATERIALS))))][k])
    if rng.random() < style["color"] and leaf not in NO_COLOR:
        words.insert(0, str(rng.choice(COLORS[k])))
    if rng.random() < style["trendy"] and fits(TRENDY):
        words.insert(0, fits(TRENDY)[int(rng.integers(len(fits(TRENDY))))][k])
    if brand and rng.random() < style["brand"]:
        words.insert(0, brand)                                  # 브랜드명은 원문 그대로 (사전 번역 대상 아님)
    return {"query": " ".join(words), "lang": lang, "leaf": leaf}


def image_keys(data_dir) -> set[tuple[str, int]]:
    """데이터 폴더에서 사진이 있는 상품 (source, item_id). 압축 해제된 images/{source}/{item_id}.jpg를 먼저 보고, 없으면 images.zip 목록."""
    from pathlib import Path
    from zipfile import ZipFile

    root = Path(data_dir)
    if (root / "images").is_dir():
        names = [f"{p.parent.name}/{p.name}" for p in (root / "images").glob("*/*.jpg")]
    elif (root / "images.zip").exists():
        with ZipFile(root / "images.zip") as z:
            names = z.namelist()
    else:
        return set()
    keys = set()
    for n in names:
        parts = n.replace("\\", "/").split("/")
        if len(parts) >= 2 and parts[-1].endswith(".jpg") and parts[-1][:-4].isdigit():
            keys.add((parts[-2], int(parts[-1][:-4])))
    return keys


def mark_has_image(items, keys: set[tuple[str, int]]):
    """items에 has_image 열을 붙인다 (image_query_item이 이 열로 후보를 거른다)."""
    items = items.copy()
    items["has_image"] = [(str(s), int(i)) in keys for s, i in zip(items.source, items.item_id)]
    return items


def image_query_item(rng: np.random.Generator, items, leaf: str, customer_source: str | None = None):
    """같은 소분류에서 **사진이 있는** 상품 하나. 고객과 다른 데이터셋을 우선하고, 거기 없으면 같은 데이터셋, 둘 다 없으면 None.
    items: source · item_id · category_l3 · has_image 열 (has_image는 mark_has_image(items, image_keys(데이터 폴더))로 붙인다).
    반환: (source, item_id) 또는 None. 호출자는 이 상품을 검색 결과에서 빼고, 이 상품 사진을 업로드한다."""
    if "has_image" not in items.columns:
        raise ValueError("items에 has_image 열이 필요합니다: mark_has_image(items, image_keys(데이터 폴더))")
    pool = items[(items.category_l3 == leaf) & items.has_image.astype(bool)]
    if customer_source is not None and (pool.source != customer_source).any():
        pool = pool[pool.source != customer_source]
    if pool.empty:
        return None
    row = pool.iloc[int(rng.integers(len(pool)))]
    return str(row.source), int(row.item_id)
