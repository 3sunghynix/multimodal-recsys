"""3단계 공통 카테고리 분류표: 대분류(대상) > 중분류(품목군) > 소분류(품목).

Amazon 카테고리 경로("Clothing, Shoes & Jewelry > Women > Shoes > Sandals > Heeled Sandals")와
H&M 상품 분류(cat_target "Women" + categories "Garment Upper body > T-shirt")를 하나의 한국어 분류표로 맞춘다.

규칙
- 대분류: 대상 단어(Women · Men · Girls · Boys · Baby · Kids …)를 경로에서 찾는다. 없으면 '공용'.
- 중 · 소분류: 경로를 **깊은 단계부터** 보면서 처음 맞는 규칙을 쓴다 (깊을수록 구체적).
  규칙 순서가 우선순위다 — 예: 'vest top'(민소매)을 'vest'(조끼)보다, 't-shirt'를 'shirt'보다 먼저.
- 일반적인 규칙(세트 · 플랫 · 기타 신발 · 기타 상의 · 기타 액세서리)에 걸리면 윗단계에서 구체적인 규칙을 다시 찾는다.
  예: 'Sleep & Lounge > Sets' → 잠옷, 'Sandals > Flats' → 샌들, 'Shorts > Flat Front' → 반바지.
- 경로에 품목 정보가 없으면(예: 'Women > Clothing'에서 끝남) 상품명으로 한 번 더 찾는다.
  상품명에서는 먼저 수식어(소매 길이 · 운동 종류 · 핏 · 무늬)를 지운다 — 'Running Shorts'가 운동화, 'Bootcut Jeans'가 부츠가 되지 않게.
- 그래도 안 맞으면 상위 종류(cat_type)로 중분류만 정하고 소분류는 '기타'.
"""
import re

TARGETS = [  # (정규식, 대분류) — 경로 앞쪽부터 찾는다
    (r"\b(women|womens|ladies|ladieswear|divided)\b", "여성"),
    (r"\b(men|mens|menswear)\b", "남성"),
    (r"\b(girls|boys|baby|babies|kids|toddler|infant|children)\b", "아동"),
]

# (정규식, 중분류, 소분류) — 위에서부터 우선
RULES = [
    # 수영복 (bra · top · bottom보다 먼저)
    (r"swim|bikini|cover-?ups?\b|rash guard|board ?shorts", "수영복", "수영복"),
    # 이너웨어
    (r"pajama|pyjama|sleep|lounge|nightwear|night ?gown|\brobes?\b", "이너웨어", "잠옷·라운지웨어"),
    (r"base layer|thermal|long johns", "이너웨어", "속옷"),
    (r"sock|hosiery|underwear tights", "이너웨어", "양말·스타킹"),
    (r"\bbras?\b|bralette|lingerie|underwear|brief|boxer|panties|panty|shapewear|undershirt", "이너웨어", "속옷"),
    # 신발
    (r"sneaker|athletic|running|trainer|walking|training|tennis|basketball|golf shoe|cleat", "신발", "운동화"),
    (r"sandal|flip-?flop|slide", "신발", "샌들"),
    (r"\bboots?\b|bootie", "신발", "부츠"),
    (r"slipper", "신발", "슬리퍼"),
    (r"pump|heel|dress shoe", "신발", "구두·힐"),
    (r"loafer|flat|oxford|slip-?on|mule|clog|moccasin|ballet|espadrille", "신발", "플랫·로퍼"),
    (r"\bshoes?\b|footwear", "신발", "기타 신발"),
    # 가방 · 지갑
    (r"wallet|card case|money organizer|coin purse", "가방·지갑", "지갑"),
    (r"handbag|\bbags?\b|backpack|tote|clutch|crossbody|shoulder bag|satchel|hobo|purse", "가방·지갑", "가방"),
    # 액세서리 · 주얼리
    (r"earring", "액세서리", "귀걸이"),
    (r"necklace|pendant|choker|chain", "액세서리", "목걸이"),
    (r"bracelet|bangle|anklet", "액세서리", "팔찌"),
    (r"\brings?\b", "액세서리", "반지"),
    (r"watch", "액세서리", "시계"),
    (r"\bhats?\b|\bcaps?\b|beanie|beret|visor|fedora", "액세서리", "모자"),
    (r"sunglass|eyewear|eyeglass", "액세서리", "선글라스"),
    (r"\bbelts?\b|suspender", "액세서리", "벨트"),
    (r"scarf|scarves|\bwraps?\b|shawl|pashmina", "액세서리", "스카프"),
    (r"\bhair\b|alice band|hair clip|hair string|headband|scrunchie", "액세서리", "헤어 액세서리"),
    (r"glove|mitten", "액세서리", "장갑"),
    (r"jewelry|jewellery|brooch|charm|body jewelry", "액세서리", "기타 주얼리"),
    (r"\bneck ?ties?\b|\bbow ?ties?\b|\bties\b|cufflink|keychain|umbrella|accessor", "액세서리", "기타 액세서리"),
    # 원피스 · 세트 (dress shirt · dress shoe는 위에서 처리)
    (r"dress shirt", "상의", "셔츠·블라우스"),
    (r"jumpsuit|romper|overall|playsuit|dungaree", "원피스·세트", "점프수트"),
    (r"\bdress(es)?\b", "원피스·세트", "원피스"),
    (r"garment set|\bsets?\b|outfit|costume", "원피스·세트", "세트"),
    # 아우터
    (r"blazer|suit jacket|\bsuits?\b|sport coat", "아우터", "블레이저·정장"),
    (r"coat|parka|trench|puffer|down jacket|anorak|peacoat", "아우터", "코트·패딩"),
    (r"jacket|windbreaker|bomber", "아우터", "재킷"),
    (r"vest top", "상의", "민소매·탱크톱"),
    (r"\bvests?\b|gilet", "아우터", "조끼"),
    # 하의
    (r"jean|denim", "하의", "청바지"),
    (r"legging|tights|jegging", "하의", "레깅스"),
    (r"skirt|skort", "하의", "스커트"),
    (r"\bshorts\b|bermuda", "하의", "반바지"),
    (r"pant|trouser|chino|jogger|sweatpant|capri|culotte|lower body", "하의", "바지"),
    # 상의
    (r"t-?shirt|\btees?\b|henley", "상의", "티셔츠"),
    (r"hood|sweatshirt|fleece", "상의", "후드·맨투맨"),
    (r"cardigan", "상의", "카디건"),
    (r"sweater|pullover|knit|jumper|turtleneck", "상의", "니트·스웨터"),
    (r"tank|cami|sleeveless|tube top", "상의", "민소매·탱크톱"),
    (r"polo", "상의", "폴로"),
    (r"blouse|button-?down|\bshirts?\b|tunic", "상의", "셔츠·블라우스"),
    (r"jersey", "상의", "티셔츠"),
    (r"\btops?\b|bodysuit|leotard|unitard|crop|upper body", "상의", "기타 상의"),
]

TYPE_FALLBACK = {  # 규칙에 안 맞을 때 상위 종류로 중분류
    "Shoes": "신발", "Jewelry": "액세서리", "Accessories": "액세서리", "Watches": "액세서리",
    "Handbags & Wallets": "가방·지갑",
}

_TARGETS = [(re.compile(p, re.I), name) for p, name in TARGETS]
_RULES = [(re.compile(p, re.I), mid, leaf) for p, mid, leaf in RULES]
LEVEL1 = ["여성", "남성", "아동", "공용"]
LEVEL2 = ["상의", "하의", "원피스·세트", "아우터", "이너웨어", "수영복", "신발", "가방·지갑", "액세서리", "기타"]


GENERIC = {"세트", "플랫·로퍼", "기타 신발", "기타 상의", "기타 액세서리"}
# 상품명에서 지울 수식어 (품목이 아니라 품목을 꾸미는 말)
_TITLE_NOISE = re.compile(r"(short|long|3/4|cap|puff|flutter|bell)[ -]?sleeves?|sleeveless|running|athletic|training|walking|"
                          r"tennis|golf|yoga|workout|gym|boot ?cut|flat[ -]front|tie[ -]?dye|high[ -]waist(ed)?|"
                          r"slim[ -]fit|loose[ -]fit|button[ -]front", re.I)


def _match(seg):
    return next(((mid, leaf) for rx, mid, leaf in _RULES if rx.search(seg)), None)


ROOTS = {"clothing, shoes & jewelry"}       # Amazon 공통 최상위 단계 — 'Shoes'가 들어 있어 분류에서 뺀다


def _segments(path) -> list[str]:
    return [s.strip() for s in str(path or "").split(">") if s.strip() and s.strip().lower() not in ROOTS]


def classify(path, target=None, cat_type=None, title=None) -> tuple[str, str, str]:
    """(대분류, 중분류, 소분류). path: 카테고리 경로, target: 대상 열(cat_target), cat_type: 상위 종류, title: 상품명(보조)."""
    segs = _segments(path)
    level1 = "공용"
    for text in [str(target or ""), *segs]:
        hit = next((name for rx, name in _TARGETS if rx.search(text)), None)
        if hit:
            level1 = hit
            break
    hits = [h for h in (_match(seg) for seg in reversed(segs)) if h]      # 깊은 단계부터
    if hits:
        specific = next((h for h in hits if h[1] not in GENERIC), None)
        mid, leaf = hits[0] if hits[0][1] not in GENERIC or specific is None else specific
        return level1, mid, leaf
    if title:                                                    # 경로에 품목 정보가 없을 때만 상품명으로
        hit = _match(_TITLE_NOISE.sub(" ", str(title)))
        if hit:
            return level1, *hit
    mid = TYPE_FALLBACK.get(str(cat_type or ""), "기타")
    return level1, mid, "기타"


def classify_frame(items):
    """items(source · categories · cat_target · cat_type) → 같은 순서의 (대, 중, 소) 열 3개 DataFrame."""
    import pandas as pd
    titles = items["title"] if "title" in items else [None] * len(items)
    rows = [classify(c, t, ty, ti) for c, t, ty, ti in zip(items["categories"], items["cat_target"], items["cat_type"], titles)]
    return pd.DataFrame(rows, columns=["category_l1", "category_l2", "category_l3"], index=items.index)
