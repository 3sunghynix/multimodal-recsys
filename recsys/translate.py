"""한국어 검색어 → 영어. 패션 용어 사전을 먼저 쓰고, 사전에 없는 단어만 번역 모델(mt)에 보낸다.

일반 번역 모델(opus-mt-ko-en)은 패션 용어에 약하다 (실측: 청자켓 → "Celadon", 양말 → "Sheep's Horses",
울 스웨터 → "Oh, no, no ..."). 검색어는 짧고 어휘가 한정적이라 사전이 더 정확하고 빠르며 결과가 항상 같다.

- 띄어쓰기 단위마다 사전의 가장 긴 단어부터 앞에서 맞춘다 (예: 청자켓 = 청자켓, 하이웨이스트청바지 = 하이웨이스트 + 청바지)
- 끝에 붙은 조사(은 · 는 · 을 · 를 · 에 · 의 …)는 떼고 다시 맞춘다
- 끝까지 못 맞춘 단어는 mt(목록 → 목록)로 번역, mt가 없으면 그대로 둔다
- 한글이 없는 단어(영어 · 숫자)는 그대로
"""
import re

HANGUL = re.compile(r"[가-힣]")
PARTICLES = ("에서", "으로", "이랑", "은", "는", "이", "가", "을", "를", "에", "의", "도", "로", "랑", "용")

GLOSSARY = {
    # 색
    "검은색": "black", "검정색": "black", "검정": "black", "검은": "black", "블랙": "black",
    "흰색": "white", "하얀색": "white", "하얀": "white", "화이트": "white",
    "빨간색": "red", "빨간": "red", "빨강": "red", "레드": "red",
    "파란색": "blue", "파란": "blue", "파랑": "blue", "블루": "blue", "하늘색": "light blue",
    "남색": "navy", "네이비": "navy", "회색": "gray", "그레이": "gray", "베이지": "beige",
    "갈색": "brown", "브라운": "brown", "초록색": "green", "초록": "green", "그린": "green",
    "노란색": "yellow", "노란": "yellow", "옐로우": "yellow", "분홍색": "pink", "분홍": "pink", "핑크": "pink",
    "보라색": "purple", "보라": "purple", "퍼플": "purple", "카키": "khaki", "아이보리": "ivory",
    "오렌지": "orange", "주황": "orange", "금색": "gold", "골드": "gold", "은색": "silver", "실버": "silver",
    # 상의
    "원피스": "dress", "드레스": "dress", "티셔츠": "t-shirt", "반팔티": "short sleeve t-shirt",
    "긴팔티": "long sleeve t-shirt", "크롭티": "crop top", "탑": "top", "나시": "tank top", "민소매": "sleeveless",
    "셔츠": "shirt", "남방": "shirt", "블라우스": "blouse", "니트": "knit sweater", "스웨터": "sweater",
    "가디건": "cardigan", "후드티": "hoodie", "후드": "hoodie", "후디": "hoodie", "맨투맨": "sweatshirt",
    "폴로": "polo", "티": "t-shirt",
    # 아우터
    "자켓": "jacket", "재킷": "jacket", "청자켓": "denim jacket", "청재킷": "denim jacket",
    "가죽자켓": "leather jacket", "코트": "coat", "트렌치코트": "trench coat", "패딩": "puffer jacket",
    "롱패딩": "long puffer coat", "조끼": "vest", "베스트": "vest", "블레이저": "blazer", "점퍼": "jacket",
    "바람막이": "windbreaker", "야상": "parka", "정장": "suit",
    # 하의
    "바지": "pants", "팬츠": "pants", "청바지": "jeans", "진": "jeans", "슬랙스": "slacks trousers",
    "반바지": "shorts", "숏팬츠": "shorts", "레깅스": "leggings", "조거": "jogger", "조거팬츠": "jogger pants",
    "카고": "cargo", "와이드": "wide leg", "치마": "skirt", "스커트": "skirt", "미니": "mini", "롱": "long",
    "플리츠": "pleated", "트레이닝복": "tracksuit", "추리닝": "tracksuit",
    # 이너 · 수영
    "수영복": "swimsuit", "비키니": "bikini", "래시가드": "rash guard", "잠옷": "pajamas", "파자마": "pajamas",
    "속옷": "underwear", "브라": "bra", "팬티": "panties", "양말": "socks", "스타킹": "tights",
    # 신발
    "신발": "shoes", "운동화": "sneakers", "스니커즈": "sneakers", "러닝화": "running shoes", "구두": "dress shoes",
    "로퍼": "loafers", "부츠": "boots", "앵클": "ankle", "샌들": "sandals", "슬리퍼": "slippers", "힐": "heels",
    "플랫슈즈": "flats", "슬립온": "slip-on",
    # 잡화
    "모자": "hat", "야구모자": "baseball cap", "볼캡": "baseball cap", "캡": "cap", "버킷햇": "bucket hat",
    "비니": "beanie", "가방": "bag", "핸드백": "handbag", "백팩": "backpack", "토트백": "tote bag",
    "크로스백": "crossbody bag", "에코백": "tote bag", "지갑": "wallet", "벨트": "belt", "스카프": "scarf",
    "머플러": "scarf", "목도리": "scarf", "장갑": "gloves", "선글라스": "sunglasses", "안경": "glasses",
    "목걸이": "necklace", "귀걸이": "earrings", "반지": "ring", "팔찌": "bracelet", "시계": "watch",
    # 소재 · 무늬
    "가죽": "leather", "울": "wool", "린넨": "linen", "리넨": "linen", "실크": "silk", "면": "cotton",
    "코튼": "cotton", "데님": "denim", "청": "denim", "캐시미어": "cashmere", "꽃무늬": "floral",
    "플로럴": "floral", "스트라이프": "striped", "줄무늬": "striped", "체크": "plaid", "도트": "polka dot",
    "땡땡이": "polka dot", "레이스": "lace", "기모": "fleece", "플리스": "fleece",
    # 대상 · 계절 · 용도 · 핏
    "남성": "men's", "남자": "men's", "여성": "women's", "여자": "women's", "아동": "kids", "키즈": "kids",
    "어린이": "kids", "아기": "baby", "유아": "toddler", "여름": "summer", "겨울": "winter", "봄": "spring",
    "가을": "fall", "요가": "yoga", "스포츠": "sports", "운동": "workout", "러닝": "running", "등산": "hiking",
    "골프": "golf", "하이웨이스트": "high waist", "오버핏": "oversized", "슬림": "slim", "긴팔": "long sleeve",
    "반팔": "short sleeve", "캐주얼": "casual", "포멀": "formal", "빈티지": "vintage", "웨딩": "wedding",
}
_MAX = max(map(len, GLOSSARY))


def _segment(word: str):
    """사전으로 앞에서부터 가장 긴 단어를 맞춘다. 끝까지 맞추면 영어 목록, 아니면 None."""
    out, i = [], 0
    while i < len(word):
        for j in range(min(len(word), i + _MAX), i, -1):
            if word[i:j] in GLOSSARY:
                out.append(GLOSSARY[word[i:j]])
                i = j
                break
        else:
            return None
    return out


def _word(word: str):
    seg = _segment(word)
    if seg is None:
        for p in PARTICLES:
            if word.endswith(p) and len(word) > len(p):
                seg = _segment(word[: -len(p)])
                if seg is not None:
                    break
    return seg


def translate_ko(query: str, mt=None) -> tuple[str, list[str]]:
    """반환: (영어 검색어, 사전에 없어 mt로 보낸 단어 목록). mt: list[str] → list[str] (없으면 원문 유지)."""
    parts, unknown = [], []
    for w in query.split():
        if not HANGUL.search(w):
            parts.append(w)
            continue
        seg = _word(w)
        if seg is None:
            unknown.append(w)
            parts.append(None)
        else:
            parts.extend(seg)
    if unknown:
        tr = mt(unknown) if mt else unknown
        tr = iter(t.strip(" .").lower() for t in tr)
        parts = [next(tr) if p is None else p for p in parts]
    out = []
    for p in parts:                                   # 같은 단어 연속 중복 제거 (예: dress dress)
        if not out or out[-1] != p:
            out.append(p)
    return " ".join(out), unknown
