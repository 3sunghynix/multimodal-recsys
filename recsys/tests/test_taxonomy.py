"""3단계 카테고리 분류 규칙."""
import pytest

from recsys.taxonomy import LEVEL1, LEVEL2, classify

A = "Clothing, Shoes & Jewelry > "


@pytest.mark.parametrize("path, target, cat_type, expect", [
    (A + "Women > Clothing > Dresses > Casual", "Women", "Clothing", ("여성", "원피스·세트", "원피스")),
    ("Garment Full body > Dress", "Women", "Clothing", ("여성", "원피스·세트", "원피스")),            # H&M
    (A + "Men > Clothing > Shirts > T-Shirts", "Men", "Clothing", ("남성", "상의", "티셔츠")),       # t-shirt가 shirt보다 먼저
    ("Garment Upper body > Vest top", "Women", "Clothing", ("여성", "상의", "민소매·탱크톱")),        # vest top ≠ 조끼
    (A + "Women > Clothing > Coats, Jackets & Vests > Vests", "Women", "Clothing", ("여성", "아우터", "조끼")),
    (A + "Women > Shoes > Sandals > Flats", "Women", "Shoes", ("여성", "신발", "샌들")),             # 일반 규칙보다 윗단계
    (A + "Women > Shoes > Flats", "Women", "Shoes", ("여성", "신발", "플랫·로퍼")),
    (A + "Women > Clothing > Shorts > Flat Front", "Women", "Clothing", ("여성", "하의", "반바지")),
    (A + "Women > Clothing > Lingerie, Sleep & Lounge > Sleep & Lounge > Sets", "Women", "Clothing", ("여성", "이너웨어", "잠옷·라운지웨어")),
    (A + "Women > Clothing > Swimsuits & Cover Ups > Bikinis > Sets", "Women", "Clothing", ("여성", "수영복", "수영복")),
    ("Socks & Tights > Socks", "Women", "Clothing", ("여성", "이너웨어", "양말·스타킹")),
    ("Garment Lower body > Leggings/Tights", "Sport", "Clothing", ("공용", "하의", "레깅스")),
    (A + "Boot Shop > Women > Fashion > Riding", "Boot Shop", "Women", ("여성", "신발", "부츠")),
    (A + "Girls > Jewelry > Earrings", "Girls", "Jewelry", ("아동", "액세서리", "귀걸이")),
    (A + "Women > Handbags & Wallets > Wallets", "Women", "Handbags & Wallets", ("여성", "가방·지갑", "지갑")),
    (A + "Novelty & More > Clothing > Novelty > Men", "Novelty & More", "Clothing", ("남성", "기타", "기타")),
    (A + "Women > Watches", "Women", "Watches", ("여성", "액세서리", "시계")),
])
def test_classify(path, target, cat_type, expect):
    assert classify(path, target, cat_type) == expect


def test_levels_are_closed_sets():
    paths = [A + "Women > Shoes > Boots", "Garment Upper body > Hoodie", "", None]
    for p in paths:
        l1, l2, _ = classify(p, "Men", "Clothing")
        assert l1 in LEVEL1 and l2 in LEVEL2


@pytest.mark.parametrize("title, leaf", [
    ("Women's Casual Maxi Dress", "원피스"),
    ("Women's Running Shorts Quick Dry", "반바지"),          # running ≠ 운동화
    ("High Waisted Bootcut Jeans", "청바지"),                # bootcut ≠ 부츠
    ("Short Sleeve Button Down Blouse", "셔츠·블라우스"),    # short sleeve ≠ 반바지
    ("Tie Dye Oversized T-Shirt", "티셔츠"),                 # tie dye ≠ 넥타이
])
def test_title_fallback_when_path_has_no_item(title, leaf):
    assert classify(A + "Women > Clothing", "Women", "Clothing", title)[2] == leaf


def test_title_not_used_when_path_has_item():
    # 경로에 품목이 있으면 상품명의 'dress'가 셔츠를 원피스로 바꾸지 않는다
    assert classify(A + "Men > Clothing > Shirts > Dress Shirts", "Men", "Clothing", "Slim Fit Dress Shirt")[2] == "셔츠·블라우스"
    assert classify(A + "Men > Clothing > Active > Active Base Layers", "Men", "Clothing") == ("남성", "이너웨어", "속옷")
    assert classify(A + "Sport Specific Clothing > Gymnastics > Girls > Leotards", "Sport Specific Clothing", "Gymnastics") == ("아동", "상의", "기타 상의")


def test_sweatshirt_and_cases():
    assert classify(A + "Women > Clothing > Fashion Hoodies & Sweatshirts", "Women", "Clothing")[2] == "후드·맨투맨"
    assert classify(A + "Women > Accessories > Sunglasses & Eyewear Accessories > Eyeglass Cases", "Women", "Accessories")[2] == "케이스·소품"
    assert classify(A + "Women > Accessories > Sunglasses & Eyewear Accessories > Sunglasses", "Women", "Accessories")[2] == "선글라스"
