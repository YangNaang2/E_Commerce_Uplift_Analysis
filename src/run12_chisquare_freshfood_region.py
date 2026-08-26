"""
Model 4: 신선식품 vs 일반상품 구매 비중이 배송지역(region_tier)에 따라
통계적으로 유의하게 다른가? — 카이제곱 독립성 검정(chi-square test of independence)

배경: H1~H4 가설검증과는 별개로 추가 검토한 분석. 새벽배송/무료배송 조건이 좋은 Tier1 지역일수록
콜드체인 신선식품 주문 비중이 높고, 배송조건이 열악한 Tier2/3는 상대적으로 일반(가공·건식)
상품 비중이 높을 것이라는 가설을 검증한다. H2(지역×배송 재구매효과)와는 독립적인 분석이며,
기존 Uplift 모델링에는 영향을 주지 않는다.

데이터 소스:
- merged_master.csv: 제품번호 -> 물품대분류(58종) 매핑용. 물품대분류는 제품번호에 대해
  1:1 결정적(1,917개 제품 전부 검증 완료)이므로 조인 키로 안전하게 사용 가능.
- merged_master_enriched.csv: 회원번호/제품번호/주문일시/is_cancelled/region_tier 보유.
  물품대분류가 없어 위 매핑을 조인해서 채운다.

신선식품 분류(58종 -> 21종 신선/37종 일반)은 "냉장 보관이 필요한 원물 또는 원물 기반
단순가공 식품"을 기준으로 분석자가 직접 분류했으며, 아래 FRESH_CATEGORIES에 명시한다.
"""
import pandas as pd
from scipy import stats

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"

FRESH_CATEGORIES = {
    "두부/유부", "잎/줄기채소", "알", "열매채소", "양념채소", "유제품", "뿌리채소",
    "과일", "콩나물", "버섯", "과일채소", "냉동수산", "해조", "중량(정육)", "쌈채소류",
    "돼지", "소", "닭/오리", "생물수산", "김치", "손질한채소",
}

print("=" * 70)
print("Model 4: 신선식품 vs 일반상품 × 배송지역(region_tier) 카이제곱 검정")
print("=" * 70)

# 1) 제품번호 -> 물품대분류 매핑 (merged_master.csv, 1:1 결정적 관계 사전 검증됨)
prod_cat = pd.read_csv(
    BASE + "merged_master.csv", encoding="utf-8-sig",
    usecols=["제품번호", "물품대분류"]
).drop_duplicates()
assert prod_cat["제품번호"].is_unique, "제품번호별 물품대분류가 1개가 아님 - 매핑 재검증 필요"

n_fresh_cat = sum(1 for c in prod_cat["물품대분류"].unique() if c in FRESH_CATEGORIES)
n_total_cat = prod_cat["물품대분류"].nunique()
print(f"\n물품대분류 총 {n_total_cat}종 중 신선식품 분류 {n_fresh_cat}종, 일반상품 {n_total_cat - n_fresh_cat}종")

# 2) 주문 데이터에 region_tier + 물품대분류 결합
orders = pd.read_csv(
    BASE + "merged_master_enriched.csv", encoding="utf-8-sig",
    usecols=["회원번호", "제품번호", "is_cancelled", "region_tier"]
)
valid = orders[orders["is_cancelled"] == 0].merge(prod_cat, on="제품번호", how="left")
missing = valid["물품대분류"].isna().sum()
print(f"취소 제외 주문 {len(valid):,}건 중 물품대분류 매핑 실패 {missing:,}건 (제외 처리)")
valid = valid.dropna(subset=["물품대분류", "region_tier"])

valid["신선식품여부"] = valid["물품대분류"].apply(
    lambda c: "신선식품" if c in FRESH_CATEGORIES else "일반상품"
)

# "미상" region_tier는 배송조건 자체를 알 수 없는 결측 취급이라 검정에서 제외
valid_known = valid[valid["region_tier"] != "미상"].copy()
print(f"region_tier=미상 제외 후 분석 대상 {len(valid_known):,}건 "
      f"(제외 {len(valid) - len(valid_known):,}건)")

# 3) 교차표 + 카이제곱 검정
ct = pd.crosstab(valid_known["region_tier"], valid_known["신선식품여부"])
ct_pct = pd.crosstab(valid_known["region_tier"], valid_known["신선식품여부"], normalize="index") * 100

print("\n--- 교차표 (건수) ---")
print(ct)
print("\n--- 교차표 (지역 내 비율, %) ---")
print(ct_pct.round(2))

chi2, p, dof, expected = stats.chi2_contingency(ct)
n = ct.values.sum()
cramers_v = (chi2 / (n * (min(ct.shape) - 1))) ** 0.5

print("\n--- 카이제곱 독립성 검정 결과 ---")
print(f"chi2 = {chi2:.2f}, dof = {dof}, p-value = {p:.6g}")
print(f"Cramer's V (효과크기) = {cramers_v:.4f}")
if p < 0.05:
    print("=> p < 0.05: region_tier와 신선식품여부는 통계적으로 유의하게 연관되어 있음.")
else:
    print("=> p >= 0.05: region_tier와 신선식품여부 사이 통계적으로 유의한 연관성 없음.")

if cramers_v < 0.1:
    strength = "매우 약함(거의 무관)"
elif cramers_v < 0.3:
    strength = "약함"
elif cramers_v < 0.5:
    strength = "중간"
else:
    strength = "강함"
print(f"=> 효과크기 해석: {strength} (Cramer's V 기준)")

out = ct_pct.round(2)
out.to_csv(BASE + "run12_freshfood_region_crosstab.csv", encoding="utf-8-sig")

summary = pd.DataFrame([{
    "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": cramers_v,
    "n_orders": n, "유의여부(p<0.05)": p < 0.05,
}])
summary.to_csv(BASE + "run12_freshfood_region_chisquare_result.csv",
                index=False, encoding="utf-8-sig")

print(f"\n저장: run12_freshfood_region_crosstab.csv, run12_freshfood_region_chisquare_result.csv")
