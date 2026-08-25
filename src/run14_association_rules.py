# -*- coding: utf-8 -*-
"""
run14: 연관규칙 기반 장바구니 분석 [추가분석, H1~H4와 독립]

배경: 교수님/팀 피드백에서 "연관규칙 기반 마케팅"이 언급됨. 이 프로젝트는 basket 재집계
(품목행->주문 단위, preprocess_pipeline.py Step2b와 동일한 (회원번호, 주문일시_dt) 키)까지는
이미 했지만 "이 상품을 사면 저 상품도 산다"는 연관규칙 마이닝 자체는 없었음 -> 신규로 수행.

물품명(SKU, 1,917종)이 아니라 물품대분류(59종) 단위로 분석함 -> SKU 단위는 조합이 너무
희소해(주문당 평균 3.34개 품목) 유의미한 규칙이 거의 안 나오고, 대분류 단위가 "정육을 사면
유제품도 산다" 같은 실제 마케팅(교차판매 프로모션)에 바로 쓸 수 있는 해상도임.
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
OUT = "C:/Users/aidan/AppData/Local/Temp/claude/C--Users-aidan-OneDrive-----------/526a44f2-eab0-428f-be6f-d07d153f63c4/scratchpad/"

df = pd.read_csv(
    BASE + "merged_master.csv", encoding="utf-8-sig",
    usecols=["회원번호", "주문일시", "주문취소여부", "물품대분류"],
)
print(f"원본 로드: {len(df):,}행")

df = df[df["주문취소여부"] != "주문취소"].copy()

# basket = (회원번호, 주문일시) 조합, preprocess_pipeline.py의 basket_all과 동일 키
baskets = df.groupby(["회원번호", "주문일시"])["물품대분류"].apply(lambda s: set(s)).reset_index()
baskets["n_categories"] = baskets["물품대분류"].apply(len)
print(f"전체 basket: {len(baskets):,}건, 이 중 카테고리 2종 이상: {(baskets['n_categories'] >= 2).sum():,}건")

multi = baskets[baskets["n_categories"] >= 2]["물품대분류"].tolist()

te = TransactionEncoder()
te_ary = te.fit(multi).transform(multi)
onehot = pd.DataFrame(te_ary, columns=te.columns_)
print(f"원핫 인코딩 완료: basket {onehot.shape[0]:,} x 카테고리 {onehot.shape[1]}종")

MIN_SUPPORT = 0.01
freq = apriori(onehot, min_support=MIN_SUPPORT, use_colnames=True)
print(f"빈발 itemset(support>={MIN_SUPPORT}): {len(freq):,}개")

rules = association_rules(freq, metric="lift", min_threshold=1.0)
rules = rules[rules["antecedents"].apply(len) == 1]  # 단일 상품 -> 단일/복수 상품 규칙만 (해석 용이)
rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
rules = rules.sort_values("lift", ascending=False)

top = rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(20)
top.to_csv(OUT + "run14_association_rules_top20.csv", index=False, encoding="utf-8-sig")
rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_csv(
    BASE + "run14_association_rules_full.csv", index=False, encoding="utf-8-sig"
)

with open(OUT + "run14_result.txt", "w", encoding="utf-8") as f:
    f.write(f"basket 총 {len(baskets):,}건, 2종 이상 구매 basket {len(multi):,}건\n")
    f.write(f"연관규칙 총 {len(rules):,}개 (min_support={MIN_SUPPORT}, lift>=1.0)\n\n")
    f.write("=== lift 상위 20개 규칙 ===\n")
    f.write(top.to_string(index=False))

print("저장 완료: run14_association_rules_full.csv, run14_result.txt")
