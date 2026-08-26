# -*- coding: utf-8 -*-
"""
run18: 결제/적립수단별 구매행동 (H1~H4 Uplift 파이프라인과 독립)

등록카드(25종 발급기관명)를 은행/카드사/간편결제/기타 4그룹으로 재분류한 뒤,
그룹별 AOV·구매빈도·취소율·구독여부 비율을 비교.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
OUT = BASE

BANK = {"국민은행", "신한은행", "농협중앙회", "우리은행", "기업은행", "하나은행", "새마을금고",
        "부산은행", "SC제일은행", "광주은행", "씨티은행", "우체국", "대구은행", "신협중앙회",
        "외환은행", "경남은행"}
CARD_CO = {"롯데카드", "신한카드", "국민카드", "BC카드", "현대카드", "삼성카드"}
SIMPLE_PAY = {"연결앱결제"}


def to_group(x):
    if x in BANK:
        return "은행"
    if x in CARD_CO:
        return "카드사"
    if x in SIMPLE_PAY:
        return "간편결제"
    return "기타"


df = pd.read_csv(BASE + "merged_master.csv", encoding="utf-8-sig",
                  usecols=["회원번호", "등록카드", "구매금액", "주문취소여부", "구독여부", "주문일시"])
df["결제그룹"] = df["등록카드"].apply(to_group)
print(f"로드 {len(df):,}행, 결제그룹 분포:\n{df['결제그룹'].value_counts()}")

not_cancelled = df[df["주문취소여부"] != "주문취소"]
order_summary = not_cancelled.groupby(["회원번호", "결제그룹"]).agg(
    n_orders=("구매금액", "size"), total_amt=("구매금액", "sum")
).reset_index()
cust_summary = order_summary.groupby("결제그룹").agg(
    회원수=("회원번호", "nunique"),
    평균주문건수=("n_orders", "mean"),
    평균AOV=("total_amt", lambda s: (s / order_summary.loc[s.index, "n_orders"]).mean()),
).round(1)

cancel_rate = df.groupby("결제그룹")["주문취소여부"].apply(lambda s: (s == "주문취소").mean() * 100).round(2)
sub_member = df.drop_duplicates("회원번호")[["회원번호", "결제그룹", "구독여부"]]
sub_rate = sub_member.groupby("결제그룹")["구독여부"].apply(lambda s: (s == True).mean() * 100).round(2)

cust_summary["취소율(%)"] = cancel_rate
cust_summary["구독률(%)"] = sub_rate
cust_summary = cust_summary.sort_values("회원수", ascending=False)

ct = pd.crosstab(df["결제그룹"], df["주문취소여부"] == "주문취소")
chi2, p, _, _ = chi2_contingency(ct)
n = ct.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

groups_aov = [g["total_amt"].values / g["n_orders"].values for _, g in order_summary.groupby("결제그룹")]
kw_stat, kw_p = kruskal(*groups_aov)

with open(OUT + "run18_result.txt", "w", encoding="utf-8") as f:
    f.write("=== 결제그룹별 요약 ===\n")
    f.write(cust_summary.to_string())
    f.write(f"\n\n[카이제곱] 결제그룹 x 취소여부: chi2={chi2:.2f}, p={p:.4f}, Cramer's V={cramers_v:.4f}")
    f.write(f"\n[Kruskal-Wallis] 결제그룹별 AOV 차이: H={kw_stat:.2f}, p={kw_p:.4f}")

cust_summary.to_csv(BASE + "run18_payment_group_summary.csv", encoding="utf-8-sig")
print("저장 완료: run18_payment_group_summary.csv, run18_result.txt")
