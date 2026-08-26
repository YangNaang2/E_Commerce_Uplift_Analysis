# -*- coding: utf-8 -*-
"""
run15: 주문취소 및 배송지연 분석 (H1~H4 Uplift 파이프라인과 독립)

배경: 기존에는 주문취소여부·배송리드타임을 전처리 단계의 데이터 정합성 체크용으로만
다뤘음(예: "배송리드타임 결측이 취소건과 정확히 일치") -> ①세그먼트별 취소율·배송지연
차이 ②취소 경험이 향후 재구매에 영향을 주는지를 신규로 확인.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
OUT = BASE
TRAIN_INDEX_DATE = pd.Timestamp("2025-09-16")

df = pd.read_csv(
    BASE + "merged_master_enriched.csv", encoding="utf-8-sig",
    usecols=["회원번호", "주문일시", "구매금액", "is_cancelled", "delivery_leadtime_days",
             "구독여부", "region_tier", "age_gender_segment", "요일", "주말여부"],
)
df["주문일시_dt"] = pd.to_datetime(df["주문일시"])
print(f"로드 완료: {len(df):,}행")

lines = []
lines.append(f"전체 주문 {len(df):,}건, 전체 취소율 {df['is_cancelled'].mean()*100:.2f}%\n")

# --- 1. 세그먼트별 취소율 ---
lines.append("=== 세그먼트별 취소율 ===")
for col in ["구독여부", "region_tier", "요일"]:
    g = df.groupby(col)["is_cancelled"].agg(["mean", "count"])
    g["mean"] = (g["mean"] * 100).round(2)
    lines.append(f"\n[{col}]\n{g.to_string()}")

ct_sub = pd.crosstab(df["구독여부"], df["is_cancelled"])
chi2, p, _, _ = chi2_contingency(ct_sub)
n = ct_sub.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(ct_sub.shape) - 1)))
lines.append(f"\n[카이제곱] 취소여부 x 구독여부: chi2={chi2:.2f}, p={p:.4f}, Cramer's V={cramers_v:.4f}")

ct_region = pd.crosstab(df["region_tier"], df["is_cancelled"])
chi2r, pr, _, _ = chi2_contingency(ct_region)
nr = ct_region.values.sum()
cramers_v_r = np.sqrt(chi2r / (nr * (min(ct_region.shape) - 1)))
lines.append(f"[카이제곱] 취소여부 x region_tier: chi2={chi2r:.2f}, p={pr:.4f}, Cramer's V={cramers_v_r:.4f}")

# --- 2. 배송 리드타임 (취소 제외) ---
lt = df[df["is_cancelled"] == 0]["delivery_leadtime_days"].dropna()
lines.append(f"\n=== 배송 리드타임(취소 제외, n={len(lt):,}) ===")
lines.append(f"평균 {lt.mean():.2f}일 / 중앙값 {lt.median():.1f}일 / P90 {lt.quantile(0.9):.1f}일")

lt_by_sub = df[df["is_cancelled"] == 0].groupby("구독여부")["delivery_leadtime_days"].median()
lines.append(f"\n[구독여부별 리드타임 중앙값]\n{lt_by_sub.to_string()}")
lt_by_region = df[df["is_cancelled"] == 0].groupby("region_tier")["delivery_leadtime_days"].median()
lines.append(f"\n[region_tier별 리드타임 중앙값]\n{lt_by_region.to_string()}")

# --- 3. 취소 경험 -> 향후 재구매 영향 (pre-period 취소이력 vs Train Y) ---
pre = df[df["주문일시_dt"] <= TRAIN_INDEX_DATE]
cust_cancel = pre.groupby("회원번호")["is_cancelled"].agg(ever_cancelled=lambda s: int((s == 1).any()))
cust_leadtime = pre[pre["is_cancelled"] == 0].groupby("회원번호")["delivery_leadtime_days"].mean().rename("avg_leadtime")

snap = pd.read_csv(BASE + "snapshot_train.csv", encoding="utf-8-sig",
                    usecols=["회원번호", "y_repurchase_14d", "label_observable_flag"])
snap = snap[snap["label_observable_flag"] == 1]

merged = snap.merge(cust_cancel, on="회원번호", how="left").merge(cust_leadtime, on="회원번호", how="left")
merged["ever_cancelled"] = merged["ever_cancelled"].fillna(0)

rp_by_cancel = merged.groupby("ever_cancelled")["y_repurchase_14d"].agg(["mean", "count"])
rp_by_cancel["mean"] = (rp_by_cancel["mean"] * 100).round(2)
lines.append(f"\n=== 취소 경험(Train 기준일 이전) x 향후 14일 재구매율 ===\n{rp_by_cancel.to_string()}")

ct_cancel_y = pd.crosstab(merged["ever_cancelled"], merged["y_repurchase_14d"])
chi2c, pc, _, _ = chi2_contingency(ct_cancel_y)
lines.append(f"[카이제곱] 취소경험 x 재구매여부: chi2={chi2c:.2f}, p={pc:.4f}")

lt_valid = merged.dropna(subset=["avg_leadtime"])
lt_repurchase = lt_valid[lt_valid["y_repurchase_14d"] == 1]["avg_leadtime"]
lt_norepurchase = lt_valid[lt_valid["y_repurchase_14d"] == 0]["avg_leadtime"]
u_stat, u_p = mannwhitneyu(lt_repurchase, lt_norepurchase, alternative="two-sided")
lines.append(f"\n=== 평균 배송리드타임 x 재구매여부 (Mann-Whitney U) ===")
lines.append(f"재구매O 평균리드타임 중앙값 {lt_repurchase.median():.2f}일 (n={len(lt_repurchase):,}) "
             f"/ 재구매X {lt_norepurchase.median():.2f}일 (n={len(lt_norepurchase):,}) / p={u_p:.4f}")

result_text = "\n".join(lines)
with open(OUT + "run15_result.txt", "w", encoding="utf-8") as f:
    f.write(result_text)

merged.to_csv(BASE + "run15_cancellation_repurchase_merged.csv", index=False, encoding="utf-8-sig")
print("저장 완료: run15_result.txt, run15_cancellation_repurchase_merged.csv")
