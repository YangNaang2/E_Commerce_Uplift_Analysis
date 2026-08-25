# -*- coding: utf-8 -*-
"""
run09: A/B/C/D 개인별 Uplift 점수 백테스팅
- "각 파트 모델이 같은 고객을 Persuadable(타겟 우선순위 높음)로 보는가"를 회원번호 기준 병합해 확인
- 비교 대상 8개 스코어: A(run01b 로지스틱 대표모델 / run08b RF 재튜닝판), B/C/D(각 logit/rf)
- 대상 모집단: treatment_h4==0(비구독자, 실제 타겟팅 후보군) — 각 파트 스크립트가 전부 이 기준으로 Uplift를 계산했음
- 두 지표: (1) Spearman 순위상관 — 전반적 순서가 비슷한가 (2) 상위 20% Jaccard 겹침 — 실제 타겟 후보군이 겹치는가
"""
import numpy as np
import pandas as pd

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
TOP_PCT = 0.20

a_logit = pd.read_csv(BASE + "run01b_uplift_scores_holdout.csv", encoding="utf-8-sig")
a_rf = pd.read_csv(BASE + "run08b_uplift_forest_scores.csv", encoding="utf-8-sig")
b_logit = pd.read_csv(BASE + "uplift_scores_partB_logit.csv", encoding="utf-8-sig")
b_rf = pd.read_csv(BASE + "uplift_scores_partB_rf.csv", encoding="utf-8-sig")
c = pd.read_csv(BASE + "uplift_scores_partC.csv", encoding="utf-8-sig")
d = pd.read_csv(BASE + "uplift_scores_partD.csv", encoding="utf-8-sig")

# --- 회원번호 기준 병합 준비: 각 파일에서 uplift 컬럼만 추출해 파트_모델명으로 통일 ---
frames = {
    "A_logit": a_logit[["회원번호", "treatment_h4", "uplift"]].rename(columns={"uplift": "uplift_A_logit"}),
    "A_rf": a_rf[["회원번호", "treatment_h4", "uplift_forest"]].rename(columns={"uplift_forest": "uplift_A_rf"}),
    "B_logit": b_logit[["회원번호", "treatment_h4", "uplift"]].rename(columns={"uplift": "uplift_B_logit"}),
    "B_rf": b_rf[["회원번호", "treatment_h4", "uplift"]].rename(columns={"uplift": "uplift_B_rf"}),
    "C_logit": c[["회원번호", "treatment_h4", "uplift_logit"]].rename(columns={"uplift_logit": "uplift_C_logit"}),
    "C_rf": c[["회원번호", "treatment_h4", "uplift_rf"]].rename(columns={"uplift_rf": "uplift_C_rf"}),
    "D_logit": d[["회원번호", "treatment_h4", "uplift_logit"]].rename(columns={"uplift_logit": "uplift_D_logit"}),
    "D_rf": d[["회원번호", "treatment_h4", "uplift_rf"]].rename(columns={"uplift_rf": "uplift_D_rf"}),
}

# --- treatment_h4 정합성 체크: 회원번호가 같으면 4개 파트에서 treatment_h4도 같아야 함 ---
# 주의: treatment_h4는 구독여부 unresolved(18.7%)인 사람은 NaN이고, pandas에서 NaN != NaN은
# 항상 True로 평가되므로 단순 != 비교는 오탐을 냄. NaN을 결측 그대로 비교(둘 다 NaN이면 일치로 간주)해야 함.
merged = frames["A_logit"][["회원번호", "treatment_h4"]].rename(columns={"treatment_h4": "treatment_h4_A"})
for key in ["B_logit", "C_logit", "D_logit"]:
    part = key.split("_")[0]
    merged = merged.merge(
        frames[key][["회원번호", "treatment_h4"]].rename(columns={"treatment_h4": f"treatment_h4_{part}"}),
        on="회원번호", how="inner",
    )
cols4 = ["treatment_h4_A", "treatment_h4_B", "treatment_h4_C", "treatment_h4_D"]
agree = merged[cols4].apply(lambda row: row.dropna().nunique() <= 1, axis=1)
real_conflict = merged[~agree]
na_pattern_consistent = merged[cols4].isna().nunique(axis=1).eq(1).all()
print(f"[정합성 체크] inner join 대상 {len(merged)}명 중 진짜 값 충돌(0 vs 1): {len(real_conflict)}명 "
      f"(NaN 패턴은 4개 파트 전부 동일: {na_pattern_consistent})")

# --- 8개 uplift 점수를 회원번호 기준 inner join으로 병합 ---
score = frames["A_logit"][["회원번호", "treatment_h4", "uplift_A_logit"]]
for key in ["A_rf", "B_logit", "B_rf", "C_logit", "C_rf", "D_logit", "D_rf"]:
    col = f"uplift_{key}"
    score = score.merge(frames[key][["회원번호", col]], on="회원번호", how="inner")

print(f"[병합] inner join 최종 표본: {len(score)}명 (A/B/C=12,391명, D=12,388명 중 교집합)")

score.to_csv(BASE + "run09_backtesting_merged_scores.csv", index=False, encoding="utf-8-sig")

# --- 타겟 모집단: 비구독자(treatment_h4==0)만 대상 (각 파트 스크립트의 target 정의와 동일) ---
target = score[score["treatment_h4"] == 0].copy()
score_cols = ["uplift_A_logit", "uplift_A_rf", "uplift_B_logit", "uplift_B_rf",
              "uplift_C_logit", "uplift_C_rf", "uplift_D_logit", "uplift_D_rf"]
print(f"[타겟 모집단] 비구독자 {len(target)}명 기준으로 상관/겹침 분석")

# ================= 1) Spearman 순위상관 =================
corr = target[score_cols].corr(method="spearman")
print("\n=== 1) Spearman 순위상관 행렬 ===")
print(corr.round(3))
corr.to_csv(BASE + "run09_spearman_correlation.csv", encoding="utf-8-sig")

# ================= 2) 상위 20% Jaccard 겹침 =================
n_top = int(len(target) * TOP_PCT)
top_sets = {col: set(target.nlargest(n_top, col)["회원번호"]) for col in score_cols}

jac = pd.DataFrame(index=score_cols, columns=score_cols, dtype=float)
for r in score_cols:
    for c_ in score_cols:
        inter = len(top_sets[r] & top_sets[c_])
        union = len(top_sets[r] | top_sets[c_])
        jac.loc[r, c_] = inter / union if union else np.nan

print(f"\n=== 2) 상위 {int(TOP_PCT*100)}%(n={n_top}) Jaccard 겹침 행렬 ===")
print(jac.round(3))
jac.to_csv(BASE + "run09_top20pct_jaccard.csv", encoding="utf-8-sig")

# --- 4개 파트(A/B/C/D) 모두의 상위20%에 동시에 들어가는 "만장일치 Persuadable" 고객 수 ---
all_four_logit = top_sets["uplift_A_logit"] & top_sets["uplift_B_logit"] & top_sets["uplift_C_logit"] & top_sets["uplift_D_logit"]
all_four_rf = top_sets["uplift_A_rf"] & top_sets["uplift_B_rf"] & top_sets["uplift_C_rf"] & top_sets["uplift_D_rf"]
expected_if_random = len(target) * (TOP_PCT ** 4)
print(f"\n[만장일치] logit 4개 파트 상위20% 교집합: {len(all_four_logit)}명 (무작위 기대치 {expected_if_random:.1f}명)")
print(f"[만장일치] rf 4개 파트 상위20% 교집합: {len(all_four_rf)}명 (무작위 기대치 {expected_if_random:.1f}명)")

print("\n저장 완료: run09_backtesting_merged_scores.csv, run09_spearman_correlation.csv, run09_top20pct_jaccard.csv")
