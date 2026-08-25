# -*- coding: utf-8 -*-
"""
A파트: run01 베이스라인 T-learner + H1(가격구간) 검증
Treatment=구독여부(treatment_h4), Y=재구매여부(y_repurchase_14d)
구독자 모델 A / 비구독자 모델 B 각각 학습 -> Uplift(X) = A(X) - B(X)
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
tr = pd.read_csv(BASE + "snapshot_train.csv", encoding="utf-8-sig")
ho = pd.read_csv(BASE + "snapshot_holdout.csv", encoding="utf-8-sig")

for d in (tr, ho):
    d["log_frequency"] = np.log1p(d["frequency"])
    d["log_monetary"] = np.log1p(d["monetary"])
    d["log_aov"] = np.log1p(d["aov"])
    d["log_recency"] = np.log1p(d["recency_days"])

NUM_FEATS = ["log_frequency", "log_monetary", "log_aov", "log_recency", "regularity_cv", "reward_usage_rate", "나이", "preperiod_제철구매비율"]
CAT_FEATS = ["age_band_h4", "gender_h4", "region_tier", "가격구간"]
ALL_FEATS = NUM_FEATS + CAT_FEATS

pre = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
])

train_known = tr[tr["include_in_uplift_model"] == 1].copy()
print(f"[run01] Train 학습표본: {len(train_known)} (구독자 {sum(train_known.treatment_h4==1)} / 비구독자 {sum(train_known.treatment_h4==0)})")

X_train_known = pre.fit_transform(train_known[ALL_FEATS])
y_train_known = train_known["y_repurchase_14d"].values
t_train_known = train_known["treatment_h4"].values

X_sub = X_train_known[t_train_known == 1]
y_sub = y_train_known[t_train_known == 1]
X_non = X_train_known[t_train_known == 0]
y_non = y_train_known[t_train_known == 0]

model_A = LogisticRegression(max_iter=2000, C=1.0).fit(X_sub, y_sub)  # 구독자 모델
model_B = LogisticRegression(max_iter=2000, C=1.0).fit(X_non, y_non)  # 비구독자 모델

# --- Holdout 평가 (모델 품질 체크, 알려진 라벨만) ---
ho_known = ho[ho["include_in_uplift_model"] == 1].copy()
X_ho_known = pre.transform(ho_known[ALL_FEATS])
ho_sub_mask = ho_known["treatment_h4"] == 1
ho_non_mask = ho_known["treatment_h4"] == 0

auc_A = roc_auc_score(ho_known.loc[ho_sub_mask, "y_repurchase_14d"], model_A.predict_proba(X_ho_known[ho_sub_mask.values])[:, 1])
auc_B = roc_auc_score(ho_known.loc[ho_non_mask, "y_repurchase_14d"], model_B.predict_proba(X_ho_known[ho_non_mask.values])[:, 1])
print(f"[run01] Holdout AUC - 구독자모델(A): {auc_A:.3f} / 비구독자모델(B): {auc_B:.3f}")

# --- 전체 Holdout(구독여부 unresolved 포함)에 두 모델 적용 -> Uplift(X) ---
X_ho_all = pre.transform(ho[ALL_FEATS])
p_A = model_A.predict_proba(X_ho_all)[:, 1]
p_B = model_B.predict_proba(X_ho_all)[:, 1]
ho["uplift"] = p_A - p_B
ho["p_A_구독시예측"] = p_A
ho["p_B_비구독시예측"] = p_B

print("\n[run01] 가격구간별 평균 Uplift (H1 검증) - Holdout 전체 고객 기준")
seg_summary = ho.groupby("가격구간", observed=True)["uplift"].agg(["mean", "median", "std", "count"])
print(seg_summary)

print("\n[run01] 가격구간별 평균 Uplift - 현재 비구독자(treatment_h4==0)만, 실제 타겟 후보군")
target_pool = ho[ho["treatment_h4"] == 0]
seg_summary_target = target_pool.groupby("가격구간", observed=True)["uplift"].agg(["mean", "median", "std", "count"])
print(seg_summary_target)

# Uplift 4분면 스타일 요약: 비구독자 중 Persuadable(uplift 상위) 후보
target_pool_sorted = target_pool.sort_values("uplift", ascending=False)
top_persuadable = target_pool_sorted[["회원번호", "가격구간", "region_tier", "age_gender_segment", "uplift", "p_A_구독시예측", "p_B_비구독시예측"]].head(20)
print("\n[run01] Uplift 상위 20명 (비구독자 중 Persuadable 후보)")
print(top_persuadable.to_string(index=False))

ho[["회원번호", "가격구간", "treatment_h4", "uplift", "p_A_구독시예측", "p_B_비구독시예측"]].to_csv(
    BASE + "run01_uplift_scores_holdout.csv", index=False, encoding="utf-8-sig"
)
print("\n[run01] 저장 완료: run01_uplift_scores_holdout.csv")
