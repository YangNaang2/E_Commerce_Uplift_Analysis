# -*- coding: utf-8 -*-
"""
run07: 이탈 방지 관점 보조 타겟 (Y=y_repurchase_14d, X를 recency_days 기준 위험군으로 슬라이스)
- 목표 슬라이드 기준 그대로: Active(<30일) / 관심필요(30~89일) / 이탈위험(90일+)
- "이탈 위험군일수록 구독의 재구매 Uplift가 더 큰가"를 검증 -> 맞으면 "복귀 프로모션으로 구독을 미는" 전략 근거
- kitchen-sink 피처 + LogisticRegressionCV(L1) + RandomForestClassifier(GridSearchCV, 정식 튜닝) 둘 다 사용
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
tr = pd.read_csv(BASE + "snapshot_train.csv", encoding="utf-8-sig")
ho = pd.read_csv(BASE + "snapshot_holdout.csv", encoding="utf-8-sig")

for d in (tr, ho):
    d["log_frequency"] = np.log1p(d["frequency"])
    d["log_monetary"] = np.log1p(d["monetary"])
    d["log_recency"] = np.log1p(d["recency_days"])  # log_aov는 8/24 발견한 완전종속 문제로 제외
    d["결혼_결측표시"] = d["결혼"].fillna("결측")

    def risk_band(days):
        if days < 30:
            return "Active(<30일)"
        if days < 90:
            return "관심필요(30~89일)"
        return "이탈위험(90일+)"

    d["이탈위험군"] = d["recency_days"].apply(risk_band)

NUM_FEATS = [
    "log_frequency", "log_monetary", "log_recency",
    "regularity_cv", "reward_usage_rate", "나이",
    "preperiod_제철구매비율", "abnormal_account_flag", "reward_ever_used",
    "preperiod_명절선물세트구매여부",
]
CAT_FEATS = ["age_band_h4", "gender_h4", "region_tier", "가격구간", "결혼_결측표시"]
ALL_FEATS = NUM_FEATS + CAT_FEATS

pre = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
])

train_known = tr[tr["include_in_uplift_model"] == 1].copy()
X_train = pre.fit_transform(train_known[ALL_FEATS])
feat_names = pre.get_feature_names_out()
y_train = train_known["y_repurchase_14d"].values
t_train = train_known["treatment_h4"].values

print(f"[run07] 학습표본: 구독자 {sum(t_train==1)} / 비구독자 {sum(t_train==0)}")
print("[run07] Train 이탈위험군 분포:")
print(train_known["이탈위험군"].value_counts())

# ================= 로지스틱 L1 (kitchen-sink, run01b와 동일 방법론) =================
Cs = np.logspace(-3, 2, 15)
logit_A = LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="liblinear", max_iter=5000, scoring="roc_auc")
logit_A.fit(X_train[t_train == 1], y_train[t_train == 1])
logit_B = LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="liblinear", max_iter=5000, scoring="roc_auc")
logit_B.fit(X_train[t_train == 0], y_train[t_train == 0])
print(f"\n[LogisticRegressionCV] 구독자모델 C={logit_A.C_[0]:.4f}, 비구독자모델 C={logit_B.C_[0]:.4f}")

# ================= RandomForest + GridSearchCV (run06과 동일 방법론) =================
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": [10, 30, 50, 100],
}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_A = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1), param_grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
gs_A.fit(X_train[t_train == 1], y_train[t_train == 1])
gs_B = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1), param_grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
gs_B.fit(X_train[t_train == 0], y_train[t_train == 0])
print(f"[GridSearchCV] 구독자모델 최적: {gs_A.best_params_} (CV AUC={gs_A.best_score_:.3f})")
print(f"[GridSearchCV] 비구독자모델 최적: {gs_B.best_params_} (CV AUC={gs_B.best_score_:.3f})")
rf_A, rf_B = gs_A.best_estimator_, gs_B.best_estimator_

# ================= Holdout 평가 =================
ho_known = ho[ho["include_in_uplift_model"] == 1].copy()
X_ho_known = pre.transform(ho_known[ALL_FEATS])
sub_mask = ho_known["treatment_h4"] == 1
non_mask = ho_known["treatment_h4"] == 0

for name, mA, mB in [("로지스틱L1", logit_A, logit_B), ("RandomForest", rf_A, rf_B)]:
    auc_A = roc_auc_score(ho_known.loc[sub_mask, "y_repurchase_14d"], mA.predict_proba(X_ho_known[sub_mask.values])[:, 1])
    auc_B = roc_auc_score(ho_known.loc[non_mask, "y_repurchase_14d"], mB.predict_proba(X_ho_known[non_mask.values])[:, 1])
    print(f"[{name}] Holdout AUC - 구독자모델: {auc_A:.3f} / 비구독자모델: {auc_B:.3f}")

X_ho_all = pre.transform(ho[ALL_FEATS])
ho = ho.copy()
ho["uplift_logit"] = logit_A.predict_proba(X_ho_all)[:, 1] - logit_B.predict_proba(X_ho_all)[:, 1]
ho["uplift_rf"] = rf_A.predict_proba(X_ho_all)[:, 1] - rf_B.predict_proba(X_ho_all)[:, 1]

target = ho[ho["treatment_h4"] == 0]  # 비구독자(실제 복귀프로모션 타겟 후보)
rng = np.random.default_rng(42)


def boot_ci(vals, n=2000):
    b = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


order = ["Active(<30일)", "관심필요(30~89일)", "이탈위험(90일+)"]
print("\n=== 이탈위험군별 재구매 Uplift (H7: 이탈위험군일수록 Uplift가 큰가) ===")
for col, label in [("uplift_logit", "로지스틱L1"), ("uplift_rf", "RandomForest")]:
    print(f"\n--- {label} ---")
    for seg in order:
        vals = target.loc[target["이탈위험군"] == seg, col].values
        lo, hi = boot_ci(vals)
        sig = "0 미포함(유의)" if lo > 0 or hi < 0 else "포함(비유의)"
        print(f"  {seg}: n={len(vals)}, 평균Uplift={vals.mean()*100:.2f}%p, 95%CI=[{lo*100:.2f}, {hi*100:.2f}]%p, {sig}")

ho[["회원번호", "이탈위험군", "가격구간", "treatment_h4", "uplift_logit", "uplift_rf"]].to_csv(
    BASE + "run07_churn_uplift_scores.csv", index=False, encoding="utf-8-sig"
)
print("\n저장 완료: run07_churn_uplift_scores.csv")
