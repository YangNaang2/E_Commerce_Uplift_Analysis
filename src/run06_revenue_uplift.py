# -*- coding: utf-8 -*-
"""
run06: 매출 관점 Uplift — Y=y_revenue_14d(14일 내 재구매 금액, 원)
- kitchen-sink 피처(run01b/run07과 동일 철학) + 이번엔 RandomForest도 정식 그리드서치로 튜닝
- 선형: ElasticNetCV(alpha, l1_ratio 교차검증) / 트리: RandomForestRegressor + GridSearchCV
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
tr = pd.read_csv(BASE + "snapshot_train.csv", encoding="utf-8-sig")
ho = pd.read_csv(BASE + "snapshot_holdout.csv", encoding="utf-8-sig")

for d in (tr, ho):
    d["log_frequency"] = np.log1p(d["frequency"])
    d["log_monetary"] = np.log1p(d["monetary"])
    d["log_aov"] = np.log1p(d["aov"])
    d["log_recency"] = np.log1p(d["recency_days"])
    d["결혼_결측표시"] = d["결혼"].fillna("결측")

NUM_FEATS = [
    # log_aov 제외: monetary = frequency * aov 이므로 log_monetary = log_frequency + log_aov (완전 종속, 8/24 발견)
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
y_train = train_known["y_revenue_14d"].values
t_train = train_known["treatment_h4"].values

print(f"[run06] 학습표본: 구독자 {sum(t_train==1)} / 비구독자 {sum(t_train==0)}")
print(f"[run06] Y(매출) 분포 - 평균 {y_train.mean():.0f}원, 중앙값 {np.median(y_train):.0f}원, 0인 비율 {(y_train==0).mean()*100:.1f}%")

# ================= 선형: ElasticNetCV =================
enet_A = ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, 1.0], alphas=np.logspace(-1, 4, 20), cv=5, max_iter=20000)
enet_A.fit(X_train[t_train == 1], y_train[t_train == 1])
enet_B = ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, 1.0], alphas=np.logspace(-1, 4, 20), cv=5, max_iter=20000)
enet_B.fit(X_train[t_train == 0], y_train[t_train == 0])

print(f"\n[ElasticNetCV] 구독자모델 alpha={enet_A.alpha_:.2f}, l1_ratio={enet_A.l1_ratio_:.2f}")
print(f"[ElasticNetCV] 비구독자모델 alpha={enet_B.alpha_:.2f}, l1_ratio={enet_B.l1_ratio_:.2f}")
kept_A = [(f, c) for f, c in zip(feat_names, enet_A.coef_) if abs(c) > 1e-6]
kept_B = [(f, c) for f, c in zip(feat_names, enet_B.coef_) if abs(c) > 1e-6]
print(f"[ElasticNetCV] 구독자모델 생존피처 {len(kept_A)}/{len(feat_names)}: {sorted(kept_A, key=lambda x:-abs(x[1]))[:8]}")
print(f"[ElasticNetCV] 비구독자모델 생존피처 {len(kept_B)}/{len(feat_names)}: {sorted(kept_B, key=lambda x:-abs(x[1]))[:8]}")

# ================= 트리: RandomForestRegressor + GridSearchCV =================
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": [10, 30, 50, 100],
}
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

# Windows에서 GridSearchCV와 RandomForest를 동시에 n_jobs=-1로 중첩 병렬화하면
# joblib 워커가 죽는 문제(TerminatedWorkerError) 발생 -> 바깥(GridSearchCV)만 병렬화
gs_A = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=1), param_grid, cv=cv5, scoring="neg_mean_squared_error", n_jobs=-1)
gs_A.fit(X_train[t_train == 1], y_train[t_train == 1])
gs_B = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=1), param_grid, cv=cv5, scoring="neg_mean_squared_error", n_jobs=-1)
gs_B.fit(X_train[t_train == 0], y_train[t_train == 0])

print(f"\n[GridSearchCV] 구독자모델 최적 파라미터: {gs_A.best_params_} (CV RMSE={np.sqrt(-gs_A.best_score_):.0f}원)")
print(f"[GridSearchCV] 비구독자모델 최적 파라미터: {gs_B.best_params_} (CV RMSE={np.sqrt(-gs_B.best_score_):.0f}원)")

rf_A, rf_B = gs_A.best_estimator_, gs_B.best_estimator_

# ================= Holdout 적용 + Uplift 비교 =================
X_ho_all = pre.transform(ho[ALL_FEATS])

uplift_enet = enet_A.predict(X_ho_all) - enet_B.predict(X_ho_all)
uplift_rf = rf_A.predict(X_ho_all) - rf_B.predict(X_ho_all)

ho = ho.copy()
ho["uplift_revenue_enet"] = uplift_enet
ho["uplift_revenue_rf"] = uplift_rf

target = ho[ho["treatment_h4"] == 0]
rng = np.random.default_rng(42)


def boot_ci(vals, n=2000):
    b = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


for col, label in [("uplift_revenue_enet", "ElasticNet(선형)"), ("uplift_revenue_rf", "RandomForest(그리드서치)")]:
    vals = target[col].values
    lo, hi = boot_ci(vals)
    print(f"\n=== {label} 매출 Uplift ===")
    print(f"평균: {vals.mean():.0f}원/14일, 95%CI=[{lo:.0f}, {hi:.0f}]원, 0포함여부: {'포함(비유의)' if lo<0<hi else '0 미포함(유의)'}")
    seg = target.groupby("가격구간", observed=True)[col].mean()
    print("가격구간별 평균 매출Uplift(원):")
    print(seg)

ho[["회원번호", "가격구간", "treatment_h4", "uplift_revenue_enet", "uplift_revenue_rf"]].to_csv(
    BASE + "run06_revenue_uplift_scores.csv", index=False, encoding="utf-8-sig"
)
print("\n저장 완료: run06_revenue_uplift_scores.csv")
