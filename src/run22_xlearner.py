# -*- coding: utf-8 -*-
"""
run22: X-learner 구현 — T-learner(run08b)와 다른 메타러너 구조로 재검증

X-learner (Kunzel et al. 2019) 4단계:
  1) 처치군(구독자)/대조군(비구독자) 각각의 결과(Y) 예측모델 mu1, mu0 학습 (T-learner와 동일한 stage)
  2) 처치 성향점수 모델 g(x) = P(T=1|X) 학습
  3) 개인별 처치효과를 "대체(impute)"함:
       - 처치군(구독자) 각자: D1 = 실제Y - mu0(X)  [반사실: 비구독이었다면 어땠을지와 비교]
       - 대조군(비구독자) 각자: D0 = mu1(X) - 실제Y  [반사실: 구독이었다면 어땠을지와 비교]
  4) D1을 처치군 데이터로, D0를 대조군 데이터로 각각 회귀해 tau1(x), tau0(x)를 학습하고,
     성향점수로 가중평균: tau_X(x) = (1-g(x))*tau0(x) + g(x)*tau1(x)
     (대조군이 더 크면 tau0가 더 많은 표본으로 학습돼 분산이 작음 -> g(x)가 작을 때(=대조군에
     가까운 영역) tau0에 더 큰 가중치를 주는 게 이론적으로 맞음. 우리 데이터는 비구독자
     8,276명 > 구독자 3,456명이라 X-learner가 원래 유리하다고 알려진 불균형 상황에 해당)
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
OUT = BASE

tr = pd.read_csv(BASE + "snapshot_train.csv", encoding="utf-8-sig")
ho = pd.read_csv(BASE + "snapshot_holdout.csv", encoding="utf-8-sig")

for d in (tr, ho):
    d["log_frequency"] = np.log1p(d["frequency"])
    d["log_monetary"] = np.log1p(d["monetary"])
    d["log_recency"] = np.log1p(d["recency_days"])  # run08b와 동일 (log_aov는 완전종속 문제로 제외)
    d["결혼_결측표시"] = d["결혼"].fillna("결측")

# run08b(T-learner RF)와 동일 피처셋 -> 메타러너 구조 차이만 비교 가능하게
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
y_train = train_known["y_repurchase_14d"].values
t_train = train_known["treatment_h4"].values

lines = []


def log(msg):
    print(msg)
    lines.append(str(msg))


log(f"[run22] 학습표본: 구독자(T=1) {sum(t_train == 1)}명 / 비구독자(T=0) {sum(t_train == 0)}명"
    f" (비구독자가 {sum(t_train == 0) / sum(t_train == 1):.2f}배 더 큼 -> X-learner가 유리하다는 불균형 상황)")

# ================= Stage 1: 결과예측모델 mu1(구독자)/mu0(비구독자) — run08b와 동일 튜닝 =================
param_grid = {"n_estimators": [200, 400], "max_depth": [3, 5, 7, None], "min_samples_leaf": [10, 30, 50, 100]}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs1 = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1), param_grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
gs1.fit(X_train[t_train == 1], y_train[t_train == 1])
mu1 = gs1.best_estimator_

gs0 = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1), param_grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
gs0.fit(X_train[t_train == 0], y_train[t_train == 0])
mu0 = gs0.best_estimator_

log(f"[Stage1] mu1(구독자 결과모델) 최적: {gs1.best_params_} (CV AUC={gs1.best_score_:.3f})")
log(f"[Stage1] mu0(비구독자 결과모델) 최적: {gs0.best_params_} (CV AUC={gs0.best_score_:.3f})")

# ================= Stage 2: 성향점수 모델 g(x) = P(T=1|X) =================
prop_model = LogisticRegression(max_iter=2000)
prop_model.fit(X_train, t_train)
g_train = prop_model.predict_proba(X_train)[:, 1]
log(f"[Stage2] 성향점수 모델 학습완료. 평균 g(x)={g_train.mean():.3f} (실제 구독비율 {t_train.mean():.3f}과 비교)")

# ================= Stage 3: 개인별 처치효과 대체(impute) =================
X_treated, y_treated = X_train[t_train == 1], y_train[t_train == 1]
D1 = y_treated - mu0.predict_proba(X_treated)[:, 1]

X_control, y_control = X_train[t_train == 0], y_train[t_train == 0]
D0 = mu1.predict_proba(X_control)[:, 1] - y_control

log(f"[Stage3] D1(처치군 대체효과) 평균={D1.mean() * 100:.2f}%p, D0(대조군 대체효과) 평균={D0.mean() * 100:.2f}%p")

# ================= Stage 4: tau1(x), tau0(x) 회귀 + 성향점수 가중결합 =================
tau1_model = RandomForestRegressor(n_estimators=400, min_samples_leaf=30, random_state=42, n_jobs=-1)
tau1_model.fit(X_treated, D1)

tau0_model = RandomForestRegressor(n_estimators=400, min_samples_leaf=30, random_state=42, n_jobs=-1)
tau0_model.fit(X_control, D0)

X_ho_all = pre.transform(ho[ALL_FEATS])
g_hat = prop_model.predict_proba(X_ho_all)[:, 1]
tau1_hat = tau1_model.predict(X_ho_all)
tau0_hat = tau0_model.predict(X_ho_all)
tau_x = (1 - g_hat) * tau0_hat + g_hat * tau1_hat

ho2 = ho.copy()
ho2["propensity"] = g_hat
ho2["uplift_xlearner"] = tau_x

target = ho2[ho2["treatment_h4"] == 0]
vals = target["uplift_xlearner"].values
rng = np.random.default_rng(42)
boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])

log(f"\n=== [Stage4] run22: X-learner Uplift (비구독자 {len(vals):,}명 대상) ===")
log(f"평균 Uplift: {vals.mean() * 100:.3f}%p, 95%CI=[{lo * 100:.2f}, {hi * 100:.2f}]%p")
log(f"표준편차: {vals.std() * 100:.3f}%p, 최댓값: {vals.max() * 100:.2f}%p, 최솟값: {vals.min() * 100:.2f}%p")
log(f"비구독자 평균 성향점수(구독확률 추정): {g_hat[(ho2['treatment_h4'] == 0).values].mean():.3f}")

seg = target.groupby("가격구간", observed=True)["uplift_xlearner"].mean() * 100
log("\n가격구간별 평균 Uplift(%p):")
log(seg.to_string())

# ================= T-learner(run08b, 같은 RF 기반)와 비교 =================
run08b = pd.read_csv(BASE + "run08b_uplift_forest_scores.csv", encoding="utf-8-sig")
merged = target[["회원번호", "uplift_xlearner"]].merge(
    run08b.loc[run08b["treatment_h4"] == 0, ["회원번호", "uplift_forest"]], on="회원번호"
)
corr = merged["uplift_xlearner"].corr(merged["uplift_forest"], method="spearman")
log(f"\n=== X-learner vs T-learner(RF, run08b) 비교 ===")
log(f"두 Uplift 점수 간 순위상관(Spearman): {corr:.3f}")

n_top = int(len(merged) * 0.2)
x_top = set(merged.nlargest(n_top, "uplift_xlearner")["회원번호"])
t_top = set(merged.nlargest(n_top, "uplift_forest")["회원번호"])
overlap = len(x_top & t_top)
expected = n_top * n_top / len(merged)
log(f"상위 20% 겹침: {overlap}명 / 각 상위20%={n_top}명 (무작위 기대치 {expected:.0f}명, {overlap / expected:.2f}배)")

ho2[["회원번호", "가격구간", "treatment_h4", "propensity", "uplift_xlearner"]].to_csv(
    BASE + "run22_xlearner_scores_holdout.csv", index=False, encoding="utf-8-sig"
)
log("\n저장 완료: run22_xlearner_scores_holdout.csv")

with open(OUT + "run22_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
