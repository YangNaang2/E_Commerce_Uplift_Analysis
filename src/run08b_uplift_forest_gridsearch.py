# -*- coding: utf-8 -*-
"""
run08b: Uplift Forest 재튜닝판 — run08(수동 하이퍼파라미터 고정)을 run06/run07과 동일하게
GridSearchCV로 정식 튜닝. run01b(로지스틱)를 "대표모델"로 정할 때 비교 대상이던 run08이
튜닝되지 않은 상태였기 때문에, 백테스팅 착수 전 정식 튜닝판으로 재실행.
- 피처셋은 run06/run07과 동일(log_aov 제외 — log_monetary=log_frequency+log_aov 완전종속 문제)
- TwoModels 래퍼 대신 run07과 같은 방식으로 구독자/비구독자 RF를 직접 GridSearchCV로 튜닝
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
tr = pd.read_csv(BASE + "snapshot_train.csv", encoding="utf-8-sig")
ho = pd.read_csv(BASE + "snapshot_holdout.csv", encoding="utf-8-sig")

for d in (tr, ho):
    d["log_frequency"] = np.log1p(d["frequency"])
    d["log_monetary"] = np.log1p(d["monetary"])
    d["log_recency"] = np.log1p(d["recency_days"])  # log_aov는 완전종속 문제로 제외(run06과 동일)
    d["결혼_결측표시"] = d["결혼"].fillna("결측")

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

print(f"[run08b] 학습표본: 구독자 {sum(t_train==1)} / 비구독자 {sum(t_train==0)}")

# ================= RandomForest + GridSearchCV (run06/run07과 동일 방법론) =================
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": [10, 30, 50, 100],
}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Windows에서 GridSearchCV+RandomForest 동시 n_jobs=-1 중첩병렬화 시 TerminatedWorkerError 발생 -> RF는 n_jobs=1
gs_A = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1), param_grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
gs_A.fit(X_train[t_train == 1], y_train[t_train == 1])
gs_B = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1), param_grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
gs_B.fit(X_train[t_train == 0], y_train[t_train == 0])
print(f"[GridSearchCV] 구독자모델 최적: {gs_A.best_params_} (CV AUC={gs_A.best_score_:.3f})")
print(f"[GridSearchCV] 비구독자모델 최적: {gs_B.best_params_} (CV AUC={gs_B.best_score_:.3f})")
rf_A, rf_B = gs_A.best_estimator_, gs_B.best_estimator_

# ================= Holdout 적용 + Uplift 비교 =================
X_ho_all = pre.transform(ho[ALL_FEATS])
uplift_forest = rf_A.predict_proba(X_ho_all)[:, 1] - rf_B.predict_proba(X_ho_all)[:, 1]
ho = ho.copy()
ho["uplift_forest"] = uplift_forest

target = ho[ho["treatment_h4"] == 0]
vals = target["uplift_forest"].values
rng = np.random.default_rng(42)
boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])

print("\n=== run08b: Uplift Forest (RandomForest, GridSearchCV 튜닝) ===")
print(f"평균 Uplift: {vals.mean()*100:.3f}%p, 95%CI=[{lo*100:.2f}, {hi*100:.2f}]%p")
print(f"표준편차: {vals.std()*100:.3f}%p, 최댓값: {vals.max()*100:.2f}%p, 최솟값: {vals.min()*100:.2f}%p")

seg = target.groupby("가격구간", observed=True)["uplift_forest"].mean() * 100
print("\n가격구간별 평균 Uplift(%p):")
print(seg)

# --- RandomForest 피처 중요도 (두 arm 평균) ---
imp_t = pd.Series(rf_A.feature_importances_, index=feat_names)
imp_c = pd.Series(rf_B.feature_importances_, index=feat_names)
imp_avg = ((imp_t + imp_c) / 2).sort_values(ascending=False)
print("\n피처 중요도 상위 10 (구독자/비구독자 모델 평균):")
print(imp_avg.head(10))

# --- 해석용 서로게이트 트리 ---
X_ho_df = pd.DataFrame(X_ho_all.toarray() if hasattr(X_ho_all, "toarray") else X_ho_all, columns=feat_names)
surrogate = DecisionTreeRegressor(max_depth=3, min_samples_leaf=300, random_state=42)
surrogate.fit(X_ho_df, ho["uplift_forest"])
r2 = surrogate.score(X_ho_df, ho["uplift_forest"])
print(f"\n=== 해석용 서로게이트 트리 (Uplift 점수 근사, R^2={r2:.3f}) ===")
print(export_text(surrogate, feature_names=list(feat_names), max_depth=3))

ho[["회원번호", "가격구간", "treatment_h4", "uplift_forest"]].to_csv(
    BASE + "run08b_uplift_forest_scores.csv", index=False, encoding="utf-8-sig"
)
print("\n저장 완료: run08b_uplift_forest_scores.csv")
