# -*- coding: utf-8 -*-
"""
run08: Uplift Forest (sklift TwoModels + RandomForest) + 해석용 서로게이트 트리
- causalml 미설치(Windows 컴파일 이슈)로 sklift 기반 TwoModels(RandomForest)를 "Uplift Forest"로 사용
- 해석: 예측된 Uplift 점수를 얕은 회귀트리로 근사(surrogate tree) -> "왜 Persuadable인가" 규칙 추출
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from sklift.models import TwoModels

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
    "log_frequency", "log_monetary", "log_aov", "log_recency",
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

rf_trmnt = RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=30, random_state=42, n_jobs=-1)
rf_ctrl = RandomForestClassifier(n_estimators=400, max_depth=5, min_samples_leaf=30, random_state=42, n_jobs=-1)

tm = TwoModels(estimator_trmnt=rf_trmnt, estimator_ctrl=rf_ctrl, method="vanilla")
tm.fit(X_train, y_train, t_train)

X_ho_all = pre.transform(ho[ALL_FEATS])
uplift_forest = tm.predict(X_ho_all)
ho = ho.copy()
ho["uplift_forest"] = uplift_forest

target = ho[ho["treatment_h4"] == 0]
vals = target["uplift_forest"].values
rng = np.random.default_rng(42)
boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])

print("=== run08: Uplift Forest (TwoModels+RandomForest) ===")
print(f"평균 Uplift: {vals.mean()*100:.3f}%p, 95%CI=[{lo*100:.2f}, {hi*100:.2f}]%p")
print(f"표준편차: {vals.std()*100:.3f}%p, 최댓값: {vals.max()*100:.2f}%p, 최솟값: {vals.min()*100:.2f}%p")

seg = target.groupby("가격구간", observed=True)["uplift_forest"].mean() * 100
print("\n가격구간별 평균 Uplift(%p):")
print(seg)

# --- RandomForest 피처 중요도 (두 arm 평균) ---
imp_t = pd.Series(rf_trmnt.feature_importances_, index=feat_names)
imp_c = pd.Series(rf_ctrl.feature_importances_, index=feat_names)
imp_avg = ((imp_t + imp_c) / 2).sort_values(ascending=False)
print("\n피처 중요도 상위 10 (구독자/비구독자 모델 평균):")
print(imp_avg.head(10))

# --- 해석용 서로게이트 트리: Uplift 점수를 얕은 회귀트리로 근사 ---
X_ho_df = pd.DataFrame(X_ho_all.toarray() if hasattr(X_ho_all, "toarray") else X_ho_all, columns=feat_names)
surrogate = DecisionTreeRegressor(max_depth=3, min_samples_leaf=300, random_state=42)
surrogate.fit(X_ho_df, ho["uplift_forest"])
r2 = surrogate.score(X_ho_df, ho["uplift_forest"])
print(f"\n=== 해석용 서로게이트 트리 (Uplift 점수 근사, R^2={r2:.3f}) ===")
print(export_text(surrogate, feature_names=list(feat_names), max_depth=3))

ho[["회원번호", "가격구간", "treatment_h4", "uplift_forest"]].to_csv(BASE + "run08_uplift_forest_scores.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: run08_uplift_forest_scores.csv")
