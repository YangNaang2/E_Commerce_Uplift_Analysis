# -*- coding: utf-8 -*-
"""
[Claude 재현본 — D 본인 코드 아님] D파트(H2 지역×배송) 개인별 Uplift 점수 재생성 (v2)
- 배경: Holdout 재설정(11/2->10/26, 8/24)으로 옛 uplift_scores_partD.csv가 낡음. D의 실제
  재검증 코드는 못 받아서 A와 동일 kitchen-sink 방법론으로 재현.
- v1(is_dawn_delivery_zone 추가)은 그 값이 region_tier에 완전히 종속(1:1)된 파생값이라
  A와 로지스틱 결과가 사실상 동일(상관 1.000)하게 나오는 문제 발견 -> region_key(224종,
  구독자 1,626명 대비 카디널리티 과다)도 기각 -> C의 force-retain 2단계 방식(핵심변수
  region_tier를 뺀 나머지로 L1 자동축소 -> 선택된 변수+region_tier로 무정규화 재적합)을
  D에도 적용해 진짜 차별화되는 재현본으로 교체.
- 출력 포맷은 기존 h2_파트D_회원별_uplift예측_logit_rf.csv(=uplift_scores_partD.csv)와 동일.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

KEY_CAT = "region_tier"
NUM_FEATS = [
    "log_frequency", "log_monetary", "log_aov", "log_recency",
    "regularity_cv", "reward_usage_rate", "나이",
    "preperiod_제철구매비율", "abnormal_account_flag", "reward_ever_used",
    "preperiod_명절선물세트구매여부",
]
CAT_REST = ["age_band_h4", "gender_h4", "가격구간", "결혼_결측표시"]
ALL_REST = NUM_FEATS + CAT_REST

train_known = tr[tr["include_in_uplift_model"] == 1].copy()
t_train = train_known["treatment_h4"].values
y_train = train_known["y_repurchase_14d"].values

# ---- Step1: 핵심변수(region_tier) 뺀 나머지로 L1 자동축소 ----
pre_rest = ColumnTransformer([("num", StandardScaler(), NUM_FEATS), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_REST)])
X_rest = pre_rest.fit_transform(train_known[ALL_REST])
feat_names_rest = pre_rest.get_feature_names_out()
Cs = np.logspace(-3, 2, 15)
tmp_A = LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="liblinear", max_iter=5000, scoring="roc_auc").fit(X_rest[t_train == 1], y_train[t_train == 1])
tmp_B = LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="liblinear", max_iter=5000, scoring="roc_auc").fit(X_rest[t_train == 0], y_train[t_train == 0])
survived_cols = set(f for f, c in zip(feat_names_rest, tmp_A.coef_[0]) if abs(c) > 1e-6) | set(f for f, c in zip(feat_names_rest, tmp_B.coef_[0]) if abs(c) > 1e-6)


def parent_feature(colname):
    for f in NUM_FEATS:
        if colname == f"num__{f}":
            return f
    for f in CAT_REST:
        if colname.startswith(f"cat__{f}_"):
            return f
    return None


survived_features = sorted({parent_feature(c) for c in survived_cols if parent_feature(c)})
print(f"[partD 재현v2][Step1] 핵심변수 제외 L1으로 살아남은 원본피처: {survived_features}")

NUM_FINAL = [f for f in survived_features if f in NUM_FEATS]
CAT_FINAL = [f for f in survived_features if f in CAT_REST] + [KEY_CAT]
print(f"[partD 재현v2][Step2] 최종 투입: 수치형={NUM_FINAL}, 범주형={CAT_FINAL}")

if NUM_FINAL:
    pre_final = ColumnTransformer([("num", StandardScaler(), NUM_FINAL), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FINAL)])
else:
    pre_final = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FINAL)])
ALL_FINAL = NUM_FINAL + CAT_FINAL
X_train_final = pre_final.fit_transform(train_known[ALL_FINAL])

logit_A = LogisticRegression(penalty=None, max_iter=5000).fit(X_train_final[t_train == 1], y_train[t_train == 1])
logit_B = LogisticRegression(penalty=None, max_iter=5000).fit(X_train_final[t_train == 0], y_train[t_train == 0])
print("[partD 재현v2][Step2] 무정규화 재적합 완료")

# ---- RandomForest GridSearchCV: 전체 kitchen-sink(region_tier 포함) 그대로, random_state만 다르게 ----
CAT_RF = CAT_REST + [KEY_CAT]
ALL_RF = NUM_FEATS + CAT_RF
pre_rf = ColumnTransformer([("num", StandardScaler(), NUM_FEATS), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_RF)])
X_train_rf = pre_rf.fit_transform(train_known[ALL_RF])
param_grid = {"n_estimators": [200, 400], "max_depth": [3, 5, 7, None], "min_samples_leaf": [10, 30, 50, 100]}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=45)
gs_A = GridSearchCV(RandomForestClassifier(random_state=45, n_jobs=1), param_grid, cv=cv5, scoring="roc_auc", n_jobs=-1).fit(X_train_rf[t_train == 1], y_train[t_train == 1])
gs_B = GridSearchCV(RandomForestClassifier(random_state=45, n_jobs=1), param_grid, cv=cv5, scoring="roc_auc", n_jobs=-1).fit(X_train_rf[t_train == 0], y_train[t_train == 0])
print(f"[partD 재현v2][RF] 구독자모델 최적: {gs_A.best_params_}")
print(f"[partD 재현v2][RF] 비구독자모델 최적: {gs_B.best_params_}")
rf_A, rf_B = gs_A.best_estimator_, gs_B.best_estimator_

# ---- Holdout 전체 적용 ----
X_ho_final = pre_final.transform(ho[ALL_FINAL])
X_ho_rf = pre_rf.transform(ho[ALL_RF])

out = ho[["회원번호", "region_tier", "treatment_h4"]].copy()
p_c_logit = logit_B.predict_proba(X_ho_final)[:, 1]
p_t_logit = logit_A.predict_proba(X_ho_final)[:, 1]
p_c_rf = rf_B.predict_proba(X_ho_rf)[:, 1]
p_t_rf = rf_A.predict_proba(X_ho_rf)[:, 1]

out["p_control_logit"] = p_c_logit
out["p_treated_logit"] = p_t_logit
out["uplift_logit"] = p_t_logit - p_c_logit
out["p_control_rf"] = p_c_rf
out["p_treated_rf"] = p_t_rf
out["uplift_rf"] = p_t_rf - p_c_rf
out.to_csv(BASE + "uplift_scores_partD.csv", index=False, encoding="utf-8-sig")

target = out[out.treatment_h4 == 0]
print(f"\n[partD 재현v2] 로지스틱 평균 Uplift(비구독자): {target['uplift_logit'].mean()*100:.3f}%p")
print(f"[partD 재현v2] RF 평균 Uplift(비구독자): {target['uplift_rf'].mean()*100:.3f}%p")
print("\n저장 완료: uplift_scores_partD.csv")
