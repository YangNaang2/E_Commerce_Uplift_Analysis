# -*- coding: utf-8 -*-
"""
B파트(H4 연령×성별) 개인별 Uplift 점수 재구성본
- Holdout 기준일을 10/26으로 재설정하면서 B의 기존 uplift_scores_partB_*.csv가 옛 홀드아웃
  기준이 되어 재산출이 필요해짐. 원본 파트B_H4_모델링_분석.ipynb에는 가설검증용 GLM 회귀만
  있고 개인별 L1/RF 스코어링 코드는 없어, A(run01b/run08b)와 동일한 kitchen-sink 방법론을
  적용해 재구성함(B 본인이 작성한 코드가 아니라 동일 방법론으로 재현한 버전).
- B의 시그니처: age_band_h4/gender_h4를 따로 두지 않고 age_gender_segment(결합 세그먼트)를
  범주형 피처로 사용 -> A와 피처 인코딩 자체가 달라져서 완전히 동일한 결과가 나오지 않음.
- 출력 포맷은 B가 이전에 보낸 CSV와 동일(회원번호, age_gender_segment, treatment_h4,
  p_control, p_treated, uplift) x 로지스틱/RF 각 파일.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
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

NUM_FEATS = [
    "log_frequency", "log_monetary", "log_aov", "log_recency",
    "regularity_cv", "reward_usage_rate", "나이",
    "preperiod_제철구매비율", "abnormal_account_flag", "reward_ever_used",
    "preperiod_명절선물세트구매여부",
]
# B 시그니처: age_band_h4/gender_h4 대신 결합 세그먼트를 통째로 투입
CAT_FEATS = ["age_gender_segment", "region_tier", "가격구간", "결혼_결측표시"]
ALL_FEATS = NUM_FEATS + CAT_FEATS

pre = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
])

train_known = tr[tr["include_in_uplift_model"] == 1].copy()
X_train = pre.fit_transform(train_known[ALL_FEATS])
y_train = train_known["y_repurchase_14d"].values
t_train = train_known["treatment_h4"].values
print(f"[partB 재현] 학습표본: 구독자 {sum(t_train==1)} / 비구독자 {sum(t_train==0)}")

# ---- 로지스틱 L1 (A와 동일 방법론) ----
Cs = np.logspace(-3, 2, 15)
logit_A = LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="liblinear", max_iter=5000, scoring="roc_auc")
logit_A.fit(X_train[t_train == 1], y_train[t_train == 1])
logit_B = LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="liblinear", max_iter=5000, scoring="roc_auc")
logit_B.fit(X_train[t_train == 0], y_train[t_train == 0])
print(f"[partB 재현][로지스틱] 구독자모델 C={logit_A.C_[0]:.4f}, 비구독자모델 C={logit_B.C_[0]:.4f}")

# ---- RandomForest GridSearchCV (A와 동일 그리드, random_state만 다르게) ----
param_grid = {"n_estimators": [200, 400], "max_depth": [3, 5, 7, None], "min_samples_leaf": [10, 30, 50, 100]}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
gs_A = GridSearchCV(RandomForestClassifier(random_state=43, n_jobs=1), param_grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
gs_A.fit(X_train[t_train == 1], y_train[t_train == 1])
gs_B = GridSearchCV(RandomForestClassifier(random_state=43, n_jobs=1), param_grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
gs_B.fit(X_train[t_train == 0], y_train[t_train == 0])
print(f"[partB 재현][RF] 구독자모델 최적: {gs_A.best_params_} (CV AUC={gs_A.best_score_:.3f})")
print(f"[partB 재현][RF] 비구독자모델 최적: {gs_B.best_params_} (CV AUC={gs_B.best_score_:.3f})")
rf_A, rf_B = gs_A.best_estimator_, gs_B.best_estimator_

# ---- Holdout 전체에 적용 ----
X_ho_all = pre.transform(ho[ALL_FEATS])
out = ho[["회원번호", "age_gender_segment", "treatment_h4"]].copy()

p_c_logit = logit_B.predict_proba(X_ho_all)[:, 1]
p_t_logit = logit_A.predict_proba(X_ho_all)[:, 1]
out_logit = out.copy()
out_logit["p_control"] = p_c_logit
out_logit["p_treated"] = p_t_logit
out_logit["uplift"] = p_t_logit - p_c_logit
out_logit.to_csv(BASE + "uplift_scores_partB_logit.csv", index=False, encoding="utf-8-sig")

p_c_rf = rf_B.predict_proba(X_ho_all)[:, 1]
p_t_rf = rf_A.predict_proba(X_ho_all)[:, 1]
out_rf = out.copy()
out_rf["p_control"] = p_c_rf
out_rf["p_treated"] = p_t_rf
out_rf["uplift"] = p_t_rf - p_c_rf
out_rf.to_csv(BASE + "uplift_scores_partB_rf.csv", index=False, encoding="utf-8-sig")

target = ho[ho["treatment_h4"] == 0]
print(f"\n[partB 재현] 로지스틱 평균 Uplift(비구독자): {out_logit.loc[out_logit.treatment_h4==0,'uplift'].mean()*100:.3f}%p")
print(f"[partB 재현] RF 평균 Uplift(비구독자): {out_rf.loc[out_rf.treatment_h4==0,'uplift'].mean()*100:.3f}%p")
print("\n저장 완료: uplift_scores_partB_logit.csv, uplift_scores_partB_rf.csv")
