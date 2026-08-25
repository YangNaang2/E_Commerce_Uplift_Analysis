# -*- coding: utf-8 -*-
"""
v4: '다 넣고 L1 정규화로 자동 축소' 방식의 T-learner
- 고카디널리티(region_key, 200+ 범주)만 제외하고 나머지 컬럼은 전부 투입
- 각 arm(구독자/비구독자)마다 LogisticRegressionCV(L1)로 정규화 강도(C)를 교차검증으로 자동 선택
- 결과: 0으로 수렴한 피처(=자동으로 걸러진 피처)와 살아남은 피처를 함께 리포트
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV
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
    d["결혼_결측표시"] = d["결혼"].fillna("결측")

# region_key(고카디널리티) 제외하고 스냅샷에 있는 사실상 모든 후보 컬럼을 투입
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

print(f"[run01b] 투입 피처 수(인코딩 후): {X_train.shape[1]}개 (원본 컬럼 {len(ALL_FEATS)}개)")
print(f"[run01b] 학습표본: 구독자 {sum(t_train==1)} / 비구독자 {sum(t_train==0)}")

Cs = np.logspace(-3, 2, 15)

model_A = LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="liblinear", max_iter=5000, scoring="roc_auc")
model_A.fit(X_train[t_train == 1], y_train[t_train == 1])

model_B = LogisticRegressionCV(Cs=Cs, cv=5, penalty="l1", solver="liblinear", max_iter=5000, scoring="roc_auc")
model_B.fit(X_train[t_train == 0], y_train[t_train == 0])

print(f"\n[run01b] 구독자모델(A) 선택된 C(정규화강도, 클수록 약한 규제): {model_A.C_[0]:.4f}")
print(f"[run01b] 비구독자모델(B) 선택된 C: {model_B.C_[0]:.4f}")


def report_coefs(model, name):
    coefs = model.coef_[0]
    kept = [(f, c) for f, c in zip(feat_names, coefs) if abs(c) > 1e-6]
    dropped = [f for f, c in zip(feat_names, coefs) if abs(c) <= 1e-6]
    print(f"\n[{name}] 살아남은 피처 {len(kept)}/{len(feat_names)}개 (0으로 수렴 = 자동 제외 {len(dropped)}개)")
    for f, c in sorted(kept, key=lambda x: -abs(x[1])):
        print(f"   {f}: {c:+.3f}")
    if dropped:
        print(f"   [자동 제외됨] {dropped}")


report_coefs(model_A, "구독자모델(A)")
report_coefs(model_B, "비구독자모델(B)")

# --- Holdout 평가 ---
ho_known = ho[ho["include_in_uplift_model"] == 1].copy()
X_ho_known = pre.transform(ho_known[ALL_FEATS])
sub_mask = ho_known["treatment_h4"] == 1
non_mask = ho_known["treatment_h4"] == 0
auc_A = roc_auc_score(ho_known.loc[sub_mask, "y_repurchase_14d"], model_A.predict_proba(X_ho_known[sub_mask.values])[:, 1])
auc_B = roc_auc_score(ho_known.loc[non_mask, "y_repurchase_14d"], model_B.predict_proba(X_ho_known[non_mask.values])[:, 1])
print(f"\n[run01b] Holdout AUC - 구독자모델(A): {auc_A:.3f} / 비구독자모델(B): {auc_B:.3f}")

X_ho_all = pre.transform(ho[ALL_FEATS])
uplift = model_A.predict_proba(X_ho_all)[:, 1] - model_B.predict_proba(X_ho_all)[:, 1]
ho_tmp = ho.copy()
ho_tmp["uplift"] = uplift
target = ho_tmp[ho_tmp["treatment_h4"] == 0]

rng = np.random.default_rng(42)
vals = target["uplift"].values
boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\n[run01b] 비구독자 전체 평균 Uplift: {vals.mean()*100:.3f}%p, 95%CI=[{lo*100:.2f}, {hi*100:.2f}]%p")
print(f"[run01b] Uplift 표준편차: {vals.std()*100:.3f}%p, 최댓값: {vals.max()*100:.2f}%p")

seg = target.groupby("가격구간", observed=True)["uplift"].mean() * 100
print("\n[run01b] 가격구간별 평균 Uplift(%p):")
print(seg)

ho_tmp[["회원번호", "가격구간", "treatment_h4", "uplift"]].to_csv(
    BASE + "run01b_uplift_scores_holdout.csv", index=False, encoding="utf-8-sig"
)
print("\n저장 완료: run01b_uplift_scores_holdout.csv")
