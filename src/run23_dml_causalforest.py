# -*- coding: utf-8 -*-
"""
run23: Double Machine Learning(EconML CausalForestDML) 기반 Uplift 재검증

우리 Treatment(구독여부)는 무작위 실험이 아니라 고객이 스스로 선택한 값이라(관측 데이터),
"구독한 사람이 원래 재구매 성향이 높은 사람이라서 재구매율이 높게 나온 것 아니냐"는
selection bias/confounding 우려가 있다. T-learner(run08b)·X-learner(run22)는 공변량을
충분히 통제하면 이 편향을 어느 정도 잡아내지만, 성향점수(propensity)와 결과모델을
직접 결합해서 편향을 상쇄(orthogonalize)하는 명시적 장치는 없다.

DML(Double/Debiased Machine Learning, Chernozhukov et al. 2018)은 이 문제를 겨냥한 기법으로,
  1) 결과모델 l(x) = E[Y|X]  (Y를 X로 예측)
  2) 처치모델 m(x) = E[T|X]  (T를 X로 예측, = propensity)
을 각각 머신러닝으로 학습해 잔차(residual) Y-l(x), T-m(x)를 만들고, 이 잔차끼리의 관계로
처치효과를 추정한다 — 두 모델 중 하나만 맞아도 편향이 줄어드는(doubly robust에 준하는)
성질이 있어 관측 데이터의 confounding에 T/X-learner보다 원리적으로 더 강건하다.

여기서는 개인별(이질적) 처치효과가 필요하므로 EconML의 CausalForestDML(Wager & Athey 2018의
Generalized Random Forest를 DML 잔차화와 결합한 모델)을 사용한다. RF 기반이라는 점에서
run08b(T-learner RF)·run22(X-learner, RF 결과모델)와 "같은 계열의 알고리즘, 다른 메타러너
구조"로 비교할 수 있다.
"""
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
OUT = BASE

tr = pd.read_csv(BASE + "snapshot_train.csv", encoding="utf-8-sig")
ho = pd.read_csv(BASE + "snapshot_holdout.csv", encoding="utf-8-sig")

for d in (tr, ho):
    d["log_frequency"] = np.log1p(d["frequency"])
    d["log_monetary"] = np.log1p(d["monetary"])
    d["log_recency"] = np.log1p(d["recency_days"])
    d["결혼_결측표시"] = d["결혼"].fillna("결측")

# run08b(T-learner RF)·run22(X-learner)와 동일 피처셋 -> 메타러너 구조 차이만 비교 가능하게
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
if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
y_train = train_known["y_repurchase_14d"].values.astype(float)
t_train = train_known["treatment_h4"].values.astype(int)

lines = []


def log(msg):
    print(msg)
    lines.append(str(msg))


log(f"[run23] 학습표본: 구독자(T=1) {sum(t_train == 1)}명 / 비구독자(T=0) {sum(t_train == 0)}명")
log("[run23] DML 잔차화 방식: model_y=RandomForestRegressor(Y를 X로 예측), "
    "model_t=RandomForestClassifier(T를 X로 예측=성향점수) 각각 5-fold cross-fitting")

# ================= CausalForestDML 학습 =================
# model_y/model_t는 run08b·run22와 동일 계열(RF)로 맞춤. CausalForestDML이 내부적으로
# cross-fitting(cv=5)을 수행해 residual(Y-l(x), T-m(x))을 만들고, 그 잔차로 개인별 처치효과를 추정.
est = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=400, min_samples_leaf=30, max_depth=7, random_state=42, n_jobs=-1),
    model_t=RandomForestClassifier(n_estimators=400, min_samples_leaf=30, max_depth=7, random_state=42, n_jobs=-1),
    discrete_treatment=True,
    cv=5,
    n_estimators=1000,
    min_samples_leaf=30,
    random_state=42,
    n_jobs=-1,
)
est.fit(Y=y_train, T=t_train, X=X_train, W=None)
log("[run23] CausalForestDML 학습 완료")

# ================= 홀드아웃 개인별 Uplift(CATE) 추정 =================
X_ho_all = pre.transform(ho[ALL_FEATS])
if hasattr(X_ho_all, "toarray"):
    X_ho_all = X_ho_all.toarray()

tau_dml = est.effect(X_ho_all)
lo_ci, hi_ci = est.effect_interval(X_ho_all, alpha=0.05)

ho2 = ho.copy()
ho2["uplift_dml"] = tau_dml
ho2["uplift_dml_ci_lo"] = lo_ci
ho2["uplift_dml_ci_hi"] = hi_ci

target = ho2[ho2["treatment_h4"] == 0]
vals = target["uplift_dml"].values

log(f"\n=== run23: CausalForestDML Uplift (비구독자 {len(vals):,}명 대상) ===")
log(f"평균 Uplift: {vals.mean() * 100:.3f}%p")
log(f"표준편차: {vals.std() * 100:.3f}%p, 최댓값: {vals.max() * 100:.2f}%p, 최솟값: {vals.min() * 100:.2f}%p")

ate_inf = est.ate_inference(X_ho_all[(ho2["treatment_h4"] == 0).values])
log(f"전체 평균처치효과(ATE, 비구독자 기준) 95%CI: [{ate_inf.conf_int_mean()[0] * 100:.2f}, "
    f"{ate_inf.conf_int_mean()[1] * 100:.2f}]%p (DML 고유의 점근적 신뢰구간 — T/X-learner의 부트스트랩 근사와 다른 방식)")

seg = target.groupby("가격구간", observed=True)["uplift_dml"].mean() * 100
log("\n가격구간별 평균 Uplift(%p):")
log(seg.to_string())

# ================= T-learner(run08b)·X-learner(run22)와 비교 =================
run08b = pd.read_csv(BASE + "run08b_uplift_forest_scores.csv", encoding="utf-8-sig")
run22 = pd.read_csv(BASE + "run22_xlearner_scores_holdout.csv", encoding="utf-8-sig")

merged = target[["회원번호", "uplift_dml"]].merge(
    run08b.loc[run08b["treatment_h4"] == 0, ["회원번호", "uplift_forest"]], on="회원번호"
).merge(
    run22[["회원번호", "uplift_xlearner"]], on="회원번호"
)

corr_t = merged["uplift_dml"].corr(merged["uplift_forest"], method="spearman")
corr_x = merged["uplift_dml"].corr(merged["uplift_xlearner"], method="spearman")
log(f"\n=== DML vs T-learner(RF, run08b) / X-learner(run22) 비교 ===")
log(f"DML vs T-learner 순위상관(Spearman): {corr_t:.3f}")
log(f"DML vs X-learner 순위상관(Spearman): {corr_x:.3f}")

n_top = int(len(merged) * 0.2)
dml_top = set(merged.nlargest(n_top, "uplift_dml")["회원번호"])
t_top = set(merged.nlargest(n_top, "uplift_forest")["회원번호"])
x_top = set(merged.nlargest(n_top, "uplift_xlearner")["회원번호"])
expected = n_top * n_top / len(merged)
log(f"상위 20% 겹침 — DML∩T-learner: {len(dml_top & t_top)}명 ({len(dml_top & t_top) / expected:.2f}배 무작위 기대치), "
    f"DML∩X-learner: {len(dml_top & x_top)}명 ({len(dml_top & x_top) / expected:.2f}배 무작위 기대치)")

all3_overlap = len(dml_top & t_top & x_top)
log(f"3개 메타러너(T/X/DML) 상위 20% 만장일치: {all3_overlap}명 (무작위 기대치 {n_top ** 3 / len(merged) ** 2:.0f}명, "
    f"{all3_overlap / (n_top ** 3 / len(merged) ** 2):.2f}배)")

ho2[["회원번호", "가격구간", "treatment_h4", "uplift_dml", "uplift_dml_ci_lo", "uplift_dml_ci_hi"]].to_csv(
    BASE + "run23_dml_scores_holdout.csv", index=False, encoding="utf-8-sig"
)
log("\n저장 완료: run23_dml_scores_holdout.csv")

with open(OUT + "run23_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
