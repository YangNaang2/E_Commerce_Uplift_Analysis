# -*- coding: utf-8 -*-
"""
Model 2: VIP 전환 예측 모델
- 교수님 피드백 이후 추가한 분석. Uplift(H1~H4, 쿠폰/구독 반응)와는 별개로,
  "지금은 VIP가 아니지만 앞으로 VIP가 될 가능성이 높은 고객"을 pre-period 신호만으로
  미리 찾아내는 이진분류 모델. 프로젝트 목표2(구매트렌드 변화 반영한 차별화 서비스)의
  업셀/성장고객 타겟팅 축에 대응한다.

설계:
- pre-period = Train 기준일(2025-09-16) 이전 누적 이력 -> snapshot_train.csv 재사용
  (frequency/monetary/aov/recency_days/regularity_cv 등 이미 이 시점 기준으로 계산되어 있음)
- future window = 2025-09-16 초과 ~ 2025-11-16(데이터 최대일) 이내 (~2개월)
  이 구간의 신규 매출로 future_monetary/future_frequency를 별도 집계 (기존 y_revenue_14d는
  14일짜리라 VIP 전환처럼 누적에 시간이 걸리는 현상을 보기엔 너무 짧아 새로 계산)
- VIP 정의: 활동 고객(frequency>0) 중 monetary 상위 20% (pre-period/future-period 각각
  자기 구간 분포 기준으로 독립적으로 재계산 - 시점이 다르면 상위 20% 경계도 다름)
- 대상 모집단: pre-period에 이미 1회 이상 구매했고(frequency>=1), pre_is_vip==0인 고객
  (완전 신규 미구매 고객은 "VIP 전환"이 아니라 "신규 획득" 문제라 이 모델의 범위 밖)
- Y = future_is_vip (해당 미래창에서 VIP 기준을 넘겼는지)
- X = pre-period RFM/보상/제철구매/가격구간/구독여부/연령성별/지역 등 kitchen-sink
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
TRAIN_INDEX_DATE = pd.Timestamp("2025-09-16")
DATA_MAX_DATE = pd.Timestamp("2025-11-16")
VIP_PCTL = 0.80  # 상위 20%

print("=" * 70)
print("Model 2: VIP 전환 예측 (pre-period 신호 -> future window VIP 전환 여부)")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1) future window 매출 집계 (원본 주문 데이터에서 basket 단위로 직접 계산)
# ---------------------------------------------------------------------------
raw = pd.read_csv(
    BASE + "merged_master_enriched.csv", encoding="utf-8-sig",
    usecols=["회원번호", "주문일시", "구매금액", "is_cancelled"]
)
raw["주문일시_dt"] = pd.to_datetime(raw["주문일시"])
basket = (
    raw[raw["is_cancelled"] == 0]
    .groupby(["회원번호", "주문일시_dt"], as_index=False)
    .agg(basket_amt=("구매금액", "sum"))
)
future = basket[(basket["주문일시_dt"] > TRAIN_INDEX_DATE) & (basket["주문일시_dt"] <= DATA_MAX_DATE)]
future_agg = future.groupby("회원번호").agg(
    frequency_future=("basket_amt", "size"),
    monetary_future=("basket_amt", "sum"),
).reset_index()
print(f"\nfuture window(9/16 초과~11/16): 활동 고객 {len(future_agg):,}명")

# ---------------------------------------------------------------------------
# 2) pre-period 스냅샷과 결합, VIP 라벨 정의
# ---------------------------------------------------------------------------
tr = pd.read_csv(BASE + "snapshot_train.csv", encoding="utf-8-sig")
tr = tr.merge(future_agg, on="회원번호", how="left")
tr["frequency_future"] = tr["frequency_future"].fillna(0).astype(int)
tr["monetary_future"] = tr["monetary_future"].fillna(0.0)

# VIP 기준선: 각 구간에서 "활동 고객"(frequency>0)만 놓고 monetary 상위 20% 경계 계산
pre_active = tr[tr["frequency"] > 0]
pre_vip_cut = pre_active["monetary"].quantile(VIP_PCTL)
tr["pre_is_vip"] = ((tr["frequency"] > 0) & (tr["monetary"] >= pre_vip_cut)).astype(int)

fut_active = tr[tr["frequency_future"] > 0]
fut_vip_cut = fut_active["monetary_future"].quantile(VIP_PCTL)
tr["future_is_vip"] = ((tr["frequency_future"] > 0) & (tr["monetary_future"] >= fut_vip_cut)).astype(int)

print(f"pre-period VIP 기준(monetary 상위 20% 경계): {pre_vip_cut:,.0f}원 "
      f"(pre-period 활동고객 {len(pre_active):,}명 중 VIP {tr['pre_is_vip'].sum():,}명)")
print(f"future window VIP 기준: {fut_vip_cut:,.0f}원 "
      f"(future 활동고객 {len(fut_active):,}명 중 VIP {tr['future_is_vip'].sum():,}명)")

# 대상 모집단: pre-period에 이미 구매 이력 있고(frequency>=1) 아직 VIP는 아닌 고객
pop = tr[(tr["frequency"] >= 1) & (tr["pre_is_vip"] == 0)].copy()
print(f"\n분석 대상(pre-period 구매이력 O, 비-VIP): {len(pop):,}명")
print(f"이 중 future window에 VIP로 전환된 고객: {pop['future_is_vip'].sum():,}명 "
      f"({pop['future_is_vip'].mean()*100:.2f}%)")

# ---------------------------------------------------------------------------
# 3) 피처 구성 (kitchen-sink, run01b와 동일 방법론)
# ---------------------------------------------------------------------------
pop["log_frequency"] = np.log1p(pop["frequency"])
pop["log_monetary"] = np.log1p(pop["monetary"])
pop["log_aov"] = np.log1p(pop["aov"])
pop["log_recency"] = np.log1p(pop["recency_days"])
pop["결혼_결측표시"] = pop["결혼"].fillna("결측")

NUM_FEATS = [
    "log_frequency", "log_monetary", "log_aov", "log_recency",
    "regularity_cv", "reward_usage_rate", "나이",
    "preperiod_제철구매비율", "reward_ever_used", "preperiod_명절선물세트구매여부",
]
CAT_FEATS = ["age_band_h4", "gender_h4", "region_tier", "가격구간", "결혼_결측표시", "treatment_h4"]
ALL_FEATS = NUM_FEATS + CAT_FEATS
Y_COL = "future_is_vip"

X_all = pop[ALL_FEATS]
y_all = pop[Y_COL].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
)

pre = ColumnTransformer([
    ("num", StandardScaler(), NUM_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
])
X_train = pre.fit_transform(X_train_raw)
X_test = pre.transform(X_test_raw)
feat_names = pre.get_feature_names_out()

print(f"\n학습표본 {X_train.shape[0]:,}명 / 테스트표본 {X_test.shape[0]:,}명 "
      f"(인코딩 후 피처 {X_train.shape[1]}개)")

# ---------------------------------------------------------------------------
# 4) 로지스틱(L1, kitchen-sink) 모델
# ---------------------------------------------------------------------------
Cs = np.logspace(-3, 2, 15)
logit = LogisticRegressionCV(
    Cs=Cs, cv=5, penalty="l1", solver="liblinear", max_iter=5000,
    scoring="roc_auc", class_weight="balanced",
)
logit.fit(X_train, y_train)
auc_logit = roc_auc_score(y_test, logit.predict_proba(X_test)[:, 1])
print(f"\n[로지스틱 L1] 선택된 C: {logit.C_[0]:.4f} / 테스트 AUC: {auc_logit:.3f}")

coefs = logit.coef_[0]
kept = sorted([(f, c) for f, c in zip(feat_names, coefs) if abs(c) > 1e-6], key=lambda x: -abs(x[1]))
print(f"[로지스틱 L1] 살아남은 피처 {len(kept)}/{len(feat_names)}개, 영향력 상위 10개:")
for f, c in kept[:10]:
    print(f"   {f}: {c:+.3f}")

# ---------------------------------------------------------------------------
# 5) RandomForest + GridSearchCV
# ---------------------------------------------------------------------------
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": [10, 30, 50, 100],
}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Windows: GridSearchCV(n_jobs=-1) + RandomForest(n_jobs=-1) 중첩병렬화 시 TerminatedWorkerError -> RF는 n_jobs=1
gs = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced"),
    param_grid, cv=cv5, scoring="roc_auc", n_jobs=-1,
)
gs.fit(X_train, y_train)
auc_rf = roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1])
print(f"\n[RandomForest] 최적 파라미터: {gs.best_params_} (CV AUC={gs.best_score_:.3f})")
print(f"[RandomForest] 테스트 AUC: {auc_rf:.3f}")

importances = sorted(zip(feat_names, gs.best_estimator_.feature_importances_), key=lambda x: -x[1])
print("[RandomForest] 피처 중요도 상위 10개:")
for f, imp in importances[:10]:
    print(f"   {f}: {imp:.4f}")

# ---------------------------------------------------------------------------
# 6) 전체 대상 모집단에 스코어링 -> 성장 잠재고객(비-VIP -> VIP 전환 가능성) 리스트
# ---------------------------------------------------------------------------
X_pop_enc = pre.transform(pop[ALL_FEATS])
pop["vip_conversion_score_logit"] = logit.predict_proba(X_pop_enc)[:, 1]
pop["vip_conversion_score_rf"] = gs.predict_proba(X_pop_enc)[:, 1]

out_cols = [
    "회원번호", "frequency", "monetary", "가격구간", "region_tier", "age_gender_segment",
    "treatment_h4", "future_is_vip", "vip_conversion_score_logit", "vip_conversion_score_rf",
]
pop[out_cols].sort_values("vip_conversion_score_rf", ascending=False).to_csv(
    BASE + "run13_vip_conversion_scores.csv", index=False, encoding="utf-8-sig"
)

summary = pd.DataFrame([{
    "population_n": len(pop),
    "actual_conversion_rate": pop["future_is_vip"].mean(),
    "auc_logit": auc_logit,
    "auc_rf": auc_rf,
    "rf_best_params": str(gs.best_params_),
}])
summary.to_csv(BASE + "run13_vip_conversion_summary.csv", index=False, encoding="utf-8-sig")

print(f"\n저장: run13_vip_conversion_scores.csv, run13_vip_conversion_summary.csv")
