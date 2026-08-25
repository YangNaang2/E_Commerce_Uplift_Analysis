# -*- coding: utf-8 -*-
"""
특성 개수를 늘리면 T-learner Uplift 추정이 실제로 좋아지는지/나빠지는지 비교 실험
버전1: 기존(run01) 피처셋
버전2: 확장(합리적 추가) - 결혼, abnormal_account_flag, reward_ever_used, 명절선물세트 추가
버전3: 확장+region_key(고카디널리티) - 과적합 위험을 보여주기 위한 극단 비교
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
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

train_known = tr[tr["include_in_uplift_model"] == 1].copy()
ho_known = ho[ho["include_in_uplift_model"] == 1].copy()

CONFIGS = {
    "v1_기존(run01)": dict(
        num=["log_frequency", "log_monetary", "log_aov", "log_recency", "regularity_cv", "reward_usage_rate", "나이", "preperiod_제철구매비율"],
        cat=["age_band_h4", "gender_h4", "region_tier", "가격구간"],
    ),
    "v2_확장(합리적 추가)": dict(
        num=["log_frequency", "log_monetary", "log_aov", "log_recency", "regularity_cv", "reward_usage_rate", "나이", "preperiod_제철구매비율", "abnormal_account_flag", "reward_ever_used", "preperiod_명절선물세트구매여부"],
        cat=["age_band_h4", "gender_h4", "region_tier", "가격구간", "결혼_결측표시"],
    ),
    "v3_확장+region_key(고카디널리티)": dict(
        num=["log_frequency", "log_monetary", "log_aov", "log_recency", "regularity_cv", "reward_usage_rate", "나이", "preperiod_제철구매비율", "abnormal_account_flag", "reward_ever_used", "preperiod_명절선물세트구매여부"],
        cat=["age_band_h4", "gender_h4", "region_key", "가격구간", "결혼_결측표시"],
    ),
}

results = []
for name, cfg in CONFIGS.items():
    feats = cfg["num"] + cfg["cat"]
    n_cat_levels = sum(train_known[c].nunique() for c in cfg["cat"])
    pre = ColumnTransformer([
        ("num", StandardScaler(), cfg["num"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["cat"]),
    ])
    X_train = pre.fit_transform(train_known[feats])
    y_train = train_known["y_repurchase_14d"].values
    t_train = train_known["treatment_h4"].values
    n_features_final = X_train.shape[1]

    X_sub, y_sub = X_train[t_train == 1], y_train[t_train == 1]
    X_non, y_non = X_train[t_train == 0], y_train[t_train == 0]

    model_A = LogisticRegression(max_iter=3000, C=1.0).fit(X_sub, y_sub)
    model_B = LogisticRegression(max_iter=3000, C=1.0).fit(X_non, y_non)

    X_ho_known = pre.transform(ho_known[feats])
    sub_mask = ho_known["treatment_h4"] == 1
    non_mask = ho_known["treatment_h4"] == 0
    auc_A = roc_auc_score(ho_known.loc[sub_mask, "y_repurchase_14d"], model_A.predict_proba(X_ho_known[sub_mask.values])[:, 1])
    auc_B = roc_auc_score(ho_known.loc[non_mask, "y_repurchase_14d"], model_B.predict_proba(X_ho_known[non_mask.values])[:, 1])

    X_ho_all = pre.transform(ho[feats])
    uplift = model_A.predict_proba(X_ho_all)[:, 1] - model_B.predict_proba(X_ho_all)[:, 1]
    ho_tmp = ho.copy()
    ho_tmp["uplift"] = uplift
    target = ho_tmp[ho_tmp["treatment_h4"] == 0]

    # 부트스트랩 CI (전체 평균)
    rng = np.random.default_rng(42)
    vals = target["uplift"].values
    boot = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(1000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # 남성 세그먼트 uplift 극단값 체크(과적합 여부 확인용)
    male_mask = target["age_gender_segment"].astype(str).str.endswith("_M") if "age_gender_segment" in target.columns else None

    results.append({
        "설정": name,
        "학습표본(구독자/비구독자)": f"{sum(t_train==1)}/{sum(t_train==0)}",
        "인코딩후 피처수": n_features_final,
        "AUC_구독자모델": round(auc_A, 3),
        "AUC_비구독자모델": round(auc_B, 3),
        "평균Uplift(%p)": round(vals.mean() * 100, 3),
        "95%CI(%p)": f"[{lo*100:.2f}, {hi*100:.2f}]",
        "CI폭(%p)": round((hi - lo) * 100, 3),
        "Uplift 최댓값(%p)": round(vals.max() * 100, 2),
        "Uplift 표준편차(%p)": round(vals.std() * 100, 3),
    })

result_df = pd.DataFrame(results)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(result_df.to_string(index=False))
