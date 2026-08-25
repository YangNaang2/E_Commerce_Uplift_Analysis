# -*- coding: utf-8 -*-
"""
전처리 파이프라인 v1
- 8-Step 계획(CLAUDE.md) + A/B/C/D 통합 인사이트(전처리_통합_인사이트_ABCD.md) 반영
- 분석단위: 단일 스냅샷(회원당 1행) x Train/Holdout 2개
  Train  index_date = 2025-09-16, Y = 9/17~9/30 재구매여부(14일)
  Holdout index_date = 2025-10-26, Y = 10/27~11/9 재구매여부(14일)
  (8/24 변경: 원래 11/2였으나, 원본 Sales_Data 자체에 11/10~14 5일 연속 결측이 있어
   14일 라벨 윈도우 안에 결측이 절반 가까이 걸림 -> Y가 과소추정될 위험 발견(팀원 EDA로 확인).
   원본에 결측이 없는 10/3·10/11~12·11/10~14를 모두 피하는 구간으로 재설정.)
- 이번 버전 범위에 포함 O: 지역 정규화/region_tier, 계절(제철) 태깅, RFM, 가격구간,
  적립금 이력, 연령x성별 세그먼트, 구독여부 재분류(Treatment), Y(14일), 이상치 플래그,
  상품중량 정규식 정제(8/25 추가 구현, 핵심 모델링 피처로는 미사용 - 데이터 정제 완결성 목적)
- 이번 버전 범위에서 제외(TODO, 핵심 모델링에 필수 아님): 등록카드 3분류(run18에서 별도 구현),
  주문시간 "XX:60" 보정(run20에서 별도 구현)
"""
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
RAW_PATH = BASE + "merged_master.csv"

TRAIN_INDEX_DATE = pd.Timestamp("2025-09-16")
HOLDOUT_INDEX_DATE = pd.Timestamp("2025-10-26")  # 8/24: 11/10~14 원본결측 회피 위해 11/2->10/26 변경
N_DAYS = 14
DATA_MAX_DATE = pd.Timestamp("2025-11-16")

# ---------------------------------------------------------------------------
# Step 0: 로드 & 무결성 재확인
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
assert df.shape == (668111, 36), f"[Step0] shape 불일치: {df.shape}"
assert df["회원번호"].isna().sum() == 0
assert df["제품번호"].isna().sum() == 0
df["주문일시_dt"] = pd.to_datetime(df["주문일시"])
assert df["주문일시_dt"].min() == pd.Timestamp("2025-01-11")
assert df["주문일시_dt"].max() == pd.Timestamp("2025-11-16")
print("[Step0] OK - shape/키/날짜범위 검증 통과")

# ---------------------------------------------------------------------------
# Step 1: 컬럼 정제 - 주소지 정규화, region_key, region_tier (D파트 반영)
# ---------------------------------------------------------------------------
ADDR_NORMALIZE = {"경기": "경기도", "광주": "광주광역시", "강원": "강원도", "서울": "서울특별시"}
df["주소지_정규화"] = df["주소지"].replace(ADDR_NORMALIZE)

# 세종시는 세부주소지를 동/로 단위로 세분화하지 않고 하나로 롤업 (D 3.2 제안)
df["세부주소지_정제"] = df["세부주소지"]
sejong_mask = df["주소지_정규화"] == "세종특별자치시"
df.loc[sejong_mask, "세부주소지_정제"] = "세종시"

df["region_key"] = df["주소지_정규화"].astype(str) + "_" + df["세부주소지_정제"].astype(str)

TIER1 = {"서울특별시", "경기도", "인천광역시"}
TIER2 = {"부산광역시", "대구광역시", "대전광역시", "광주광역시", "울산광역시", "세종특별자치시"}
TIER3 = {"강원도", "충청남도", "충청북도", "전라남도", "전라북도", "경상남도", "경상북도", "제주특별자치도"}


def to_tier(addr):
    if addr in TIER1:
        return "Tier1_새벽배송가능권(가정)"
    if addr in TIER2:
        return "Tier2_익일배송표준권(가정)"
    if addr in TIER3:
        return "Tier3_배송취약권(가정)"
    return "미상"


df["region_tier"] = df["주소지_정규화"].map(to_tier)
df["is_dawn_delivery_zone"] = (df["region_tier"] == "Tier1_새벽배송가능권(가정)").astype(int)
print("[Step1] 지역 정규화/region_key/region_tier 생성 완료")
print(df["region_tier"].value_counts(dropna=False).to_dict())

# 취소/배송 관련 플래그
df["is_cancelled"] = (df["주문취소여부"] == "주문취소").astype(int)
df["is_cancelled_no_delivery"] = df["is_cancelled"]  # D: 배송일 결측=취소와 100% 일치, 동일 플래그로 취급
df["delivery_leadtime_days"] = (
    pd.to_datetime(df["배송완료일"]) - df["주문일시_dt"]
).dt.days  # 취소건은 NaN 유지 (구조적 결측)

# 요일/주말여부 (C파트)
df["요일"] = df["주문일시_dt"].dt.day_name()
df["주말여부"] = df["주문일시_dt"].dt.dayofweek.isin([5, 6]).astype(int)

# ---------------------------------------------------------------------------
# Step 1b: 명절선물세트여부 (C파트) - 식품 계열 + 키워드
# ---------------------------------------------------------------------------
GIFT_KEYWORDS = r"선물모음|선물용|선물세트|세트"
FOOD_CATEGORIES = set(df.loc[~df["물품대분류"].astype(str).str.contains("생활|화장|잡화|리빙", na=False), "물품대분류"].unique())
df["명절선물세트여부"] = (
    df["물품명"].astype(str).str.contains(GIFT_KEYWORDS, regex=True, na=False)
    & df["물품대분류"].isin(FOOD_CATEGORIES)
).astype(int)
print(f"[Step1b] 명절선물세트 태깅 {df['명절선물세트여부'].sum()}건")

# ---------------------------------------------------------------------------
# Step 1c: 자체 공휴일 캘린더 보정 + 연휴전후구간 (C파트 3-1)
# ---------------------------------------------------------------------------
HOLIDAY_DATES = pd.to_datetime(
    [
        "2025-01-29", "2025-01-30", "2025-01-31",  # 설
        "2025-03-01",                                # 삼일절 (보정 추가)
        "2025-05-05", "2025-05-06", "2025-05-08",   # 어린이날/부처님오신날 대체
        "2025-06-06",                                 # 현충일
        "2025-08-15",                                 # 광복절
        "2025-10-03", "2025-10-04", "2025-10-05", "2025-10-06",  # 개천절~추석연휴~추석당일
        "2025-10-07", "2025-10-08",                                 # 추석연휴~대체공휴일
        "2025-10-09",                                                # 한글날
    ]
)
holiday_set = set(HOLIDAY_DATES)
pre_holiday = set().union(*[set(HOLIDAY_DATES - pd.Timedelta(days=d)) for d in (1, 2, 3)])
post_holiday = set().union(*[set(HOLIDAY_DATES + pd.Timedelta(days=d)) for d in (1, 2, 3)])


def to_holiday_period(d):
    if d in holiday_set:
        return "연휴중"
    if d in pre_holiday:
        return "연휴직전"
    if d in post_holiday:
        return "연휴직후"
    return "평시"


df["연휴전후구간"] = df["주문일시_dt"].map(to_holiday_period)
print("[Step1c]", df["연휴전후구간"].value_counts().to_dict())

# ---------------------------------------------------------------------------
# Step 1d: 상품중량 정제 (8/25 구현 - 기존에 TODO로 미뤄뒀던 항목)
#   - 정규식으로 중량/용량(g·kg·ml·㎖·l)만 추출해 그램(g) 환산값으로 정규화(부피는 g=ml 근사)
#   - 범위 표기("0.7~1kg")는 평균값으로 환산
#   - 나머지 텍스트(용도·원산지·묶음정보·개수단위 등)는 옵션정보 컬럼으로 분리
#   - 개수단위(개/알/장/봉 등)만 있어 중량 정보가 없는 경우는 상품중량_g를 결측으로 남김
# ---------------------------------------------------------------------------
_WEIGHT_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*~\s*(\d+(?:\.\d+)?)\s*(kg|g|ml|㎖|ℓ|l)", re.IGNORECASE)
_WEIGHT_SINGLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|g|ml|㎖|ℓ|l)", re.IGNORECASE)
_UNIT_TO_G = {"g": 1.0, "kg": 1000.0, "ml": 1.0, "㎖": 1.0, "ℓ": 1000.0, "l": 1000.0}


def _parse_weight(raw):
    if pd.isna(raw):
        return np.nan, np.nan
    s = str(raw)
    m = _WEIGHT_RANGE_RE.search(s)
    if m:
        lo, hi, unit = float(m.group(1)), float(m.group(2)), m.group(3).lower()
        val_g = (lo + hi) / 2 * _UNIT_TO_G[unit]
        option = (s[:m.start()] + s[m.end():]).strip(" :/-*")
        return val_g, (option if option else np.nan)
    m = _WEIGHT_SINGLE_RE.search(s)
    if m:
        val_g = float(m.group(1)) * _UNIT_TO_G[m.group(2).lower()]
        option = (s[:m.start()] + s[m.end():]).strip(" :/-*")
        return val_g, (option if option else np.nan)
    return np.nan, s  # 중량 정보 없음(개수단위·프로모션 문구 등) -> 원문 그대로 옵션정보로


_weight_map = {v: _parse_weight(v) for v in df["상품중량"].unique()}
df["상품중량_g"] = df["상품중량"].map(lambda v: _weight_map[v][0])
df["옵션정보"] = df["상품중량"].map(lambda v: _weight_map[v][1])

_n_notna = df["상품중량"].notna().sum()
_n_extracted = df["상품중량_g"].notna().sum()
print(f"[Step1d] 상품중량 정제: 원본 비결측 {_n_notna:,}건 중 {_n_extracted:,}건"
      f"({_n_extracted / _n_notna * 100:.1f}%) 중량(g) 추출 성공, "
      f"나머지 {_n_notna - _n_extracted:,}건은 개수단위/프로모션 문구 등이라 옵션정보로만 분리")

# ---------------------------------------------------------------------------
# Step 2a: 제철여부 (C파트 3-2) - 물품중분류별 월별 판매비중 top3 >=65%
# ---------------------------------------------------------------------------
valid = df[df["is_cancelled"] == 0].copy()
monthly_qty = valid.groupby(["물품중분류", "주문년월"])["구매수량"].sum().reset_index()
total_qty = monthly_qty.groupby("물품중분류")["구매수량"].transform("sum")
monthly_qty["비중"] = monthly_qty["구매수량"] / total_qty
item_total_count = valid.groupby("물품중분류").size()

seasonal_months = {}
for item, g in monthly_qty.groupby("물품중분류"):
    top3 = g.nlargest(3, "비중")
    concentration = top3["비중"].sum()
    if item_total_count.get(item, 0) < 200:
        seasonal_months[item] = None  # 표본부족 -> 상시 취급(외부데이터 보완 필요, TODO)
    elif concentration >= 0.65:
        seasonal_months[item] = set(top3["주문년월"])
    else:
        seasonal_months[item] = None  # 상시 품목

n_seasonal_items = sum(1 for v in seasonal_months.values() if v is not None)
print(f"[Step2a] 계절성 품목 {n_seasonal_items}/{len(seasonal_months)}개 (임계값 65%, 표본<200건은 상시 처리)")


def is_in_season(row):
    months = seasonal_months.get(row["물품중분류"])
    if months is None:
        return 0
    return int(row["주문년월"] in months)


df["제철상품구매여부"] = df.apply(is_in_season, axis=1)

# ---------------------------------------------------------------------------
# Step 2b: 이상치 플래그 (Step3 항목이지만 스냅샷 계산 전에 필요해 미리 생성)
#   - amount_outlier_flag: TRAIN 기간 basket_amt IQR 기준(정보 누수 방지, A파트)
# ---------------------------------------------------------------------------
basket_all = (
    df[df["is_cancelled"] == 0]
    .groupby(["회원번호", "주문일시_dt"], as_index=False)
    .agg(
        basket_amt=("구매금액", "sum"),
        basket_item_count=("구매금액", "size"),
        reward_used=("사용 적립금", lambda s: int((s > 0).any())),
        reward_amt=("사용 적립금", "sum"),
    )
)
train_baskets_for_iqr = basket_all[basket_all["주문일시_dt"] <= TRAIN_INDEX_DATE]
q1, q3 = train_baskets_for_iqr["basket_amt"].quantile([0.25, 0.75])
iqr = q3 - q1
iqr_lo, iqr_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
basket_all["amount_outlier_flag"] = (
    (basket_all["basket_amt"] < iqr_lo) | (basket_all["basket_amt"] > iqr_hi)
).astype(int)
print(f"[Step2b] basket_amt IQR(Train기준) = [{iqr_lo:.0f}, {iqr_hi:.0f}], 이상치 {basket_all['amount_outlier_flag'].mean()*100:.1f}%")

print("스크립트 1부(Step0~2b) 완료")

# ---------------------------------------------------------------------------
# Step 2c: 회원 정적 속성 테이블 (연령대/성별/지역 등, B/D파트)
# ---------------------------------------------------------------------------
def age_band_h4(age):
    if age < 30:
        return "<30"
    if age < 40:
        return "30s"
    if age < 50:
        return "40s"
    if age < 60:
        return "50s"
    return "60+"


member_static = df.groupby("회원번호").first().reset_index()[
    ["회원번호", "나이", "성별", "결혼", "주소지_정규화", "region_key", "region_tier", "is_dawn_delivery_zone", "구독여부"]
]
member_static["age_band_h4"] = member_static["나이"].map(age_band_h4)
member_static["gender_h4"] = member_static["성별"].map({"여": "F", "남": "M"}).fillna("Unknown")
member_static["age_gender_segment"] = member_static["age_band_h4"] + "_" + member_static["gender_h4"]
print("[Step2c] 회원 정적 속성 테이블:", member_static.shape)

# 제철구매비율/명절선물세트구매여부용 원본(취소 제외) 행 데이터
item_valid = df[df["is_cancelled"] == 0][
    ["회원번호", "주문일시_dt", "제철상품구매여부", "명절선물세트여부"]
]


# ---------------------------------------------------------------------------
# Step 2d: 스냅샷 빌더 (Train/Holdout 공용 함수)
# ---------------------------------------------------------------------------
def build_snapshot(index_date, label_start, label_end, price_bins=None):
    pre_b = basket_all[basket_all["주문일시_dt"] <= index_date].copy()
    pre_i = item_valid[item_valid["주문일시_dt"] <= index_date]

    agg = pre_b.groupby("회원번호").agg(
        frequency=("basket_amt", "size"),
        monetary=("basket_amt", "sum"),
        aov=("basket_amt", "mean"),
        last_date=("주문일시_dt", "max"),
        reward_ever_used=("reward_used", "max"),
        reward_usage_rate=("reward_used", "mean"),
    ).reset_index()
    agg["recency_days"] = (index_date - agg["last_date"]).dt.days

    # 재구매 간격 규칙성(CV) - subscription 재분류용 보조피처
    pre_b_sorted = pre_b.sort_values(["회원번호", "주문일시_dt"])
    pre_b_sorted["interval"] = pre_b_sorted.groupby("회원번호")["주문일시_dt"].diff().dt.days
    cv = pre_b_sorted.groupby("회원번호")["interval"].agg(lambda s: s.std() / s.mean() if s.mean() else np.nan)
    agg["regularity_cv"] = agg["회원번호"].map(cv)
    median_cv = agg["regularity_cv"].median()
    agg["regularity_cv"] = agg["regularity_cv"].fillna(median_cv)

    seasonal_share = pre_i.groupby("회원번호")["제철상품구매여부"].mean().rename("preperiod_제철구매비율")
    gift_flag = pre_i.groupby("회원번호")["명절선물세트여부"].max().rename("preperiod_명절선물세트구매여부")
    agg = agg.merge(seasonal_share, on="회원번호", how="left").merge(gift_flag, on="회원번호", how="left")
    agg["preperiod_제철구매비율"] = agg["preperiod_제철구매비율"].fillna(0)
    agg["preperiod_명절선물세트구매여부"] = agg["preperiod_명절선물세트구매여부"].fillna(0)

    # 이상거래계정 플래그: 이 스냅샷 preperiod frequency 상위 1%
    p99 = agg["frequency"].quantile(0.99)
    agg["abnormal_account_flag"] = (agg["frequency"] >= p99).astype(int)

    snap = agg.merge(member_static, on="회원번호", how="left")

    # Y: index_date 초과 ~ label_end 이내 유효 basket 존재 여부
    future = basket_all[(basket_all["주문일시_dt"] > index_date) & (basket_all["주문일시_dt"] <= label_end)]
    repurchase_ids = set(future["회원번호"].unique())
    snap["y_repurchase_14d"] = snap["회원번호"].isin(repurchase_ids).astype(int)
    future_revenue = future.groupby("회원번호")["basket_amt"].sum().rename("y_revenue_14d")
    snap = snap.merge(future_revenue, on="회원번호", how="left")
    snap["y_revenue_14d"] = snap["y_revenue_14d"].fillna(0.0)  # run06: 매출 관점 Uplift용 Y(14일 내 재구매 금액, 미구매시 0)
    snap["label_observable_flag"] = int(label_end <= DATA_MAX_DATE)

    # 가격구간 (Train 기준 분위수 경계 고정, A파트)
    if price_bins is None:
        _, price_bins = pd.qcut(snap["aov"], 4, retbins=True, duplicates="drop")
        price_bins = price_bins.copy()
        price_bins[0], price_bins[-1] = -np.inf, np.inf
    snap["가격구간"] = pd.cut(snap["aov"], bins=price_bins, labels=["Q1_저가", "Q2", "Q3", "Q4_고가"])

    # Treatment(구독여부) 원본 + 재분류
    snap["treatment_source"] = np.select(
        [snap["구독여부"] == True, snap["구독여부"] == False],
        ["original_true", "original_false"],
        default="unresolved",
    )

    feat_cols = ["frequency", "monetary", "recency_days", "reward_usage_rate", "regularity_cv", "나이"]
    snap["log_frequency"] = np.log1p(snap["frequency"])
    snap["log_monetary"] = np.log1p(snap["monetary"])
    snap["log_recency"] = np.log1p(snap["recency_days"])
    model_feats = ["log_frequency", "log_monetary", "log_recency", "reward_usage_rate", "regularity_cv", "나이"]

    known = snap[snap["treatment_source"] != "unresolved"]
    unknown = snap[snap["treatment_source"] == "unresolved"]
    prob = pd.Series(index=snap.index, dtype=float)
    if len(unknown) > 0 and len(known) > 0:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(known[model_feats], known["구독여부"] == True)
        prob.loc[known.index] = clf.predict_proba(known[model_feats])[:, 1]
        prob.loc[unknown.index] = clf.predict_proba(unknown[model_feats])[:, 1]
    snap["구독_추정확률"] = prob  # 참고용. unresolved 구간은 0.40~0.58 근처로 사실상 판별력 없음(팀 검토 완료, 8/23)

    # 8/23 결정: 행동기반 강제분류(0.5 threshold) 대신 unresolved는 NaN으로 남기고
    # 메인 Uplift 분석 표본에서 제외한다. include_in_uplift_model로 필터링.
    snap["treatment_h4"] = np.select(
        [snap["treatment_source"] == "original_true", snap["treatment_source"] == "original_false"],
        [1, 0],
        default=np.nan,
    )
    snap["include_in_uplift_model"] = (snap["treatment_source"] != "unresolved").astype(int)

    return snap, price_bins


snap_train, PRICE_BINS = build_snapshot(
    TRAIN_INDEX_DATE, TRAIN_INDEX_DATE + pd.Timedelta(days=1), TRAIN_INDEX_DATE + pd.Timedelta(days=N_DAYS)
)
snap_holdout, _ = build_snapshot(
    HOLDOUT_INDEX_DATE, HOLDOUT_INDEX_DATE + pd.Timedelta(days=1), HOLDOUT_INDEX_DATE + pd.Timedelta(days=N_DAYS),
    price_bins=PRICE_BINS,
)

print(f"[Step2d] Train 스냅샷 {snap_train.shape}, Holdout 스냅샷 {snap_holdout.shape}")
print("Train treatment_source:", snap_train["treatment_source"].value_counts().to_dict())
print("Train y_repurchase_14d 양성률:", snap_train["y_repurchase_14d"].mean())
print("Holdout treatment_source:", snap_holdout["treatment_source"].value_counts().to_dict())
print("Holdout y_repurchase_14d 양성률:", snap_holdout["y_repurchase_14d"].mean())

# ---------------------------------------------------------------------------
# Step 7 (부분): 최종 저장 + 검증
# ---------------------------------------------------------------------------
OUT_COLS = [
    "회원번호", "나이", "age_band_h4", "gender_h4", "age_gender_segment", "결혼",
    "주소지_정규화", "region_key", "region_tier", "is_dawn_delivery_zone",
    "frequency", "monetary", "aov", "recency_days", "regularity_cv",
    "reward_ever_used", "reward_usage_rate",
    "preperiod_제철구매비율", "preperiod_명절선물세트구매여부",
    "가격구간", "abnormal_account_flag",
    "구독여부", "treatment_h4", "treatment_source", "구독_추정확률", "include_in_uplift_model",
    "y_repurchase_14d", "y_revenue_14d", "label_observable_flag",
]
snap_train[OUT_COLS].to_csv(BASE + "snapshot_train.csv", index=False, encoding="utf-8-sig")
snap_holdout[OUT_COLS].to_csv(BASE + "snapshot_holdout.csv", index=False, encoding="utf-8-sig")

df = df.merge(
    member_static[["회원번호", "나이", "age_band_h4", "gender_h4", "age_gender_segment"]],
    on="회원번호", how="left", suffixes=("", "_dup"),
)
ENRICHED_COLS = [
    "회원번호", "제품번호", "주문일시", "구매금액", "구매수량", "주문취소여부",
    "구독여부",  # 원본 True/False/NaN 그대로. NaN(unresolved)은 Treatment 분석에서 제외 권장
    "나이", "age_band_h4", "gender_h4", "age_gender_segment",
    "주소지_정규화", "region_key", "region_tier", "is_dawn_delivery_zone",
    "is_cancelled", "delivery_leadtime_days",
    "요일", "주말여부", "연휴전후구간", "명절선물세트여부",
    "물품중분류", "제철상품구매여부",
    "상품중량_g", "옵션정보",
]
df[ENRICHED_COLS].to_csv(BASE + "merged_master_enriched.csv", index=False, encoding="utf-8-sig")

print("[Step7] 저장 완료: snapshot_train.csv, snapshot_holdout.csv, merged_master_enriched.csv")
print(f"가격구간 경계(Train 기준, AOV): {PRICE_BINS}")
