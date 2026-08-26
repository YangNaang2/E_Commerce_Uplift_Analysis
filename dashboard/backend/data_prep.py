# -*- coding: utf-8 -*-
"""
대시보드용 데이터 준비 — 원본 CSV(668,111행 등 대용량)를 매번 읽지 않도록,
필요한 집계만 미리 계산해 dashboard/backend/data/ 에 작은 CSV로 저장해둔다.
FastAPI(main.py)는 이 작은 파일들만 읽는다. 데이터가 갱신되면(전처리 재실행 등) 이 스크립트만 다시 돌리면 됨.
"""
import numpy as np
import pandas as pd

PROC = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
OUT = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/dashboard/backend/data/"

# ============================================================
# 1) KPI 요약
# ============================================================
snap_ho = pd.read_csv(PROC + "snapshot_holdout.csv", encoding="utf-8-sig")
known = snap_ho[snap_ho["include_in_uplift_model"] == 1]

run01b = pd.read_csv(PROC + "run01b_uplift_scores_holdout.csv", encoding="utf-8-sig")
run06 = pd.read_csv(PROC + "run06_revenue_uplift_scores.csv", encoding="utf-8-sig")
run11 = pd.read_csv(PROC + "run11_final_targeting_list.csv", encoding="utf-8-sig")

uplift_v = run01b.loc[run01b.treatment_h4 == 0, "uplift"]
rev_v = run06.loc[run06.treatment_h4 == 0, "uplift_revenue_rf"]

kpi = {
    "총_회원수": int(snap_ho["회원번호"].nunique()),
    "구독자수": int((known["treatment_h4"] == 1).sum()),
    "비구독자수": int((known["treatment_h4"] == 0).sum()),
    "구독율(%)": round((known["treatment_h4"] == 1).mean() * 100, 1),
    "전체_재구매율(%)": round(known["y_repurchase_14d"].mean() * 100, 1),
    "평균_AOV(원)": round(known["aov"].mean()),
    "재구매Uplift_평균(%p)": round(uplift_v.mean() * 100, 2),
    "매출Uplift_평균(원, RF)": round(rev_v.mean()),
    "Tier1_최우선_타겟수": int((run11["targeting_tier"] == "Tier1_최우선").sum()),
    "Tier2_매출리스크주의_타겟수": int((run11["targeting_tier"] == "Tier2_신호강함_매출리스크주의").sum()),
    "Tier3_보조신호_타겟수": int((run11["targeting_tier"] == "Tier3_보조신호").sum()),
}

# --- 티어별 "100% 전환 시" 14일 매출 Uplift 합계 (전환율 슬라이더의 기준값) ---
run11_rev = run11.merge(run06[["회원번호", "uplift_revenue_rf"]], on="회원번호", how="left")
for tier_key, tier_val in [("Tier1", "Tier1_최우선"), ("Tier2", "Tier2_신호강함_매출리스크주의"), ("Tier3", "Tier3_보조신호")]:
    total = run11_rev.loc[run11_rev["targeting_tier"] == tier_val, "uplift_revenue_rf"].sum()
    kpi[f"{tier_key}_100퍼센트전환시_14일매출증가(원)"] = round(total)
pd.DataFrame([kpi]).to_csv(OUT + "dash_kpi_summary.csv", index=False, encoding="utf-8-sig")
print("[1/5] dash_kpi_summary.csv 저장:", kpi)

# ============================================================
# 2) 타겟팅 리스트 (run11 + 인구통계 enrich)
# ============================================================
enrich_cols = ["회원번호", "age_band_h4", "gender_h4", "region_tier", "frequency", "monetary", "recency_days"]
targeting = run11.merge(snap_ho[enrich_cols], on="회원번호", how="left")
targeting = targeting.sort_values(["targeting_tier", "uplift_rf"], ascending=[True, False])
targeting.to_csv(OUT + "dash_targeting_list.csv", index=False, encoding="utf-8-sig")
print(f"[2/5] dash_targeting_list.csv 저장: {len(targeting)}행")

# ============================================================
# 3) 구독자 관리(이탈 알림) — treatment_h4==1(구독자)에게 run07과 동일한 위험군 밴드 적용
# ============================================================
subs = snap_ho[snap_ho["treatment_h4"] == 1].copy()


def risk_band(days):
    if days < 30:
        return "Active(<30일)"
    if days < 90:
        return "관심필요(30~89일)"
    return "이탈위험(90일+)"


subs["구독자_상태"] = subs["recency_days"].apply(risk_band)
sub_cols = ["회원번호", "recency_days", "구독자_상태", "frequency", "monetary", "aov",
            "regularity_cv", "age_band_h4", "gender_h4", "region_tier", "가격구간"]
subs_out = subs[sub_cols].sort_values("recency_days", ascending=False)
subs_out.to_csv(OUT + "dash_subscriber_watch.csv", index=False, encoding="utf-8-sig")
print(f"[3/5] dash_subscriber_watch.csv 저장: {len(subs_out)}명 "
      f"(이탈위험 {int((subs_out['구독자_상태']=='이탈위험(90일+)').sum())}명, "
      f"관심필요 {int((subs_out['구독자_상태']=='관심필요(30~89일)').sum())}명)")

# ============================================================
# 4) 계절/월별 매출 추이
# ============================================================
m = pd.read_csv(PROC + "merged_master_enriched.csv", encoding="utf-8-sig",
                 usecols=["주문일시", "구매금액", "주문취소여부", "구독여부"])
m = m[m["주문취소여부"].isna()]
m["주문일시"] = pd.to_datetime(m["주문일시"])
m["월"] = m["주문일시"].dt.month


def season(mon):
    if mon in (3, 4, 5):
        return "봄"
    if mon in (6, 7, 8):
        return "여름"
    if mon in (9, 10, 11):
        return "가을"
    return "겨울"


m["계절"] = m["월"].apply(season)
m["날짜"] = m["주문일시"].dt.date
# 원본 자체에 결측일이 있어(10월 3일·11월 5일) 월합계만 비교하면 왜곡됨
# -> 관측일수로 나눈 일평균매출을 같이 계산해 보정
month_trend = m.groupby("월").agg(
    총매출=("구매금액", "sum"), 주문건수=("구매금액", "size"), 관측일수=("날짜", "nunique"),
).reset_index()
month_trend["일평균매출"] = (month_trend["총매출"] / month_trend["관측일수"]).round().astype(int)
month_trend.to_csv(OUT + "dash_month_trend.csv", index=False, encoding="utf-8-sig")
print(f"[4/5] dash_month_trend.csv 저장 (관측일수·일평균매출 컬럼 추가):")
print(month_trend[["월", "관측일수", "총매출", "일평균매출"]].to_string(index=False))

season_trend = m.groupby("계절").agg(총매출=("구매금액", "sum"), 주문건수=("구매금액", "size")).reset_index()
season_order = {"봄": 0, "여름": 1, "가을": 2, "겨울": 3}
season_trend["순서"] = season_trend["계절"].map(season_order)
season_trend = season_trend.sort_values("순서").drop(columns="순서")
season_trend.to_csv(OUT + "dash_season_trend.csv", index=False, encoding="utf-8-sig")
print(f"dash_season_trend.csv 저장")

# ============================================================
# 5) 성별×나이별 주요 품목/매출
# ============================================================
m2 = pd.read_csv(PROC + "merged_master_enriched.csv", encoding="utf-8-sig",
                  usecols=["구매금액", "주문취소여부", "gender_h4", "age_band_h4", "물품중분류"])
m2 = m2[m2["주문취소여부"].isna()]

seg_summary = m2.groupby(["gender_h4", "age_band_h4"]).agg(
    총매출=("구매금액", "sum"), 구매건수=("구매금액", "size")
).reset_index()
seg_summary.to_csv(OUT + "dash_segment_summary.csv", index=False, encoding="utf-8-sig")

seg_top_items = (
    m2.groupby(["gender_h4", "age_band_h4", "물품중분류"])["구매금액"].sum()
    .reset_index()
    .sort_values(["gender_h4", "age_band_h4", "구매금액"], ascending=[True, True, False])
    .groupby(["gender_h4", "age_band_h4"]).head(5)
)
seg_top_items.to_csv(OUT + "dash_segment_top_items.csv", index=False, encoding="utf-8-sig")
print(f"[5/5] dash_segment_summary.csv / dash_segment_top_items.csv 저장")

print("\n=== 데이터 준비 완료 ===")
