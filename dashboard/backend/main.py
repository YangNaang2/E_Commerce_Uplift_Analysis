# -*- coding: utf-8 -*-
"""
FastAPI 백엔드 — dashboard/backend/data/ 의 사전집계 CSV를 로드해 JSON으로 제공.
실행: uvicorn main:app --reload --port 8000  (dashboard/backend 폴더에서)
"""
import io
from datetime import datetime

import numpy as np
import pandas as pd

# 프로젝트 데이터의 기준 시점(Holdout index_date). 신규회원 최근성 계산에 datetime.now()를 쓰면
# 실제 시스템 날짜(2026년)와 비교돼 말이 안 되는 값이 나와서, 데이터 기준일로 고정.
REFERENCE_DATE = datetime(2025, 11, 16)
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

DATA = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/dashboard/backend/data/"
PROC = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"

app = FastAPI(title="이커머스 Uplift 대시보드 API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---- 서버 시작 시 한 번만 로드 (매 요청마다 CSV 안 읽음) ----
KPI = pd.read_csv(DATA + "dash_kpi_summary.csv", encoding="utf-8-sig").iloc[0].to_dict()
TARGETING = pd.read_csv(DATA + "dash_targeting_list.csv", encoding="utf-8-sig")
SUB_WATCH = pd.read_csv(DATA + "dash_subscriber_watch.csv", encoding="utf-8-sig")
MONTH_TREND = pd.read_csv(DATA + "dash_month_trend.csv", encoding="utf-8-sig")
SEASON_TREND = pd.read_csv(DATA + "dash_season_trend.csv", encoding="utf-8-sig")
SEG_SUMMARY = pd.read_csv(DATA + "dash_segment_summary.csv", encoding="utf-8-sig")
SEG_TOP_ITEMS = pd.read_csv(DATA + "dash_segment_top_items.csv", encoding="utf-8-sig")

# 회원번호 조회용: 원자료(run11/run01b/run06/스냅샷)를 회원번호 인덱스로 미리 합쳐둠
_snap = pd.read_csv(PROC + "snapshot_holdout.csv", encoding="utf-8-sig")
_run06 = pd.read_csv(PROC + "run06_revenue_uplift_scores.csv", encoding="utf-8-sig")[["회원번호", "uplift_revenue_rf"]]
MEMBER_LOOKUP = (
    TARGETING[["회원번호", "uplift_rf", "uplift_logit", "targeting_tier"]]
    .merge(_snap[["회원번호", "가격구간", "age_band_h4", "gender_h4", "region_tier",
                   "frequency", "monetary", "aov", "recency_days", "treatment_h4"]], on="회원번호", how="right")
    .merge(_run06, on="회원번호", how="left")
).set_index("회원번호")


@app.get("/api/kpi")
def get_kpi():
    return KPI


@app.get("/api/targeting")
def get_targeting(tier: str | None = None, limit: int = 50):
    df = TARGETING if tier is None else TARGETING[TARGETING["targeting_tier"] == tier]
    return df.head(limit).to_dict(orient="records")


@app.get("/api/subscriber-watch")
def get_subscriber_watch(status: str | None = None, limit: int = 50):
    df = SUB_WATCH if status is None else SUB_WATCH[SUB_WATCH["구독자_상태"] == status]
    return df.head(limit).to_dict(orient="records")


@app.get("/api/trend/month")
def get_month_trend():
    return MONTH_TREND.to_dict(orient="records")


@app.get("/api/trend/season")
def get_season_trend():
    return SEASON_TREND.to_dict(orient="records")


@app.get("/api/segment/summary")
def get_segment_summary():
    return SEG_SUMMARY.to_dict(orient="records")


@app.get("/api/segment/top-items")
def get_segment_top_items(gender_h4: str | None = None, age_band_h4: str | None = None):
    df = SEG_TOP_ITEMS
    if gender_h4:
        df = df[df["gender_h4"] == gender_h4]
    if age_band_h4:
        df = df[df["age_band_h4"] == age_band_h4]
    return df.to_dict(orient="records")


@app.get("/api/member/{member_id}")
def get_member(member_id: int):
    """회원번호로 우리 데이터에 이미 있는 고객인지 조회 (Tier/Uplift 점수 포함)."""
    if member_id not in MEMBER_LOOKUP.index:
        return {"found": False, "회원번호": member_id}
    row = MEMBER_LOOKUP.loc[member_id]
    if isinstance(row, pd.DataFrame):  # 혹시 중복 인덱스면 첫 행만
        row = row.iloc[0]
    out = row.replace({np.nan: None}).to_dict()
    out["found"] = True
    out["회원번호"] = member_id
    return out


@app.post("/api/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    """
    고객 주문 CSV 업로드 분석.
    - 회원번호 컬럼이 있고 우리 데이터에 이미 있는 회원이면: 저장된 Uplift/Tier 점수를 그대로 반환
    - 없는(신규) 회원이면: 업로드된 주문 행 자체로 RFM만 간단 계산해서 반환(모델 점수는 제공 안 함 — 학습된 모델이 없는 신규고객이라 정확한 추정 불가함을 명시)
    """
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(raw), encoding="cp949")

    if "회원번호" not in df.columns:
        return {"error": "CSV에 '회원번호' 컬럼이 없습니다. 원본 주문 데이터(Sales_Data) 형식을 확인해주세요."}

    member_ids = df["회원번호"].dropna().unique().tolist()
    results = []
    for mid in member_ids:
        known = get_member(int(mid))
        if known["found"]:
            results.append(known)
        else:
            # 신규 회원: 업로드된 행 자체로 RFM만 계산 (모델 예측은 불가 - 학습 데이터에 없음)
            sub = df[df["회원번호"] == mid]
            rfm = {"found": False, "회원번호": int(mid), "신규_주문건수": int(len(sub))}
            if "구매금액" in sub.columns:
                rfm["신규_총구매금액"] = float(sub["구매금액"].sum())
            if "주문일시" in sub.columns:
                try:
                    dates = pd.to_datetime(sub["주문일시"])
                    rfm["최근성_경과일"] = int((REFERENCE_DATE - dates.max()).days)
                except Exception:
                    pass
            rfm["안내"] = "학습 데이터에 없는 신규 회원이라 Uplift 모델 점수는 제공되지 않습니다(RFM 요약만 계산)."
            results.append(rfm)

    return {"member_count": len(results), "results": results}
