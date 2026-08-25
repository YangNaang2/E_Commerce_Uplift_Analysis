# -*- coding: utf-8 -*-
"""
run17: 코호트 리텐션 분석 [추가분석, H1~H4와 독립]

회원가입일이 원본에 없어서, "첫 구매월"을 가입 프록시로 삼아 코호트를 구성함.
각 코호트가 이후 몇 개월째까지 재구매(비취소 주문)를 이어가는지 리텐션율로 추적.
"""
import numpy as np
import pandas as pd

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
OUT = "C:/Users/aidan/AppData/Local/Temp/claude/C--Users-aidan-OneDrive-----------/526a44f2-eab0-428f-be6f-d07d153f63c4/scratchpad/"

df = pd.read_csv(BASE + "merged_master.csv", encoding="utf-8-sig",
                  usecols=["회원번호", "주문일시", "주문취소여부"])
df = df[df["주문취소여부"] != "주문취소"].copy()
df["주문일시_dt"] = pd.to_datetime(df["주문일시"])
df["order_month"] = df["주문일시_dt"].dt.to_period("M")
print(f"비취소 주문 {len(df):,}건")

first_month = df.groupby("회원번호")["order_month"].min().rename("cohort_month")
df = df.merge(first_month, on="회원번호")
df["month_offset"] = (df["order_month"] - df["cohort_month"]).apply(lambda p: p.n)

# 11월은 반쪽 달(11/16까지)이라 코호트 크기/리텐션이 왜곡됨 -> 코호트 시작월에서 제외
cohort_sizes = first_month[first_month.astype(str) < "2025-11"].value_counts().sort_index()

active = df.drop_duplicates(["회원번호", "order_month"])
retention = active.groupby(["cohort_month", "month_offset"])["회원번호"].nunique().unstack(fill_value=0)
retention = retention.loc[retention.index.astype(str) < "2025-11"]

# 버그 수정(8/25): 코호트 "시작월"이 11월인 경우만 빼는 걸로는 부족함 -> 예를 들어 10월 코호트의
# offset=1(11월 관측)처럼, 시작월은 11월이 아니어도 "관측 대상 월"이 11월(반쪽달)인 셀은
# 전부 분자가 과소집계되어 리텐션이 실제보다 낮게 나옴. 코호트월+offset=관측월이 11월인 셀은
# 전부 NaN 처리(사용자 지적으로 발견 - 10월 코호트 1개월후 재구매율이 유독 낮았던 원인이었음).
observed_month = pd.PeriodIndex(retention.index, freq="M").to_series(index=retention.index)
for offset_col in retention.columns:
    is_nov = (observed_month + offset_col).astype(str) == "2025-11"
    retention.loc[is_nov, offset_col] = np.nan

retention_pct = retention.div(cohort_sizes, axis=0) * 100
retention_pct = retention_pct.round(1)

# 코호트 전체 평균 리텐션 곡선(월별 오프셋 평균, 코호트마다 관측 가능한 offset 수가 다르므로 NaN 제외)
avg_curve = retention_pct.mean(axis=0, skipna=True).round(1)

with open(OUT + "run17_result.txt", "w", encoding="utf-8") as f:
    f.write(f"코호트(첫구매월) 수: {len(cohort_sizes)}개, 코호트별 인원:\n{cohort_sizes.to_string()}\n\n")
    f.write("=== 코호트 x offset 리텐션율(%) 매트릭스 ===\n")
    f.write(retention_pct.to_string())
    f.write("\n\n=== offset별 평균 리텐션율(%) (전 코호트 평균) ===\n")
    f.write(avg_curve.to_string())

retention_pct.to_csv(BASE + "run17_cohort_retention_matrix.csv", encoding="utf-8-sig")
print("저장 완료: run17_cohort_retention_matrix.csv, run17_result.txt")
