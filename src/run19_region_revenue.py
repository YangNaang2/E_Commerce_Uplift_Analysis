# -*- coding: utf-8 -*-
"""
run19: 지역별 매출 분포 [추가분석, H1~H4와 독립]

기존 H2/run15 분석은 region_tier(3단계 배송권역 가정)로만 지역을 봤음. 이번엔 실제
주소지_정규화(시/도 17개) 단위로 매출·객단가·회원수를 집계해 더 세밀한 지역 그림을 제공.
"""
import pandas as pd

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
OUT = "C:/Users/aidan/AppData/Local/Temp/claude/C--Users-aidan-OneDrive-----------/526a44f2-eab0-428f-be6f-d07d153f63c4/scratchpad/"

df = pd.read_csv(BASE + "merged_master_enriched.csv", encoding="utf-8-sig",
                  usecols=["회원번호", "구매금액", "주문일시", "is_cancelled", "region_tier", "주소지_정규화"])
not_cancelled = df[df["is_cancelled"] == 0]

summary = not_cancelled.groupby("주소지_정규화").agg(
    회원수=("회원번호", "nunique"),
    총매출=("구매금액", "sum"),
    주문건수=("구매금액", "size"),
).reset_index()
summary["AOV"] = (summary["총매출"] / summary["주문건수"]).round(0)
summary["회원당매출"] = (summary["총매출"] / summary["회원수"]).round(0)
summary["매출비중(%)"] = (summary["총매출"] / summary["총매출"].sum() * 100).round(1)
summary = summary.sort_values("총매출", ascending=False)

with open(OUT + "run19_result.txt", "w", encoding="utf-8") as f:
    f.write("=== 시/도별 매출 분포 (매출 내림차순) ===\n")
    f.write(summary.to_string(index=False))
    f.write(f"\n\n상위 5개 지역 매출 비중 합: {summary.head(5)['매출비중(%)'].sum():.1f}%")

summary.to_csv(BASE + "run19_region_revenue.csv", index=False, encoding="utf-8-sig")
print("저장 완료: run19_region_revenue.csv, run19_result.txt")
