# -*- coding: utf-8 -*-
"""
run16: RFM 고객 세그먼트 지도 (H1~H4 Uplift 파이프라인과 독립)

snapshot_train.csv에 이미 있는 pre-period RFM(frequency/monetary/recency_days)을
분위(1~5) 스코어로 바꾼 뒤, R+F+M 합산 점수의 상위/중위/하위 3등분으로 등급을 부여함.
"등급별 인원·매출기여도"를 한 장의 표/그림으로 보여주는 게 목적(신규 계산 없이 기존
RFM 값을 재활용).

R/F/M 개별 조건 분기로 세그먼트를 나누는 방식(예: 최우수/충성/신규유망/이탈위험/휴면/일반
6분류) 대신, R+F+M 합산 점수 기준 3등분(핵심/일반/관리필요)을 채택했다 — VIP 정의나
개별 R/F/M 스코어링에 이미 쓰인 "상위 20%, 다음 20%…" 식 분위 기준 등분과 논리를 통일하고,
조건 분기 없이 합산점수 하나로 설명할 수 있어 해석이 단순해진다는 장점이 있다.
"""
import pandas as pd

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
OUT = BASE

df = pd.read_csv(BASE + "snapshot_train.csv", encoding="utf-8-sig",
                  usecols=["회원번호", "frequency", "monetary", "recency_days", "label_observable_flag"])
df = df[df["label_observable_flag"] == 1].copy()
print(f"대상: {len(df):,}명")

# recency는 작을수록 좋음 -> 역순으로 스코어링
df["R"] = pd.qcut(df["recency_days"].rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)
df["F"] = pd.qcut(df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
df["M"] = pd.qcut(df["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)


df["rfm_score"] = df["R"] + df["F"] + df["M"]  # 3(최저)~15(최고)
df["rfm_segment"] = pd.qcut(
    df["rfm_score"].rank(method="first"), 3, labels=["3.관리필요고객", "2.일반고객", "1.핵심고객"]
).astype(str)

summary = df.groupby("rfm_segment").agg(
    인원=("회원번호", "count"),
    평균recency일=("recency_days", "mean"),
    평균frequency=("frequency", "mean"),
    평균monetary=("monetary", "mean"),
).round(1)
summary["매출기여비중(%)"] = (df.groupby("rfm_segment")["monetary"].sum() / df["monetary"].sum() * 100).round(1)
summary["인원비중(%)"] = (summary["인원"] / summary["인원"].sum() * 100).round(1)
summary = summary.sort_values("인원", ascending=False)

with open(OUT + "run16_result.txt", "w", encoding="utf-8") as f:
    f.write(f"대상 {len(df):,}명\n\n=== RFM 세그먼트 요약 ===\n")
    f.write(summary.to_string())

df[["회원번호", "R", "F", "M", "rfm_score", "rfm_segment"]].to_csv(
    BASE + "run16_rfm_segments.csv", index=False, encoding="utf-8-sig"
)
print("저장 완료: run16_rfm_segments.csv, run16_result.txt")
