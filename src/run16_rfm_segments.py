# -*- coding: utf-8 -*-
"""
run16: RFM 고객 세그먼트 지도 [추가분석, H1~H4와 독립]

snapshot_train.csv에 이미 있는 pre-period RFM(frequency/monetary/recency_days)을
분위(1~5) 스코어로 바꾼 뒤, 널리 쓰이는 RFM 세그먼트 룰로 명명된 등급을 부여함.
"등급별 인원·매출기여도"를 한 장의 표/그림으로 보여주는 게 목적(신규 계산 없이 기존
RFM 값을 재활용).
"""
import pandas as pd

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
OUT = "C:/Users/aidan/AppData/Local/Temp/claude/C--Users-aidan-OneDrive-----------/526a44f2-eab0-428f-be6f-d07d153f63c4/scratchpad/"

df = pd.read_csv(BASE + "snapshot_train.csv", encoding="utf-8-sig",
                  usecols=["회원번호", "frequency", "monetary", "recency_days", "label_observable_flag"])
df = df[df["label_observable_flag"] == 1].copy()
print(f"대상: {len(df):,}명")

# recency는 작을수록 좋음 -> 역순으로 스코어링
df["R"] = pd.qcut(df["recency_days"].rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)
df["F"] = pd.qcut(df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
df["M"] = pd.qcut(df["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)


def segment(row):
    r, f, m = row["R"], row["F"], row["M"]
    if r >= 4 and f >= 4 and m >= 4:
        return "1.최우수고객"
    if f >= 4 and m >= 3:
        return "2.충성고객"
    if r >= 4 and f <= 2:
        return "3.신규유망고객"
    if r <= 2 and f >= 3:
        return "4.이탈위험고객"
    if r <= 2 and f <= 2:
        return "5.휴면고객"
    return "6.일반고객"


df["rfm_segment"] = df.apply(segment, axis=1)

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

df[["회원번호", "R", "F", "M", "rfm_segment"]].to_csv(
    BASE + "run16_rfm_segments.csv", index=False, encoding="utf-8-sig"
)
print("저장 완료: run16_rfm_segments.csv, run16_result.txt")
