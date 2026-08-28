# -*- coding: utf-8 -*-
"""
run25: 타겟팅 전략별 ROI 시뮬레이션 — "그래서 돈으로 얼마인가"

run11의 Tier1–3 리스트가 무차별 발송·무작위 발송 대비 예산 효율에서 얼마나 나은지를
같은 가정 위에서 비교한다. 모델 점수를 실제 캠페인 손익으로 환산하는 단계라
**가정을 전부 명시**하고, 가정에 민감한 부분(접촉 단가)은 손익분기점으로 함께 보고한다.

가정
  A1. 기대 증분 전환수 = 해당 그룹 고객들의 개인별 Uplift 점수(ITE 추정치) 합
      — 로지스틱(run01b)/RandomForest(run08b) 두 모델로 각각 계산해 범위로 보고
  A2. 증분 전환 1건당 기대 매출 = Holdout 재구매 고객의 14일 실측 평균 매출
  A3. 접촉 단가(쿠폰·발송비) = 시나리오 파라미터(기본 3,000원), 손익분기 단가도 함께 산출
  A4. 매출 관점 검증은 run06(매출 Uplift 회귀)의 개인별 추정치 합으로 별도 계산
      — A1×A2(전환 기반)와 서로 독립적인 두 번째 추정선

한계: Treatment가 무작위 배정이 아니라 관측된 자기선택이므로(run23 DML의 평균효과 95% CI는
0을 포함), 아래 수치는 "실험으로 보증된 증분"이 아니라 **같은 가정 위에서 전략끼리 비교하는
상대 시뮬레이션**이다. 절대 금액을 캠페인 KPI로 그대로 약속하는 데 쓰지 않는다.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")

sys.stdout.reconfigure(encoding="utf-8")  # 콘솔이 cp949여도 en dash 출력이 깨지지 않게

CONTACT_COST = 3000  # A3. 1인당 접촉 단가(원)
N_RANDOM_TRIALS = 2000

snap = pd.read_csv(os.path.join(DATA, "snapshot_holdout.csv"), encoding="utf-8-sig")
snap = snap[snap["include_in_uplift_model"] == 1]
REV_PER_CONV = snap.loc[snap["y_repurchase_14d"] == 1, "y_revenue_14d"].mean()  # A2

tgt = pd.read_csv(os.path.join(DATA, "run11_final_targeting_list.csv"), encoding="utf-8-sig")
rev = pd.read_csv(os.path.join(DATA, "run06_revenue_uplift_scores.csv"), encoding="utf-8-sig")
df = tgt.merge(rev[["회원번호", "uplift_revenue_rf"]], on="회원번호", how="left")

t1 = df["targeting_tier"] == "Tier1_최우선"
t2 = df["targeting_tier"].str.startswith("Tier2")
t3 = df["targeting_tier"] == "Tier3_보조신호"

rng = np.random.default_rng(11)


def evaluate(name, mask_or_idx, note=""):
    g = df.loc[mask_or_idx]
    n = len(g)
    conv_lo, conv_hi = g["uplift_logit"].sum(), g["uplift_rf"].sum()
    cost = n * CONTACT_COST
    rev_conv_lo, rev_conv_hi = conv_lo * REV_PER_CONV, conv_hi * REV_PER_CONV
    rev_direct = g["uplift_revenue_rf"].sum()  # A4
    return {
        "전략": name,
        "접촉 인원": n,
        "접촉 비용(원)": int(cost),
        "기대 증분전환(명)": f"{conv_lo:.1f}–{conv_hi:.1f}",
        "1인당 증분전환율": f"{conv_lo / n * 100:.2f}–{conv_hi / n * 100:.2f}%p",
        "기대 증분매출 A1×A2(원)": f"{int(rev_conv_lo):,}–{int(rev_conv_hi):,}",
        "기대 증분매출 run06직접(원)": f"{int(rev_direct):,}",
        "ROI(A1×A2 기준)": f"{rev_conv_lo / cost:.1f}–{rev_conv_hi / cost:.1f}배",
        "손익분기 접촉단가(원)": f"{int(rev_conv_lo / n):,}–{int(rev_conv_hi / n):,}",
        "비고": note,
    }


rows = [
    evaluate("① 전체 비구독자 무차별 발송", df.index, "현행 방식(baseline)"),
    evaluate("③ Tier1 최우선 418명", t1, "이중검증 ∩ 매출리스크 없음"),
    evaluate("④ Tier1+Tier3 2,376명", t1 | t3, "보조신호까지 확장"),
    evaluate("⑤ Tier2 95명(제외 대상)", t2, "구독 유도 시 매출 훼손 구간"),
]

# ② 무작위 418명 — 시드 고정 2,000회 평균(우연 수준의 기준선)
k = int(t1.sum())
lo = np.mean([df["uplift_logit"].to_numpy()[rng.choice(len(df), k, replace=False)].sum() for _ in range(N_RANDOM_TRIALS)])
hi = np.mean([df["uplift_rf"].to_numpy()[rng.choice(len(df), k, replace=False)].sum() for _ in range(N_RANDOM_TRIALS)])
cost = k * CONTACT_COST
rows.insert(1, {
    "전략": f"② 무작위 {k}명(대조군)",
    "접촉 인원": k,
    "접촉 비용(원)": int(cost),
    "기대 증분전환(명)": f"{lo:.1f}–{hi:.1f}",
    "1인당 증분전환율": f"{lo / k * 100:.2f}–{hi / k * 100:.2f}%p",
    "기대 증분매출 A1×A2(원)": f"{int(lo * REV_PER_CONV):,}–{int(hi * REV_PER_CONV):,}",
    "기대 증분매출 run06직접(원)": f"{int(df['uplift_revenue_rf'].mean() * k):,}",
    "ROI(A1×A2 기준)": f"{lo * REV_PER_CONV / cost:.1f}–{hi * REV_PER_CONV / cost:.1f}배",
    "손익분기 접촉단가(원)": f"{int(lo * REV_PER_CONV / k):,}–{int(hi * REV_PER_CONV / k):,}",
    "비고": f"{N_RANDOM_TRIALS}회 복원 없는 무작위추출 평균",
})

out = pd.DataFrame(rows)
out.to_csv(os.path.join(DATA, "run25_roi_simulation.csv"), index=False, encoding="utf-8-sig")

print(f"[가정] 증분 전환 1건당 기대매출 = {REV_PER_CONV:,.0f}원 (Holdout 재구매자 14일 평균)")
print(f"[가정] 접촉 단가 = {CONTACT_COST:,}원/인\n")
print(out.to_string(index=False))

eff_lo = df.loc[t1, "uplift_logit"].mean() / df["uplift_logit"].mean()
eff_hi = df.loc[t1, "uplift_rf"].mean() / df["uplift_rf"].mean()
saved = (len(df) - k) * CONTACT_COST
print(f"\n[요약] Tier1의 1인당 기대 Uplift는 전체 무차별 발송 대비 {min(eff_lo, eff_hi):.1f}–{max(eff_lo, eff_hi):.1f}배")
print(f"[요약] 접촉 인원 {len(df):,}명 → {k}명(-{(1 - k / len(df)) * 100:.1f}%), 접촉비용 {saved:,}원 절감")
print(f"[요약] Tier2 {int(t2.sum())}명을 구독 유도에서 제외해 회피하는 기대 매출 훼손: "
      f"{abs(int(df.loc[t2, 'uplift_revenue_rf'].sum())):,}원")
