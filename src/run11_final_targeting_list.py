# -*- coding: utf-8 -*-
"""
run11: 최종 타겟팅 리스트 — "그래서 누구한테 뭘 하라는 건데"에 대한 실행 가능한 산출물

**8/24 재설계**: 기존 버전은 A/B/C/D "4파트 만장일치"(run09) 기반이었으나, B/C/D 개인별 점수가
실제 팀원 코드가 아니라 Claude 재현본이라 만장일치 인원이 인위적으로 부풀려지는 문제가 확인됨
(방법론이 사실상 같은 4개 변형을 비교한 것이라 서로 과도하게 닮아있음). 발표에서 캐비엇 없이
쓸 수 있는 숫자를 만들기 위해, **100% 실제 데이터로 학습된 A 본인의 두 모델(run01b 로지스틱 /
run08b RandomForest, 둘 다 10/26 Holdout 기준)의 교집합**으로 타겟팅 로직을 다시 짠다.
- 근거1(run01b+run08b): 같은 사람이 만든 로지스틱과 RF는 원래 거의 무상관(A_logit↔A_rf 상관 0.089,
  `모델링_통합결과_H1_H4.md` 5절)이므로, 두 모델이 "동시에" 상위20%로 지목하는 고객은 우연
  (기대 4%)보다 신호가 강한 이중검증 그룹으로 해석 가능
- 근거2(run06): 고가군(Q4)은 재구매 확률 Uplift는 양수여도 매출 Uplift가 강하게 음수 -> 구독 유도의 리스크 구간
- run09(4파트 백테스팅)는 참고용 강건성 체크로만 별도 보고(이 스크립트의 최종 타겟팅 리스트에는 미반영)
"""
import numpy as np
import pandas as pd

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"

logit = pd.read_csv(BASE + "run01b_uplift_scores_holdout.csv", encoding="utf-8-sig")
rf = pd.read_csv(BASE + "run08b_uplift_forest_scores.csv", encoding="utf-8-sig")
rev = pd.read_csv(BASE + "run06_revenue_uplift_scores.csv", encoding="utf-8-sig")[["회원번호", "uplift_revenue_rf"]]

target = logit[logit["treatment_h4"] == 0][["회원번호", "가격구간", "uplift"]].rename(columns={"uplift": "uplift_logit"})
target = target.merge(
    rf[rf["treatment_h4"] == 0][["회원번호", "uplift_forest"]].rename(columns={"uplift_forest": "uplift_rf"}),
    on="회원번호", how="inner",
)
target = target.merge(rev, on="회원번호", how="left")

print(f"타겟 후보(비구독자, 두 모델 공통 채점): {len(target):,}명")

TOP_PCT = 0.20
n_top = int(len(target) * TOP_PCT)

logit_top = set(target.nlargest(n_top, "uplift_logit")["회원번호"])
rf_top = set(target.nlargest(n_top, "uplift_rf")["회원번호"])
both = logit_top & rf_top
either_only = (logit_top | rf_top) - both

expected_random = len(target) * (TOP_PCT ** 2)
print(f"\n로지스틱 상위20%: {len(logit_top):,}명 / RF 상위20%: {len(rf_top):,}명")
print(f"두 모델 동시 상위20%(이중검증): {len(both):,}명 (무작위 기대치 {expected_random:.0f}명, "
      f"{len(both)/expected_random:.1f}배)")
print(f"둘 중 하나만 상위20%: {len(either_only):,}명")

target["double_verified"] = target["회원번호"].isin(both)
target["single_verified"] = target["회원번호"].isin(either_only)

# --- 매출 리스크 플래그: run06 RandomForest 매출Uplift가 음수인 고가군(Q4) ---
target["revenue_risk_flag"] = (target["가격구간"] == "Q4_고가") & (target["uplift_revenue_rf"] < 0)


def tier(row):
    if row["double_verified"] and not row["revenue_risk_flag"]:
        return "Tier1_최우선"
    if row["double_verified"] and row["revenue_risk_flag"]:
        return "Tier2_신호강함_매출리스크주의"
    if row["single_verified"] and not row["revenue_risk_flag"]:
        return "Tier3_보조신호"
    return "비타겟"


target["targeting_tier"] = target.apply(tier, axis=1)

print("\n=== 타겟팅 티어별 인원 ===")
print(target["targeting_tier"].value_counts())

out_cols = ["회원번호", "가격구간", "uplift_logit", "uplift_rf", "double_verified", "single_verified",
            "revenue_risk_flag", "targeting_tier"]
final = target[out_cols].sort_values(
    ["targeting_tier", "uplift_rf"], ascending=[True, False]
)
final.to_csv(BASE + "run11_final_targeting_list.csv", index=False, encoding="utf-8-sig")

print(f"\n[Tier1_최우선] {sum(target['targeting_tier']=='Tier1_최우선')}명 — 로지스틱+RF 두 모델이 동시에 상위20%로 지목 AND 고가군 매출리스크 아님 -> 즉시 타겟팅 권고")
print(f"[Tier2_신호강함_매출리스크주의] {sum(target['targeting_tier']=='Tier2_신호강함_매출리스크주의')}명 — 재구매 신호는 이중검증됐지만 Q4+매출Uplift음수 -> 구독보다 다른 오퍼(대량구매 혜택 등) 검토")
print(f"[Tier3_보조신호] {sum(target['targeting_tier']=='Tier3_보조신호')}명 — 로지스틱·RF 둘 중 한 모델만 상위20%, 참고용")

print("\n저장 완료: run11_final_targeting_list.csv")
