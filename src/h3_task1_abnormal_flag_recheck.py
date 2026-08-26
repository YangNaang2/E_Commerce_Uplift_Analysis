# -*- coding: utf-8 -*-
"""
H3 Task 1 재검증 — 이상거래계정 플래그 교체 후 회귀 결론 재확인
- 기존에는 merged_master_enriched.csv에서 회원별 구매빈도 상위 1%(임계 479건)로
  abnormal_account_flag를 프록시 계산했음. snapshot_train.csv의 공식 abnormal_account_flag로
  교체해서, H3 핵심 회귀(구독여부×제철상품구매여부 월별 교호항, 매출) 결론이 달라지는지 재확인.
- 원 회귀식(H3_명절_모델링검증.md 2절): log(구매금액) ~ 구독여부*제철상품구매여부 + 주말여부 + region_tier
  여기에 abnormal_account_flag(공식값)를 통제변수로 추가해 월별로 재적합.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"

m = pd.read_csv(BASE + "merged_master_enriched.csv", encoding="utf-8-sig",
                 usecols=["회원번호", "주문일시", "구매금액", "주문취소여부", "구독여부",
                          "region_tier", "주말여부", "제철상품구매여부"])
official_flag = pd.read_csv(BASE + "snapshot_train.csv", encoding="utf-8-sig",
                             usecols=["회원번호", "abnormal_account_flag"])

print(f"[로드] merged_master_enriched: {len(m):,}행, snapshot_train(공식플래그): {len(official_flag):,}명")

# --- Task 1 지시대로: 구독여부 NaN 제외, 취소건 제외 ---
# 주문취소여부는 정상주문=NaN, 취소주문="주문취소" 문자열로 인코딩됨(확인 완료)
m = m[m["구독여부"].notna()].copy()
m = m[m["주문취소여부"].isna()]

m["주문일시"] = pd.to_datetime(m["주문일시"])
m["month"] = m["주문일시"].dt.month
m["구독_bin"] = m["구독여부"].astype(str).isin(["True", "1", "1.0"]).astype(int)
m["log_구매금액"] = np.log(m["구매금액"].clip(lower=1))

# --- 공식 abnormal_account_flag 회원번호로 매칭 ---
before = len(m)
m = m.merge(official_flag, on="회원번호", how="left")
missing_flag = m["abnormal_account_flag"].isna().sum()
print(f"[매칭] {before:,}행 중 공식 플래그 결측(스냅샷 train에 없는 회원): {missing_flag:,}행 "
      f"({missing_flag/before*100:.2f}%) — 이 행은 회귀에서 제외됨(결측 처리)")
print(f"[최종 분석대상] {m['abnormal_account_flag'].notna().sum():,}행 "
      f"(구독자 {(m['구독_bin']==1).sum():,} / 비구독자 {(m['구독_bin']==0).sum():,})")

results = []
for mon in sorted(m["month"].dropna().unique()):
    sub = m[m["month"] == mon].dropna(subset=["abnormal_account_flag", "region_tier"])
    n = len(sub)
    try:
        model = smf.ols(
            "log_구매금액 ~ 구독_bin * 제철상품구매여부 + 주말여부 + C(region_tier) + abnormal_account_flag",
            data=sub,
        ).fit()
        coef = model.params.get("구독_bin:제철상품구매여부", np.nan)
        ci = model.conf_int().loc["구독_bin:제철상품구매여부"] if "구독_bin:제철상품구매여부" in model.params.index else (np.nan, np.nan)
        pval = model.pvalues.get("구독_bin:제철상품구매여부", np.nan)
        results.append({
            "월": int(mon), "n": n,
            "교호항(%)": (np.exp(coef) - 1) * 100,
            "CI_low(%)": (np.exp(ci[0]) - 1) * 100,
            "CI_high(%)": (np.exp(ci[1]) - 1) * 100,
            "p값": pval,
        })
    except Exception as e:
        results.append({"월": int(mon), "n": n, "교호항(%)": np.nan, "CI_low(%)": np.nan, "CI_high(%)": np.nan, "p값": np.nan})
        print(f"[경고] {mon}월 회귀 실패: {e}")

res_df = pd.DataFrame(results)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print("\n=== H3 월별 교호항 (공식 abnormal_account_flag 통제, Task 1 재검증) ===")
print(res_df.to_string(index=False))

sig = res_df[res_df["p값"] < 0.05]
print(f"\n유의(p<0.05)한 달: {sig['월'].tolist()}")

res_df.to_csv(BASE + "h3_task1_abnormal_flag_recheck_results.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: h3_task1_abnormal_flag_recheck_results.csv")
