# -*- coding: utf-8 -*-
"""
run10: 리더보드 — 지금까지 나온 모든 run의 결과를 CSV에서 다시 계산해 한 표로 정리
(A 파트가 만든 run01/01b/06/07/08b CSV를 원자료로 재검증. B/C/D 결과는 각자 리포트 수치를 그대로 인용)
run08(미튜닝) 대신 run08b(GridSearchCV 튜닝판)를 대표값으로 사용하고, run09 백테스팅 요약
섹션을 추가함. run09(4파트 백테스팅)의 B/C/D는 원본 코드가 아닌 재구성본이라 만장일치 인원이
부풀려지는 문제가 있어, **최종 타겟팅(run11)의 근거는 A 본인의 실제 두 모델(run01b∩run08b)
이중검증으로 사용**한다. 이 파일의 run09 요약 섹션은 "참고용 강건성 체크"로 라벨만 바꾸고
계산 로직은 그대로 유지(정성적 패턴 확인용, 발표에서 근거로 쓰는 숫자는 아래 "A 본인 이중검증"
섹션 것을 사용할 것).
"""
import numpy as np
import pandas as pd

BASE = "C:/Users/aidan/OneDrive/바탕 화면/종합실습/data/processed/"
rng = np.random.default_rng(42)


def boot_ci(vals, n=2000):
    b = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


rows = []

# run01(수동피처 버전)은 run01b로 대체된 뒤 삭제됨 -- 아래 run01b부터 시작

# run01b
df = pd.read_csv(BASE + "run01b_uplift_scores_holdout.csv", encoding="utf-8-sig")
v = df.loc[df.treatment_h4 == 0, "uplift"].values
lo, hi = boot_ci(v)
rows.append(["run01b", "재구매(Y binary)", "로지스틱(L1 자동선택)", f"{v.mean()*100:.2f}%p", f"[{lo*100:.2f}, {hi*100:.2f}]", "★ A파트 대표 모델"])

# run06 (2개 모델)
df = pd.read_csv(BASE + "run06_revenue_uplift_scores.csv", encoding="utf-8-sig")
target = df[df.treatment_h4 == 0]
for col, name in [("uplift_revenue_enet", "ElasticNet"), ("uplift_revenue_rf", "RandomForest(GridSearchCV)")]:
    v = target[col].values
    lo, hi = boot_ci(v)
    rows.append(["run06", "매출(Y=14일 매출액)", name, f"{v.mean():.0f}원", f"[{lo:.0f}, {hi:.0f}]원", "고가군(Q4) 음수 -5,400~5,800원 (두 모델 일치)"])

# run07 (2개 모델, 세그먼트별)
df = pd.read_csv(BASE + "run07_churn_uplift_scores.csv", encoding="utf-8-sig")
target = df[df.treatment_h4 == 0]
for col, name in [("uplift_logit", "로지스틱L1"), ("uplift_rf", "RandomForest(GridSearchCV)")]:
    v = target[col].values
    lo, hi = boot_ci(v)
    rows.append(["run07", "재구매(이탈위험군 슬라이스)", name, f"{v.mean()*100:.2f}%p(전체평균)", f"[{lo*100:.2f}, {hi*100:.2f}]", "⚠️ 결론유보(모델간 방향 불일치)"])

# run08b (GridSearchCV 재튜닝판, run08 대체)
df = pd.read_csv(BASE + "run08b_uplift_forest_scores.csv", encoding="utf-8-sig")
v = df.loc[df.treatment_h4 == 0, "uplift_forest"].values
lo, hi = boot_ci(v)
rows.append(["run08b", "재구매(Y binary)", "RandomForest(GridSearchCV 튜닝)", f"{v.mean()*100:.2f}%p", f"[{lo*100:.2f}, {hi*100:.2f}]", "가격구간 1위가 run01b와 다름(Q3→Q4), 튜닝해도 유지 -> 모델클래스 차이"])

leaderboard = pd.DataFrame(rows, columns=["run", "Y(대상)", "모델", "평균 Uplift", "95% CI", "비고"])
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)
print(leaderboard.to_string(index=False))
leaderboard.to_csv(BASE + "run10_leaderboard.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: run10_leaderboard.csv")

# ================= A 본인 이중검증 (run01b∩run08b, 100% 실제 데이터 - 발표 근거용 1순위) =================
logit_df = pd.read_csv(BASE + "run01b_uplift_scores_holdout.csv", encoding="utf-8-sig")
rf_df = pd.read_csv(BASE + "run08b_uplift_forest_scores.csv", encoding="utf-8-sig")
dv = logit_df[logit_df.treatment_h4 == 0][["회원번호", "uplift"]].merge(
    rf_df[rf_df.treatment_h4 == 0][["회원번호", "uplift_forest"]], on="회원번호", how="inner"
)
n_dv_top = int(len(dv) * 0.20)
dv_logit_top = set(dv.nlargest(n_dv_top, "uplift")["회원번호"])
dv_rf_top = set(dv.nlargest(n_dv_top, "uplift_forest")["회원번호"])
dv_both = dv_logit_top & dv_rf_top
dv_expected = len(dv) * (0.20 ** 2)
print("\n=== A 본인 이중검증 (run01b 로지스틱 ∩ run08b RF, 둘 다 실제 10/26 Holdout 기준) ===")
print(f"타겟 후보 {len(dv):,}명 중 두 모델 동시 상위20%: {len(dv_both):,}명 "
      f"(무작위 기대치 {dv_expected:.0f}명 대비 {len(dv_both)/dv_expected:.1f}배)")
print("-> run11 최종 타겟팅 리스트(Tier1/2/3)의 근거. 캐비엇 없이 인용 가능한 실측 수치")

# ================= run09 백테스팅 요약 (별도 CSV, B/C/D=재구성본 - 참고용 강건성 체크) =================
corr = pd.read_csv(BASE + "run09_spearman_correlation.csv", index_col=0, encoding="utf-8-sig")
jac = pd.read_csv(BASE + "run09_top20pct_jaccard.csv", index_col=0, encoding="utf-8-sig")

rf_cols = ["uplift_A_rf", "uplift_B_rf", "uplift_C_rf", "uplift_D_rf"]
logit_cols = ["uplift_A_logit", "uplift_B_logit", "uplift_C_logit", "uplift_D_logit"]
within_part_corr = {
    p: corr.loc[f"uplift_{p}_logit", f"uplift_{p}_rf"] for p in ["A", "B", "C", "D"]
}
rf_cross_corr = corr.loc[rf_cols, rf_cols].values
rf_cross_avg = rf_cross_corr[np.triu_indices(4, k=1)].mean()

merged = pd.read_csv(BASE + "run09_backtesting_merged_scores.csv", encoding="utf-8-sig")
target = merged[merged["treatment_h4"] == 0]
target_n = len(target)

# 만장일치 인원수는 하드코딩하지 않고 매번 top20%/교집합을 다시 계산
TOP_PCT = 0.20
n_top = int(target_n * TOP_PCT)
rf_top = [set(target.nlargest(n_top, c)["회원번호"]) for c in rf_cols]
logit_top = [set(target.nlargest(n_top, c)["회원번호"]) for c in logit_cols]
rf_consensus_n = len(set.intersection(*rf_top))
logit_consensus_n = len(set.intersection(*logit_top))
expected_random = target_n * (TOP_PCT ** 4)

bt_rows = [
    ["같은 파트 로지스틱↔RF 상관(평균)", f"{np.mean(list(within_part_corr.values())):.3f}", "⚠️B/C/D=재구성본. A/B/C/D 개별: " + ", ".join(f"{p}={v:.3f}" for p, v in within_part_corr.items())],
    ["RF 계열 파트간 상관(평균)", f"{rf_cross_avg:.3f}", "⚠️B/C/D=재구성본. A_rf/B_rf/C_rf/D_rf 6쌍 평균"],
    ["[참고용,과대추정] 4파트 만장일치 상위20% (RF 계열)", f"{rf_consensus_n}명", f"⚠️B/C/D=재구성본이라 부풀려짐. 무작위 기대치 대비 {rf_consensus_n/expected_random:.1f}배" if expected_random else "-"],
    ["[참고용,과대추정] 4파트 만장일치 상위20% (로지스틱 계열)", f"{logit_consensus_n}명", f"⚠️B/C/D=재구성본이라 부풀려짐. 무작위 기대치 대비 {logit_consensus_n/expected_random:.1f}배" if expected_random else "-"],
]
bt_df = pd.DataFrame(bt_rows, columns=["지표", "값", "비고"])
print("\n=== run09 백테스팅 요약 (참고용 강건성 체크 — B/C/D는 재구성본, 발표 근거로 쓰지 말 것) ===")
print(bt_df.to_string(index=False))
bt_df.to_csv(BASE + "run10_run09_backtesting_summary.csv", index=False, encoding="utf-8-sig")
print("\n저장 완료: run10_run09_backtesting_summary.csv")
