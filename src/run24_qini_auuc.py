# -*- coding: utf-8 -*-
"""
run24: Qini / AUUC — 업리프트 모델 전용 랭킹 평가지표

지금까지의 평가는 (1) 부트스트랩 평균 Uplift 신뢰구간, (2) 모델 간 순위상관·상위20% 겹침,
(3) 각 arm의 Holdout AUC 세 가지였다. 이건 "추정치가 안정적인가 / 모델끼리 합의하는가"는
말해주지만, **"이 점수 순서대로 타겟팅했을 때 실제로 증분 전환이 더 많이 잡히는가"**라는
업리프트 모델 본연의 질문에는 답하지 않는다. 그 질문에 답하는 지표가 Qini/AUUC다.

정의(Radcliffe 2007; Gutierrez & Gerardy 2017):
  점수 내림차순으로 상위 k명을 타겟팅했다고 가정할 때
    Qini(k) = Y_t(k) - Y_c(k) * N_t(k)/N_c(k)
      Y_t(k)/Y_c(k) = 상위 k명 중 처치군/대조군의 전환자 수
      N_t(k)/N_c(k) = 상위 k명 중 처치군/대조군의 인원수
  = "상위 k명을 처치했을 때 대조군 대비 추가로 얻는 전환 수"의 불편추정.
  Qini 계수 = (Qini 곡선 아래 면적 - 무작위 대각선 아래 면적) / 전체 인원  (높을수록 좋음)
  AUUC     = 정규화 없이 곡선 아래 면적을 인원수로 나눈 값 (여기서는 두 값을 모두 리포트)

**해석상 주의(정직하게 명시)**: 우리 Treatment(구독여부)는 무작위 배정이 아니라 고객의
자기선택이다. Qini/AUUC는 원래 RCT를 가정한 지표이므로, 여기서 나오는 값은 "무작위 실험에서
검증된 증분"이 아니라 **관측 데이터 위에서의 랭킹 품질 비교치**로만 읽어야 한다. 모델 간
상대 비교(어느 점수가 더 잘 정렬하는가)에는 유효하지만, 절대 수치를 그대로 캠페인 증분으로
약속하는 데는 쓰지 않는다. (같은 이유로 run23 DML의 평균효과 CI가 0을 포함한다는 점도 병기.)

입력: 리포지토리 data/ 폴더의 스냅샷·점수 CSV (별도 전처리 재실행 없이 바로 실행 가능)
출력: data/run24_qini_auuc_summary.csv, assets/qini_curve.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")
ASSETS = os.path.join(REPO, "assets")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load(name, cols=None):
    df = pd.read_csv(os.path.join(DATA, name), encoding="utf-8-sig")
    return df[cols] if cols else df


snap = load("snapshot_holdout.csv")
snap = snap[snap["include_in_uplift_model"] == 1]
base = snap[["회원번호", "treatment_h4", "y_repurchase_14d", "y_revenue_14d"]].copy()

MODELS = [
    ("T-learner 로지스틱(run01b)", "run01b_uplift_scores_holdout.csv", "uplift"),
    ("T-learner RandomForest(run08b)", "run08b_uplift_forest_scores.csv", "uplift_forest"),
    ("X-learner(run22)", "run22_xlearner_scores_holdout.csv", "uplift_xlearner"),
    ("DML CausalForest(run23)", "run23_dml_scores_holdout.csv", "uplift_dml"),
]


def qini_curve(score, treat, y):
    """점수 내림차순 누적 Qini 곡선. 반환: (x=타겟팅 비율, q=누적 증분 전환수)"""
    order = np.argsort(-score, kind="mergesort")
    t, yy = treat[order], y[order]
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(yy * t), np.cumsum(yy * (1 - t))
    ratio = np.divide(nt, nc, out=np.zeros_like(nt, dtype=float), where=nc > 0)
    q = yt - yc * ratio
    n = len(score)
    return np.arange(1, n + 1) / n, q


def coefficients(x, q):
    """Qini 계수(무작위 대비 초과면적)와 AUUC(곡선 아래 면적), 둘 다 인원수로 정규화."""
    n = len(x)
    auuc = np.trapezoid(q, x) / n * n  # 곡선 아래 면적(전환수 x 비율 단위)
    rand = q[-1] * 0.5                 # 무작위 타겟팅 대각선 아래 면적
    return auuc - rand, auuc


rows, curves = [], []
RNG = np.random.default_rng(7)
N_BOOT = 400

for label, fname, col in MODELS:
    sc = load(fname, ["회원번호", col])
    d = base.merge(sc, on="회원번호", how="inner").dropna(subset=[col])
    score = d[col].to_numpy(float)
    treat = d["treatment_h4"].to_numpy(float)
    y = d["y_repurchase_14d"].to_numpy(float)
    n = len(d)
    k10, k20 = int(n * 0.1), int(n * 0.2)

    x, q = qini_curve(score, treat, y)
    qini, auuc = coefficients(x, q)

    # 부트스트랩 95% CI — Qini는 "처치군 전환수 - 대조군 보정 전환수"의 누적차라
    # 표본 재추출 변동이 크다. 점추정치만 보고하면 과신하게 되므로 CI를 함께 낸다.
    boot = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        bx, bq = qini_curve(score[idx], treat[idx], y[idx])
        boot.append((bq[k10 - 1], bq[k20 - 1], coefficients(bx, bq)[0]))
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)

    rows.append({
        "모델": label,
        "N(holdout)": n,
        "Qini 계수": round(qini, 2),
        "Qini 95% CI": f"[{lo[2]:.2f}, {hi[2]:.2f}]",
        "AUUC": round(auuc, 2),
        "상위10% 증분전환(명)": round(q[k10 - 1], 1),
        "상위10% 95% CI": f"[{lo[0]:.1f}, {hi[0]:.1f}]",
        "상위20% 증분전환(명)": round(q[k20 - 1], 1),
        "상위20% 95% CI": f"[{lo[1]:.1f}, {hi[1]:.1f}]",
        "전체 증분전환(명)": round(q[-1], 1),
        "상위20% 포착률(%)": round(q[k20 - 1] / q[-1] * 100, 1) if q[-1] else np.nan,
    })
    curves.append((label, x, q))

out = pd.DataFrame(rows)
out.to_csv(os.path.join(DATA, "run24_qini_auuc_summary.csv"), index=False, encoding="utf-8-sig")
print(out.to_string(index=False))

COLORS = ["#b3311f", "#2f6b4f", "#3d6fa5", "#8a6d1f"]
fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=130)
for (label, x, q), c in zip(curves, COLORS):
    ax.plot(x * 100, q, label=label, color=c, lw=2)
total = curves[0][2][-1]
ax.plot([0, 100], [0, total], "--", color="#8a9488", lw=1.5, label="무작위 타겟팅(기준선)")
ax.axhline(0, color="#444", lw=0.8)
ax.set_xlabel("타겟팅 비율 (Uplift 점수 상위 %)")
ax.set_ylabel("누적 증분 전환 수 (Qini)")
ax.set_title("Qini 곡선 — 메타러너 4종 비교 (Holdout 10/26 실측)")
ax.legend(frameon=False, fontsize=10.5, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "qini_curve.png"))
print("saved:", os.path.join(ASSETS, "qini_curve.png"))
