# 이커머스데이터 — Uplift 기반 프로모션 타겟팅 분석

제공받은 이커머스 기업의 데이터를 분석하고, **Uplift 모델링**으로 "프로모션을 줘야 실제로 반응하는 고객"을 골라내는 타겟팅 전략을 설계한 팀 프로젝트입니다.

## 배경 & 문제의식

"모든 고객에게 같은 프로모션을 뿌리는" 방식 대신 **개인별 Uplift(순증효과) 점수**로 타겟을 좁히는 전략을 제안합니다. 이미 어차피 구독·구매했을 고객(Sure Thing)에게 쓰는 예산은 낭비이므로, 프로모션이 있었을 때와 없었을 때의 차이가 실제로 큰 사람만 골라내는 것이 핵심입니다.

제공받은 이커머스 회사 데이터와 국내 온라인 시장을 조사해 보았을때 9월에서 10월 넘어갈 때의 하락은 업계 전체가 겪는 하락이었습니다. 이 시기를 위기가 아닌, 오히려 남들보다 먼저 체질을 개선할 기회로 포착했습니다.

<img src="assets/05_매출추이_비교.png" alt="자사 매출 증감률과 온라인쇼핑 시장 증감률(전월대비) 비교 꺾은선 그래프" width="650">

## Uplift 모델링이란?

한 사람에게 "프로모션을 줬을 때(treated)"와 "안 줬을 때(not treated)" 각각 구매할지를 놓고 보면, 고객은 아래 4가지 유형으로 나뉩니다. 이 중 프로모션 예산을 써야 하는 사람은 **Persuadable**(초록) 뿐이고, 오히려 역효과가 나는 **Sleeping Dog**(빨강)은 반드시 제외해야 합니다.

<img src="assets/uplift_quadrant.png" alt="Uplift 4분면 — Buy if treated x Buy if NOT treated 매트릭스" width="420">

| 유형 | 처치 시 구매 | 미처치 시 구매 | 대응 |
|---|---|---|---|
| Persuadable | YES | NO | 타겟 1순위 — 프로모션이 실제로 구매를 만들어냄 |
| Sure Thing | YES | YES | 프로모션 없어도 어차피 구매 — 예산 낭비 |
| Lost Cause | NO | NO | 프로모션을 줘도 안 삼 — 효과 없음 |
| Sleeping Dog | NO | YES | 프로모션이 오히려 구매를 막음 — 역효과, 절대 제외 |

실제 계산은 구독자 그룹과 비구독자 그룹을 각각 독립적으로 모델링(T-learner)해, 두 예측 확률의 차이를 Uplift 점수로 씁니다:

```
Uplift(고객 X) = P(구매 | X, 구독시켰다면) − P(구매 | X, 구독 안 시켰다면)
```

이 점수가 높은 사람일수록 Persuadable에 가깝다고 보고 우선 타겟팅합니다.

## 접근 방법

- **Treatment**: 멤버십(정기배송) 구독 여부
- **Y**: 기준일 이후 14일 내 재구매 여부
- **메타러너**: T-learner(구독자/비구독자 각각 독립 모델 학습 후 예측값 차이로 Uplift 계산)를 메인으로 채택하고, X-learner·DML(Double ML, EconML `CausalForestDML`)로 교차검증. S-learner는 Treatment 신호가 약할 때 Sure Thing만 찾아내는 함정(Sure Thing 함정)이 있어 채택하지 않음
- **대표 모델 2종**: 로지스틱회귀(L1 정규화, 해석 쉬운 선형 모델)와 RandomForest(비선형 상호작용을 잡는 앙상블 모델) — 두 모델이 동시에 지목한 고객만 신뢰하는 이중검증 방식으로 최종 타겟팅 리스트 산출
- **4대 가설(H1~H4)**: 가격 민감도(구매금액 구간), 지역×배송조건, 계절성 상품, 연령×성별 세그먼트별로 프로모션 반응 차이를 검증

자세한 방법론과 실측 수치는 [`docs/모델링_통합결과_H1_H4.md`](docs/모델링_통합결과_H1_H4.md), [`docs/리더보드_run10.md`](docs/리더보드_run10.md)를 참고하세요.

## 리포지토리 구조

```
docs/        프로젝트 정의서·EDA 인사이트·모델링 통합결과 등 핵심 문서
reports/     발표용 리포트/대시보드
notebooks/   제출용 Jupyter Notebook
src/         전처리·모델링 파이프라인 스크립트 (run01~run22)
data/        전처리 결과·모델 출력 CSV + 원본/병합 데이터(zip)
dashboard/   FastAPI + Streamlit 인터랙티브 대시보드
```

### `src/` — 파이프라인 실행 순서

1. `merge_and_enrich.py` — 원본 3테이블 병합 + 외부변수(계절·금리·소비자심리지수 등) 결합
2. `preprocess_pipeline.py` — 정제·파생변수(RFM, 가격구간, Y 정의 등)·Train/Holdout 시간 분할
3. `run01b_kitchen_sink_l1.py`, `run08b_uplift_forest_gridsearch.py` — 대표 T-learner 2종(로지스틱 L1 / RandomForest, GridSearchCV 튜닝)
4. `run22_xlearner.py`, `run23_dml_causalforest.py` — X-learner·DML(CausalForestDML) 교차검증
5. `run06`~`run19` — 매출/이탈/RFM/코호트/연관규칙 등 보조 분석
6. `run09_backtesting_persuadable.py`, `run10_leaderboard.py`, `run11_final_targeting_list.py` — 팀 4인 결과 백테스팅 + 최종 타겟팅 리스트 산출

### `data/`

- `raw_and_merged_data.zip` — 원본 3테이블(`Sales_Data.csv` 등)과 병합본(`merged_master*.csv`)을 압축(원본 약 370MB → 약 48MB). 압축 해제 후 `raw/`, `processed/` 경로에 그대로 두면 위 스크립트들이 그대로 동작합니다.
- `snapshot_train.csv` / `snapshot_holdout.csv` — 실제 모델 학습·검증에 쓰인 스냅샷(시간 기준 분할)
- 그 외 `run*_scores*.csv` — 각 스크립트의 실행 결과(개인별 Uplift 점수, 백테스팅 상관, 최종 타겟팅 리스트 등)

### `dashboard/`

FastAPI 백엔드 + Streamlit 프론트엔드로 만든 인터랙티브 버전입니다. 실행하려면:

```bash
pip install -r dashboard/requirements.txt
python dashboard/backend/main.py        # http://localhost:8000
streamlit run dashboard/frontend/app.py # http://localhost:8501
```

가벼운 정적 버전은 `reports/대시보드_통합본.html`을 그냥 더블클릭해서 열면 됩니다(서버·인터넷 연결 불필요).

## 핵심 결과 요약

- 로지스틱·RandomForest 두 모델이 각각 독립적으로 계산했는데도 둘 다 Persuadable로 지목한 고객 수는 무작위로 겹쳤을 때 기대되는 수보다 훨씬 많았습니다(팀 4명의 모델을 모두 합친 만장일치 그룹은 무작위 기대치의 12~34배). 계산 방식을 T-learner에서 X-learner로 바꿔도 고객 순위가 크게 달라지지 않아(순위상관 0.80), 특정 방법론에서만 우연히 나온 결과가 아님을 확인했습니다.
- "재구매 확률" 기준으로 계산한 Uplift는 양수였지만, "매출액" 기준으로 계산하면 고가 상품 구매군에서는 오히려 음수가 나왔습니다 — 구독을 유도하면 재구매 횟수는 늘어도 건당 결제 금액은 줄어들 수 있다는 뜻입니다. 그래서 최종 타겟팅 리스트에는 "매출 리스크 등급(Tier)"을 함께 표시해, 재구매만 보고 타겟팅했을 때 매출이 줄어들 위험을 미리 걸러낼 수 있게 했습니다.
- Double ML(EconML `CausalForestDML`)로 한 번 더 스트레스 테스트해봐도 개인별 타겟팅 순위는 어느 정도 재현됐습니다(상위 20% 겹침 무작위 대비 1.9배) — 다만 평균 효과를 단정하는 표현은 신중해야 한다는 점을 확인했습니다. 자세한 내용은 아래 "방법론 검증 노트" 참고.

## 최종 타겟팅 리스트 — Tier 1~3

로지스틱(`run01b`)과 RandomForest(`run08b`) 두 모델이 각각 상위 20% Persuadable로 지목한 고객의 **교집합(이중검증)**에, 매출 리스크 필터(고가군 Q4는 매출 Uplift가 음수)를 더해 최종 3단계로 나눴습니다(`run11_final_targeting_list.py`). 타겟 후보(비구독자) 8,276명 기준입니다.

| Tier | 인원 | 정의 | 권장 액션 |
|---|---|---|---|
| **Tier1_최우선** | **418명** | 이중검증 통과 + 매출 리스크 없음 | 구독 프로모션 즉시 타겟팅 |
| **Tier2_매출리스크주의** | **95명** | 이중검증 통과 + 고가군(Q4, 매출 Uplift 음수) | 구독 유도 대신 대량구매 혜택 등 다른 오퍼 검토 |
| **Tier3_보조신호** | **1,958명** | 로지스틱·RF 중 한 모델만 상위 20% | 참고용, 우선순위 낮음 |

Tier1+2(이중검증 통과 513명)는 무작위로 두 모델이 우연히 겹칠 때 기대되는 인원(약 331명)보다 1.6배 많아, 우연이 아닌 신호로 판단했습니다. 원자료는 [`data/run11_final_targeting_list.csv`](data/run11_final_targeting_list.csv), 산출 근거는 [`docs/리더보드_run10.md`](docs/리더보드_run10.md) 1-1절을 참고하세요.

### 대표 모델 2종 진단

T-learner를 구성하는 두 모델 각각의 하이퍼파라미터 튜닝·회귀계수(또는 피처 중요도)·혼동행렬·ROC 곡선입니다. 둘 다 Holdout AUC 0.88 안팎으로 성능은 비슷하지만, 개인별 Uplift 순위는 거의 무상관(0.09)이라 두 모델이 동시에 지목한 고객만 신뢰하는 이중검증 방식을 씁니다.

<img src="assets/model1_logistic_l1.png" alt="모델1 로지스틱 회귀(L1) 진단 그래프 4종" width="800">

<img src="assets/model2_randomforest.png" alt="모델2 RandomForest 진단 그래프 4종" width="800">

## 방법론 검증 노트 — DML까지 돌려본 이유

로지스틱·RandomForest·X-learner 세 가지 방법 모두에서 **"고객마다 반응이 다르고, 그중 일부는 뚜렷하게 반응한다"는 타겟팅의 핵심 전제는 일관되게 재현**됐습니다(상위 20% 겹침 무작위 대비 1.9~3.3배, 세 모델 만장일치는 7.3배). 이걸 한 번 더 스트레스 테스트해보기 위해 Treatment(구독여부)가 무작위 실험이 아니라 고객이 스스로 선택한 값이라는 점을 명시적으로 다루는 Double ML(EconML `CausalForestDML`)까지 적용해봤습니다. 개인별 타겟팅 순위는 이번에도 어느 정도 재현됐지만, "구독이 평균적으로 재구매를 유의하게 늘린다"처럼 평균 효과를 단정하는 문장은 DML 기준 95% 신뢰구간이 [-4.37, 6.10]%p로 나와 조금 더 신중하게 표현하는 게 정확합니다. 최종 타겟팅 전략(위 Tier1~3)은 애초에 평균 효과가 아니라 개인별 순위에 기반하고 있어 이 결론에 좌우되지 않습니다. 자세한 비교는 [`docs/리더보드_run10.md`](docs/리더보드_run10.md) 5-2절을 참고하세요.

## 참고 문헌

- 김주현, 문현실 (2025). 「광고 효율성 제고를 위한 Uplift 모델 기반 모바일 광고 타겟팅 방법」. 한국경영과학회지, 제50권 제1호. [https://doi.org/10.7737/JKORMS.2025.50.1.049](https://doi.org/10.7737/JKORMS.2025.50.1.049)
- 장채연 (2026). 「저차 상호작용 및 네트워크 직교성 제약을 통한 Uplift Modeling 성능 개선 연구」(석사학위논문). 연세대학교 대학원 디지털애널리틱스 융합협동과정.
