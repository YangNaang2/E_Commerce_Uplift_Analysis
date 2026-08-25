# FRESH.DATA — Uplift 기반 프로모션 타겟팅 분석

친환경 신선식품 이커머스 기업(가상 케이스, FRESH.DATA)의 매출 하락을 데이터로 진단하고, **Uplift 모델링**으로 "프로모션을 줘야 실제로 반응하는 고객"만 골라내는 타겟팅 전략을 설계한 팀 프로젝트입니다.

## 배경 & 문제의식

2024년 분기 매출 성장률이 13→15→25→27%로 정점을 찍은 직후, 금리 인상과 원자재(식자재) 가격 급등으로 월매출이 꺾였습니다. 친환경·유기농 수요 자체는 견조하다고 보고, "모든 고객에게 같은 프로모션을 뿌리는" 방식 대신 **개인별 Uplift(순증효과) 점수**로 타겟을 좁히는 전략을 제안합니다. 이미 어차피 구독·구매했을 고객(Sure Thing)에게 쓰는 예산은 낭비이므로, 프로모션이 있었을 때와 없었을 때의 차이가 실제로 큰 사람만 골라내는 것이 핵심입니다.

## 접근 방법

- **Treatment**: 멤버십(정기배송) 구독 여부
- **Y**: 기준일 이후 14일 내 재구매 여부
- **메타러너**: T-learner(구독자/비구독자 각각 독립 모델 학습 후 예측값 차이로 Uplift 계산)를 메인으로 채택하고, X-learner로 교차검증. S-learner는 Treatment 신호가 약할 때 Sure Thing만 찾아내는 함정(Sure Thing 함정)이 있어 채택하지 않음
- **대표 모델 2종**: 로지스틱회귀(L1 정규화, 해석 쉬운 선형 모델)와 RandomForest(비선형 상호작용을 잡는 앙상블 모델) — 두 모델이 동시에 지목한 고객만 신뢰하는 이중검증 방식으로 최종 타겟팅 리스트 산출
- **4대 가설(H1~H4)**: 가격 민감도(구매금액 구간), 지역×배송조건, 계절성 상품, 연령×성별 세그먼트별로 프로모션 반응 차이를 검증(모두 뚜렷한 세그먼트 차이는 기각되어, "세그먼트로 나누기"보다 "개인별 점수로 타겟팅"이 더 타당하다는 결론으로 이어짐)

자세한 방법론과 실측 수치는 [`docs/모델링_통합결과_H1_H4.md`](docs/모델링_통합결과_H1_H4.md), [`docs/리더보드_run10.md`](docs/리더보드_run10.md)를 참고하세요.

## 리포지토리 구조

```
docs/        프로젝트 정의서·EDA 인사이트·모델링 통합결과 등 핵심 문서
reports/     발표용 리포트/대시보드 (브라우저에서 바로 열리는 HTML 산출물)
notebooks/   제출용 Jupyter Notebook
src/         전처리·모델링 파이프라인 스크립트 (run01~run22)
data/        전처리 결과·모델 출력 CSV + 원본/병합 데이터(zip)
dashboard/   FastAPI + Streamlit 인터랙티브 대시보드
```

### `src/` — 파이프라인 실행 순서

1. `merge_and_enrich.py` — 원본 3테이블 병합 + 외부변수(계절·금리·소비자심리지수 등) 결합
2. `preprocess_pipeline.py` — 정제·파생변수(RFM, 가격구간, Y 정의 등)·Train/Holdout 시간 분할
3. `run01b_kitchen_sink_l1.py`, `run08b_uplift_forest_gridsearch.py` — 대표 T-learner 2종(로지스틱 L1 / RandomForest, GridSearchCV 튜닝)
4. `run22_xlearner.py` — X-learner 교차검증
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

- 로지스틱·RandomForest 두 모델이 동시에 Persuadable로 지목한 고객은 무작위 기대치 대비 뚜렷하게 많았고(팀 4파트 만장일치 그룹도 무작위 기대치의 12~34배), X-learner로 다시 계산해도 T-learner와 순위상관 0.80으로 재현됨 — 메타러너 구조를 바꿔도 핵심 발견이 유지됨을 확인
- 재구매 확률 Uplift는 양수인데 매출 Uplift는 고가군에서 오히려 음수 — "구독은 재구매를 늘리지만 건당 매출은 줄일 수 있다"는 점을 반영해 최종 타겟팅 리스트에 매출 리스크 등급(Tier)을 함께 부여
