# 이커머스 Uplift 대시보드 — 팀 인수인계용

Uplift 모델링 결과(누구에게 프로모션을 뿌려야 하는지)를 보여주는 대시보드 1차 버전입니다. **데이터/로직은 다 연결돼 있고, 디자인·레이아웃을 다듬는 작업이 남아있습니다.**

## 실행 방법

Python 3.10 이상 필요. 처음 한 번만:
```
pip install -r requirements.txt
```

터미널 2개를 띄워서 각각 실행:
```
# 터미널 1 — 백엔드 (데이터 API)
cd backend
uvicorn main:app --reload --port 8000

# 터미널 2 — 프론트엔드 (화면)
cd frontend
streamlit run app.py
```
브라우저에서 `http://localhost:8501` 접속하면 바로 보입니다. 백엔드가 먼저 떠 있어야 프론트가 데이터를 받아옵니다.

## 폴더 구조

| 폴더/파일 | 설명 |
|---|---|
| `원본_초안/` | 처음 만들었던 애니메이션 랜딩페이지 디자인(mock 데이터). **디자인 참고용으로 남겨둠** — 다크테마+민트/코랄/골드 색상 팔레트를 여기서 가져왔음 |
| `backend/main.py` | FastAPI 서버 — 데이터 API 제공. **로직 파일이라 수정 시 주의**, API 엔드포인트 목록은 파일 안 주석 참고 |
| `backend/data_prep.py` | 원본 대용량 CSV에서 대시보드용 요약 데이터를 미리 계산하는 스크립트. 모델 결과가 갱신되면 이거 재실행 |
| `backend/data/dash_*.csv` | 위 스크립트가 만든 요약 데이터(작은 CSV 6개) |
| `frontend/app.py` | **여기가 화면 담당 — 꾸밀 때 주로 건드릴 파일**. Streamlit이라 파이썬 코드로 화면을 만듦(HTML/CSS 아님) |
| `demo_customer_upload.csv` | "고객 CSV 분석" 탭 시연용 샘플 파일 |

## 지금 상태 (기능은 다 됨, 디자인만 기본형)

4개 탭 다 실제 데이터로 동작합니다:
1. **🎯 프로모션 타겟팅** — 누구한테 프로모션을 줘야 효과가 좋은지 (Tier1/2/3 리스트)
2. **🔔 구독자 관리** — 기존 구독자 중 이탈 조짐 보이는 사람 알림 리스트
3. **📊 계절·인구통계별 매출** — 계절/월별 추이, 성별×연령대별 매출·주요 품목
4. **📁 고객 CSV 분석** — CSV 업로드하면 그 고객 분석 결과 표시

## 꾸밀 때 참고

- `frontend/app.py` 맨 위 `st.markdown("""<style>...""")` 부분이 커스텀 CSS입니다. 색상은 `원본_초안/dashboard.html`의 `:root` 변수(`--mint`, `--coral`, `--amber`, `--violet`, `--gold`)와 맞춰뒀으니 톤 유지하려면 여기서 색상 값만 바꾸면 됩니다.
- Streamlit은 `st.columns()`, `st.tabs()`, `st.metric()`, `plotly` 차트 등으로 레이아웃을 짭니다. HTML을 직접 쓰고 싶으면 `st.markdown(..., unsafe_allow_html=True)`로 넣을 수 있습니다(카드 스타일 부분이 이미 이 방식).
- 데이터를 더 예쁘게 보여줄 아이디어(카드 디자인, 애니메이션, 아이콘 등)는 자유롭게 추가해도 됩니다 — API가 주는 데이터 구조(JSON)만 안 깨면 됩니다.
- 백엔드(`backend/main.py`)가 제공하는 API 목록: `/api/kpi`, `/api/targeting`, `/api/subscriber-watch`, `/api/trend/month`, `/api/trend/season`, `/api/segment/summary`, `/api/segment/top-items`, `/api/member/{id}`, `/api/analyze-csv`(POST, CSV 업로드)
- 궁금한 점 있으면 원래 만든 사람(나)한테 물어봐줘.
