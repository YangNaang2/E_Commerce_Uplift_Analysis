# -*- coding: utf-8 -*-
"""
프로모션 추천 서비스 — Streamlit 프론트엔드
'우리가 이커머스데이터 사내 데이터마케팅팀'이라는 톤으로 문구를 쉽게 풀어씀.
백엔드(FastAPI, backend/main.py)가 8000번 포트에 떠 있어야 함:
  cd dashboard/backend && uvicorn main:app --reload --port 8000
실행: cd dashboard/frontend && streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="이커머스데이터 — 프로모션 추천 서비스", page_icon="🥑", layout="wide")

# ---- 팀 초안(dashboard.html)의 다크 + 민트/코랄/골드 팔레트를 이어받은 커스텀 스타일 ----
st.markdown("""
<style>
:root{
  --mint:#1c9a68; --coral:#dd4636; --amber:#c9791a; --violet:#6d4fc4; --gold:#b98d16;
}
.stApp { background-color:#0b0e11; }
h1, h2, h3, h4, p, span, div, label { color:#f5f7fa !important; }
.kpi-card{
  background:rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.09); border-radius:14px;
  padding:18px 20px; margin-bottom:8px;
}
.kpi-card .label{ font-size:12.5px; color:#93a0ab !important; letter-spacing:.05em; }
.kpi-card .value{ font-size:26px; font-weight:700; margin-top:4px; }
.section-title{ font-size:15px; letter-spacing:.08em; color:#93a0ab !important; margin:22px 0 8px; text-transform:uppercase; }
.model-card{
  background:rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.09); border-radius:14px;
  padding:20px 22px; margin-bottom:14px;
}
.model-card .q{ font-size:15px; font-weight:700; color:#f5f7fa !important; margin-bottom:6px; }
.model-card .a{ font-size:13.5px; color:#93a0ab !important; line-height:1.6; }
.badge-mint{ display:inline-block; background:rgba(111,207,151,.15); color:#6fcf97 !important; padding:2px 10px; border-radius:20px; font-size:11.5px; margin-bottom:8px; }

.quad-grid{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:10px 0 24px; }
.quad-card{ border-radius:14px; padding:20px 22px; border:1px solid rgba(255,255,255,.09); background:rgba(255,255,255,.035); }
.quad-card.target{ background:rgba(111,207,151,.10); border-color:rgba(111,207,151,.35); }
.quad-card.avoid{ background:rgba(221,70,54,.10); border-color:rgba(221,70,54,.35); }
.quad-card .qname{ font-size:17px; font-weight:800; color:#f5f7fa !important; margin-bottom:6px; }
.quad-card .qdesc{ font-size:13px; color:#93a0ab !important; line-height:1.55; margin-bottom:12px; min-height:38px; }
.quad-badge{ display:inline-block; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700; }
.quad-badge.target{ background:#1c9a68; color:#0b0e11 !important; }
.quad-badge.avoid{ background:#dd4636; color:#0b0e11 !important; }
.quad-badge.neutral{ background:rgba(255,255,255,.09); color:#93a0ab !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def api_get(path, params=None):
    r = requests.get(f"{API}{path}", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


# ---- 기술 용어를 쉬운 말로 바꾸는 매핑 (원본 데이터 값은 그대로, 화면 표시만 변환) ----
TIER_LABELS = {
    "Tier1_최우선": "⭐ 최우선 추천",
    "Tier2_신호강함_매출리스크주의": "⚠️ 추천 (다른 혜택 권장)",
    "Tier3_보조신호": "추천 (참고)",
    "비타겟": "추천 대상 아님",
}
STATUS_LABELS = {
    "이탈위험(90일+)": "🔴 이탈 위험 (90일 이상 무주문)",
    "관심필요(30~89일)": "🟡 관심 필요 (30~89일 무주문)",
    "Active(<30일)": "🟢 정상 (30일 이내 주문)",
}

try:
    kpi = api_get("/api/kpi")
except Exception as e:
    st.error(f"백엔드(FastAPI, {API})에 연결할 수 없습니다. `uvicorn main:app --port 8000`을 backend 폴더에서 먼저 실행해주세요.\n\n{e}")
    st.stop()

st.title("🥑 프로모션 추천 서비스")
st.caption("이커머스데이터(친환경 신선식품 커머스) 고객 주문 데이터를 분석해, 프로모션을 드리면 효과가 좋을 고객을 찾아드립니다.")

# ================= KPI 카드 =================
c1, c2, c3, c4 = st.columns(4)
kpi_items = [
    (c1, "전체 회원", f"{kpi['총_회원수']:,.0f}명"),
    (c2, "구독 중인 회원", f"{kpi['구독율(%)']:.1f}%"),
    (c3, "평균 재구매율(14일 내)", f"{kpi['전체_재구매율(%)']:.1f}%"),
    (c4, "추천 고객 수 (최우선)", f"{int(kpi['Tier1_최우선_타겟수']):,}명"),
]
for col, label, value in kpi_items:
    col.markdown(f'<div class="kpi-card"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

tab_intro, tab_main, tab_sub, tab_seg, tab_upload = st.tabs([
    "🥑 서비스 소개", "🎯 프로모션 추천 (메인)", "🔔 구독자 관리", "📊 매출 리포트", "📁 고객 조회",
])

# ================= 서비스 소개 =================
with tab_intro:
    st.markdown('<div class="section-title">문제 상황 — 9월을 정점으로 매출이 꺾였습니다</div>', unsafe_allow_html=True)
    month_df_full = pd.DataFrame(api_get("/api/trend/month")).sort_values("월")
    # 11월은 11/16까지만 집계된 반쪽 달이라 이 비교 그래프에서는 제외(하락이 계속되는 것처럼 오인 방지)
    month_df = month_df_full[month_df_full["월"] <= 10]
    # 8/24: 원본 자체에 결측일(10월 3일)이 있어 월합계(총매출)로 비교하면 왜곡됨 -> 일평균매출(관측일 보정)로 비교
    sep_rev = month_df.loc[month_df["월"] == 9, "일평균매출"]
    oct_rev = month_df.loc[month_df["월"] == 10, "일평균매출"]
    our_mom = float((oct_rev.values[0] / sep_rev.values[0] - 1) * 100) if len(sep_rev) and len(oct_rev) else None

    # 국내 온라인 음식료품 카테고리 전년동월대비 성장률(%, 국가데이터처 온라인쇼핑동향 월별 보도자료 기준, 8/24 확보)
    market_food = {1: 9.2, 2: 8.3, 3: 9.8, 4: 9.1, 5: 5.6, 6: 11.0, 7: 12.6, 8: 5.8, 9: 17.7, 10: 4.4}
    market_df = pd.DataFrame({"월": list(market_food.keys()), "증가율": list(market_food.values())}).sort_values("월")

    fig_stack = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("이커머스데이터 일평균 매출(원, 관측일 보정)", "국내 온라인 음식료품 카테고리 성장률(전년동월대비, %)"),
    )
    fig_stack.add_trace(go.Scatter(
        x=month_df["월"], y=month_df["일평균매출"], mode="lines+markers",
        line=dict(color="#6fcf97", width=3), marker=dict(size=7), showlegend=False,
    ), row=1, col=1)
    if len(sep_rev) and len(oct_rev):
        fig_stack.add_trace(go.Scatter(
            x=[9, 10], y=[sep_rev.values[0], oct_rev.values[0]], mode="lines",
            line=dict(color="#dd4636", width=5), showlegend=False,
        ), row=1, col=1)
    fig_stack.add_trace(go.Scatter(
        x=market_df["월"], y=market_df["증가율"], mode="lines+markers",
        line=dict(color="#93a0ab", width=3), marker=dict(size=7), showlegend=False,
    ), row=2, col=1)
    fig_stack.add_trace(go.Scatter(
        x=[9, 10], y=[market_food[9], market_food[10]], mode="lines",
        line=dict(color="#dd4636", width=5), showlegend=False,
    ), row=2, col=1)
    fig_stack.update_xaxes(title_text="월", row=2, col=1)
    fig_stack.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10), height=460,
    )
    st.plotly_chart(fig_stack, use_container_width=True)
    st.caption(
        "위: 이커머스데이터 일평균 매출(원본 자체에 10월 3일 등 결측일이 있어, 월합계 대신 관측일수로 나눈 "
        "일평균으로 비교). 아래: 국내 온라인 음식료품 카테고리 성장률(전년동월대비, 국가데이터처 "
        "온라인쇼핑동향 월별 보도자료). 두 그래프 모두 **9월에서 10월로 넘어가며 같은 시점에 꺾입니다**(빨간 구간). "
        "해외(미국)는 같은 달 온라인 소비가 전년대비 +8.2%로 여전히 성장세였습니다(Adobe Digital Economy Index)."
    )

    if our_mom is not None:
        st.write(
            f"저희 매출은 9월 대비 10월에(일평균 기준) **{our_mom:.1f}%** 꺾였습니다. 처음엔 저희 회사만의 "
            "문제인가 걱정했지만, 확인해보니 **국내 온라인 식품 시장 전체도 같은 시기에 더 큰 폭(-15.9%)으로 "
            "위축**됐다는 걸 알게 됐습니다 — 오히려 저희는 시장 평균보다는 완만하게 흔들린 셈입니다. "
            "반면 해외 이커머스 시장은 같은 기간에도 여전히 성장세를 유지했습니다. 다들 겪는 계절적 흐름이라면, "
            "오히려 **지금이 남들보다 먼저 체질을 개선해서 반등할 기회**라고 판단했고, 그래서 저희 데이터마케팅"
            "(DM)팀이 직접 데이터 분석에 착수했습니다."
        )

    st.markdown('<div class="section-title">저희 DM팀은 무엇을 하나요</div>', unsafe_allow_html=True)
    st.write(
        "모든 고객에게 똑같이 쿠폰이나 구독 프로모션을 뿌리면, 어차피 살 사람에게도 혜택을 주고 "
        "예산이 낭비됩니다. 저희 DM팀은 회원님의 주문 데이터를 분석해서 **\"프로모션을 받으면 실제로 "
        "마음이 움직일 고객\"만 골라냅니다.** 이미 살 사람, 어차피 안 살 사람에게 쓰는 "
        "예산을 줄이고, 반응할 사람에게 집중할 수 있습니다."
    )

    st.markdown('<div class="section-title">우리가 노리는 고객은 정해져 있습니다</div>', unsafe_allow_html=True)
    st.write(
        "구독을 권했을 때 고객의 반응은 4가지 유형으로 나뉩니다. 저희 팀은 이 중 **왼쪽 위 "
        "'Persuadable'만 정확히 찾아내는 것**을 목표로 합니다 — 나머지 세 유형에게 예산을 쓰는 건 "
        "낭비이거나, 심지어 역효과입니다."
    )
    st.markdown("""
    <div class="quad-grid">
      <div class="quad-card target">
        <div class="qname">Persuadable</div>
        <div class="qdesc">구독을 권하면 재구매가 늘어나지만, 안 권하면 그대로인 고객</div>
        <span class="quad-badge target">🎯 타겟 1순위</span>
      </div>
      <div class="quad-card">
        <div class="qname">Sure Thing</div>
        <div class="qdesc">구독 여부와 상관없이 어차피 재구매하는 고객</div>
        <span class="quad-badge neutral">예산 낭비</span>
      </div>
      <div class="quad-card">
        <div class="qname">Lost Cause</div>
        <div class="qdesc">구독을 권해도 재구매하지 않는 고객</div>
        <span class="quad-badge neutral">효과 없음</span>
      </div>
      <div class="quad-card avoid">
        <div class="qname">Sleeping Dog</div>
        <div class="qdesc">구독을 권하면 오히려 재구매가 줄어드는 고객</div>
        <span class="quad-badge avoid">🚫 절대 제외</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("다음 탭(🎯 프로모션 추천)의 추천 리스트는 이 중 Persuadable에 가까운 고객만 골라낸 것입니다.")

    st.markdown('<div class="section-title">어떤 고객이 더 잘 반응하는지, 4가지 관점으로 확인했습니다</div>', unsafe_allow_html=True)
    models = [
        ("💰 가격대로 보면?", "저렴한 걸 사는 고객이 프로모션에 더 크게 반응할 거라 예상했지만, "
         "실제로는 가격대만으로는 뚜렷한 차이가 없었습니다. 다만 같은 가격대 안에서도 사람마다 "
         "반응 정도가 크게 달랐습니다 — 그래서 가격대가 아니라 개인 단위로 봐야 한다는 걸 알게 됐습니다."),
        ("👥 나이·성별로 보면?", "특정 연령대나 성별이 더 잘 반응할 거라 예상했지만, 통계적으로 뚜렷한 "
         "차이는 발견되지 않았습니다."),
        ("🍂 계절·명절로 보면?", "제철 상품이나 명절 시즌에 반응이 다를 거라 예상했지만, 역시 뚜렷한 "
         "차이는 없었습니다. 다만 이 분석 과정에서 \"명절 직후 매출이 63.8% 급감했다\"는 예전 통계가 "
         "실제로는 약 28.5% 감소였다는 것도 새로 확인해서 바로잡았습니다."),
        ("🚚 사는 지역으로 보면?", "새벽배송 가능 지역이든 아니든, 지역·배송 조건에 따른 뚜렷한 반응 "
         "차이는 없었습니다."),
    ]
    mc1, mc2 = st.columns(2)
    for i, (q, a) in enumerate(models):
        target = mc1 if i % 2 == 0 else mc2
        target.markdown(f'<div class="model-card"><div class="q">{q}</div><div class="a">{a}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">그래서 내린 결론</div>', unsafe_allow_html=True)
    st.write(
        "**\"어떤 그룹인가\"가 아니라 \"어떤 사람인가\"가 중요했습니다.** 가격대·나이·성별·지역·계절 "
        "어느 기준으로 나눠도 반응이 뚜렷하게 갈리지 않았기 때문에, 저희는 그룹으로 나누는 대신 "
        "**4가지 분석을 각각 통과해서 공통으로 반응할 것 같다고 나온 고객만 개인 단위로 추천**하는 "
        "방식을 씁니다. 자세한 추천 리스트는 다음 탭(🎯 프로모션 추천)에서 확인하실 수 있습니다."
    )
    with st.expander("🔧 분석 방법이 궁금하다면 (기술적으로 더 자세히)"):
        st.write(
            "- 4개 분석은 각각 **가격구간(H1)·연령×성별(H2)·계절×제철상품(H3)·지역×배송조건(H4)** "
            "가설을 통계 모델(로지스틱회귀·랜덤포레스트 기반 Uplift 모델)로 검증한 결과입니다.\n"
            "- '반응할 고객'은 Uplift 모델링(구독을 유도했을 때와 안 했을 때의 재구매 확률 차이 추정)"
            "으로 계산했고, 로지스틱·랜덤포레스트 두 계열 모델이 서로 다른 고객을 짚는 경향이 있어 "
            "**4개 분석·2개 모델 계열이 공통으로 상위 20%로 꼽은 고객**만 최우선 추천으로 분류했습니다."
        )

# ================= 메인: 프로모션 추천 =================
with tab_main:
    st.markdown('<div class="section-title">지금 프로모션을 드리면 좋을 고객</div>', unsafe_allow_html=True)
    st.write(
        "4가지 분석이 공통으로 \"반응할 것 같다\"고 짚은 고객을 최우선 추천으로 골랐습니다. "
        "다만 평소 고액 구매를 하던 고객은 프로모션을 드리면 재구매는 늘어도 건당 구매액이 줄어드는 "
        "경향이 있어, 이런 경우는 구독 대신 다른 혜택(대량구매 할인 등)을 권해드립니다."
    )

    t1, t2, t3 = st.columns(3)
    t1.metric("⭐ 최우선 추천", f"{int(kpi['Tier1_최우선_타겟수'])}명", help="4가지 분석이 공통으로 짚은 고객 — 매출 감소 위험도 없음")
    t2.metric("⚠️ 추천 (다른 혜택 권장)", f"{int(kpi['Tier2_매출리스크주의_타겟수'])}명", help="반응은 좋을 것 같지만, 평소 고액 구매 고객이라 구독보다 다른 혜택이 나을 수 있음")
    t3.metric("추천 (참고)", f"{int(kpi['Tier3_보조신호_타겟수'])}명", help="보조 지표에서만 짚힌 고객, 참고용")

    # ---- 예상 추가매출 시뮬레이션 (전환율 슬라이더) ----
    st.markdown('<div class="section-title">이 프로모션을 실행하면 매출이 얼마나 늘어날까</div>', unsafe_allow_html=True)
    st.write(
        "위 추천 고객이 실제로 구독 전환에 성공했을 때의 매출 증가분을 미리 계산해뒀습니다. "
        "다만 프로모션을 받은 고객이 실제로 몇 %나 전환할지는 캠페인을 돌려봐야 아는 값이라, "
        "**전환율을 직접 움직여보면서** 예상 매출 증가를 확인하실 수 있습니다."
    )
    tier100 = {
        "Tier1_최우선": kpi.get("Tier1_100퍼센트전환시_14일매출증가(원)", 0),
        "Tier2_신호강함_매출리스크주의": kpi.get("Tier2_100퍼센트전환시_14일매출증가(원)", 0),
        "Tier3_보조신호": kpi.get("Tier3_100퍼센트전환시_14일매출증가(원)", 0),
    }
    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        include = st.multiselect(
            "포함할 추천 등급",
            list(tier100.keys()),
            default=["Tier1_최우선"],
            format_func=lambda x: TIER_LABELS[x],
        )
        conv_rate = st.slider("가정 전환율 (%)", min_value=0, max_value=100, value=20, step=1)
        base_total = sum(tier100[t] for t in include)
        projected = base_total * conv_rate / 100
        annualize = st.checkbox("연간 환산해서 보기 (14일 → 연간, 단순 배수 · 참고용)", value=False)
        shown = projected * (365 / 14) if annualize else projected
        st.metric(
            f"예상 매출 증가 ({'연간 환산' if annualize else '14일 기준'})",
            f"{shown:,.0f}원",
            delta=f"전환율 {conv_rate}%, {'+'.join(TIER_LABELS[t] for t in include) if include else '선택 없음'}",
        )
        if base_total < 0:
            st.warning("선택한 등급의 100% 전환 기준 매출 합계가 마이너스입니다 — 재구매는 늘어도 건당 구매액이 줄어드는 고가군이 섞여 있기 때문입니다.")
    with sim_col2:
        rates = list(range(0, 101, 5))
        values = [base_total * r / 100 * ((365 / 14) if annualize else 1) for r in rates]
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=rates, y=values, mode="lines", line=dict(color="#6fcf97", width=3), showlegend=False))
        fig_sim.add_trace(go.Scatter(
            x=[conv_rate], y=[shown], mode="markers",
            marker=dict(size=14, color="#ffb74d", line=dict(width=2, color="#0b0e11")),
            showlegend=False,
        ))
        fig_sim.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="전환율 (%)", yaxis_title=f"예상 매출 증가 ({'연간' if annualize else '14일'}, 원)",
            margin=dict(t=10, b=10), height=280,
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    tier_options = list(TIER_LABELS.keys())
    tier_filter = st.selectbox("등급 필터", ["전체"] + tier_options, format_func=lambda x: "전체" if x == "전체" else TIER_LABELS[x])
    params = {} if tier_filter == "전체" else {"tier": tier_filter}
    targeting = pd.DataFrame(api_get("/api/targeting", {**params, "limit": 200}))

    if not targeting.empty:
        show = targeting.copy()
        show["추천 등급"] = show["targeting_tier"].map(TIER_LABELS)
        show["예상 반응도"] = (show["uplift_rf"] * 100).round(1).astype(str) + "%p"
        show = show.rename(columns={
            "회원번호": "회원번호", "가격구간": "구매력 구간", "age_band_h4": "연령대",
            "gender_h4": "성별", "region_tier": "배송권역",
        })
        st.dataframe(
            show[["회원번호", "추천 등급", "예상 반응도", "구매력 구간", "연령대", "성별", "배송권역"]],
            use_container_width=True, height=420, hide_index=True,
        )
    st.caption("전체 추천 리스트는 회사 내부 데이터 파일(run11_final_targeting_list.csv)에서 확인할 수 있습니다.")

# ================= 사이드: 구독자 관리 =================
with tab_sub:
    st.markdown('<div class="section-title">기존 구독자 관리</div>', unsafe_allow_html=True)
    st.write(
        "위의 추천 리스트는 아직 구독하지 않은 고객 중에서 찾은 것입니다. 이미 구독 중인 고객은 "
        "별도로, **평소보다 주문이 뜸해진 분들에게 알림을 보내드릴 수 있도록** 목록을 정리했습니다."
    )
    status_options = list(STATUS_LABELS.keys())
    status_filter = st.selectbox("상태 필터", ["전체"] + status_options, format_func=lambda x: "전체" if x == "전체" else STATUS_LABELS[x])
    sparams = {} if status_filter == "전체" else {"status": status_filter}
    watch = pd.DataFrame(api_get("/api/subscriber-watch", {**sparams, "limit": 200}))
    if not watch.empty:
        show = watch.copy()
        show["상태"] = show["구독자_상태"].map(STATUS_LABELS)
        show = show.rename(columns={"recency_days": "무주문 일수", "age_band_h4": "연령대", "gender_h4": "성별", "region_tier": "배송권역"})
        st.dataframe(
            show[["회원번호", "상태", "무주문 일수", "연령대", "성별", "배송권역"]],
            use_container_width=True, height=420, hide_index=True,
        )
    st.caption("무주문 일수가 긴 순으로 정렬했습니다 — 알림 발송 우선순위로 바로 활용할 수 있습니다.")

# ================= 매출 리포트 =================
with tab_seg:
    st.markdown('<div class="section-title">계절별 매출 추이</div>', unsafe_allow_html=True)
    season = pd.DataFrame(api_get("/api/trend/season"))
    month = pd.DataFrame(api_get("/api/trend/month"))
    sc1, sc2 = st.columns(2)
    with sc1:
        fig = px.bar(season, x="계절", y="총매출", color="계절",
                     color_discrete_sequence=["#6fcf97", "#ffb74d", "#dd4636", "#9575cd"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with sc2:
        fig2 = px.line(month, x="월", y="총매출", markers=True, labels={"월": "월", "총매출": "총매출"})
        fig2.update_traces(line_color="#6fcf97")
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">성별·연령대별 매출 및 인기 상품</div>', unsafe_allow_html=True)
    seg_summary = pd.DataFrame(api_get("/api/segment/summary"))
    fig3 = px.bar(seg_summary, x="age_band_h4", y="총매출", color="gender_h4", barmode="group",
                  labels={"age_band_h4": "연령대", "총매출": "총매출", "gender_h4": "성별"},
                  color_discrete_map={"F": "#dd4636", "M": "#6d4fc4", "Unknown": "#93a0ab"})
    fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

    gcol, acol = st.columns(2)
    gender_pick = gcol.selectbox("성별", sorted(seg_summary["gender_h4"].unique()))
    age_pick = acol.selectbox("연령대", sorted(seg_summary["age_band_h4"].unique()))
    top_items = pd.DataFrame(api_get("/api/segment/top-items", {"gender_h4": gender_pick, "age_band_h4": age_pick}))
    st.write(f"**{gender_pick} / {age_pick}** 인기 상품 Top 5")
    st.dataframe(
        top_items[["물품중분류", "구매금액"]].rename(columns={"물품중분류": "상품 카테고리", "구매금액": "매출"}),
        use_container_width=True, hide_index=True,
    )

# ================= 고객 조회 =================
with tab_upload:
    st.markdown('<div class="section-title">고객 데이터로 바로 조회하기</div>', unsafe_allow_html=True)
    st.write(
        "고객 주문 내역이 담긴 CSV 파일(`회원번호` 컬럼 포함)을 올리면, 저희 팀이 이미 분석해 둔 "
        "고객은 추천 등급과 예상 반응도를 바로 보여드립니다. 처음 보는 고객은 업로드하신 주문 "
        "내역만으로 간단한 요약(주문 횟수·총 구매액·최근 주문일)을 계산해드립니다 — 다만 아직 "
        "분석 이력이 없는 고객이라 정확한 반응 예측까지는 어렵습니다."
    )
    uploaded = st.file_uploader("고객 주문 CSV 업로드", type=["csv"])
    if uploaded is not None:
        files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
        try:
            resp = requests.post(f"{API}/api/analyze-csv", files=files, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            st.error(f"분석 실패: {e}")
        else:
            if "error" in payload:
                st.error(payload["error"])
            else:
                st.success(f"회원 {payload['member_count']}명 조회 완료")
                for r in payload["results"]:
                    with st.expander(f"회원번호 {r['회원번호']} — {'분석 이력 있음' if r.get('found') else '신규 고객'}"):
                        if r.get("found"):
                            m1, m2, m3 = st.columns(3)
                            m1.metric("추천 등급", TIER_LABELS.get(r.get("targeting_tier"), "-"))
                            m2.metric("예상 반응도", f"{r.get('uplift_rf', 0)*100:.1f}%p" if r.get("uplift_rf") is not None else "-")
                            m3.metric("매출 영향(구독 시)", f"{r.get('uplift_revenue_rf', 0):,.0f}원" if r.get("uplift_revenue_rf") is not None else "-")
                        else:
                            st.info(r.get("안내", ""))
                            m1, m2, m3 = st.columns(3)
                            m1.metric("주문 횟수", r.get("신규_주문건수", "-"))
                            m2.metric("총 구매액", f"{r.get('신규_총구매금액', 0):,.0f}원" if r.get("신규_총구매금액") is not None else "-")
                            m3.metric("최근 주문 경과", f"{r.get('최근성_경과일', '-')}일" if r.get("최근성_경과일") is not None else "-")
