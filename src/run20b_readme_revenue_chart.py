# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13

GREEN = '#2f6b4f'
RED = '#b3311f'
GREY = '#8a9488'

months = list(range(1, 11))
month_labels = [f"{m}월" for m in months]

# dash_month_trend.csv 일평균매출(원) x30 = 30일 환산 월매출(억원). 1월/10월처럼 관측일수가
# 30일에 못 미치는 달의 왜곡을 없애기 위해 raw 총매출 대신 이 값을 쓴다.
daily_avg = [24620990, 29269858, 25522148, 23797002, 24086768,
             24602930, 25652562, 25320664, 27387355, 24981229]
revenue_billion = [v * 30 / 1e8 for v in daily_avg]

# 국내 온라인 식품카테고리 전년동월대비 성장률(%) - 국가데이터처 온라인쇼핑동향 보도자료
food_growth = [9.2, 8.3, 9.8, 9.1, 5.6, 11.0, 12.6, 5.8, 17.7, 4.4]

fig, ax1 = plt.subplots(figsize=(10, 5.6))
ax2 = ax1.twinx()

l1, = ax1.plot(month_labels, revenue_billion, color=GREEN, lw=2.5, marker='o', markersize=7, zorder=3, label="자사 월매출(억원, 30일 환산)")
l2, = ax2.plot(month_labels, food_growth, color=RED, lw=2.5, ls='--', marker='s', markersize=7, zorder=3, label="국내 온라인 식품카테고리 성장률(%, 전년동월대비)")

# 9->10월 급락 구간 강조
ax1.axvspan(8.5, 9.5, color=GREY, alpha=0.12, zorder=1)

for i, v in enumerate(revenue_billion):
    dy = 22 if i == 8 else 10  # 9월(index 8)은 두 선이 만나는 지점이라 더 띄움
    ax1.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, dy),
                 ha='center', fontsize=11, fontweight='bold', color=GREEN)
for i, v in enumerate(food_growth):
    xy = (18, -14) if i == 8 else (0, -16)
    ha = 'left' if i == 8 else 'center'
    ax2.annotate(f"{v:.1f}%", (i, v), textcoords="offset points", xytext=xy,
                 ha=ha, fontsize=11, fontweight='bold', color=RED)

ax1.set_ylabel("자사 월매출(억원)", color=GREEN, fontweight='bold')
ax1.tick_params(axis='y', colors=GREEN)
ax1.set_ylim(0, max(revenue_billion) * 1.22)

ax2.set_ylabel("식품카테고리 성장률(%, 전년동월대비)", color=RED, fontweight='bold')
ax2.tick_params(axis='y', colors=RED)
ax2.set_ylim(min(food_growth) - 3, max(food_growth) * 1.28)
ax2.axhline(0, color=GREY, lw=1, ls=':')

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax1.set_title("9월 피크 이후 10월 급락 — 자사와 업계 전체가 같은 시기에 겪은 하락", fontsize=17, fontweight='bold')
ax1.legend(handles=[l1, l2], loc='lower left', fontsize=12, frameon=False)

plt.tight_layout()
OUT = r"C:\Users\aidan\OneDrive\바탕 화면\종합실습\data\processed\run20_charts\05_매출추이_비교.png"
plt.savefig(OUT, dpi=150, facecolor='white', bbox_inches='tight')
print("saved:", OUT)
