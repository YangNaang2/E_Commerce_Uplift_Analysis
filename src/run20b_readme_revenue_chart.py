# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.6), sharex=True)

colors1 = [RED if m == 10 else GREEN for m in months]
ax1.bar(month_labels, revenue_billion, color=colors1)
for i, v in enumerate(revenue_billion):
    ax1.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_title("FRESH.DATA 월별 매출 (30일 환산, 억원)", fontsize=18, fontweight='bold')
ax1.set_ylabel("매출(억원)")
ax1.set_ylim(0, max(revenue_billion) * 1.18)

colors2 = [RED if m == 10 else GREEN for m in months]
ax2.plot(month_labels, food_growth, color=GREY, lw=2, zorder=1)
ax2.scatter(month_labels, food_growth, color=colors2, s=90, zorder=2)
for i, v in enumerate(food_growth):
    ax2.text(i, v + 0.6, f"{v:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.axhline(0, color=GREY, lw=1, ls='--')
ax2.set_title("국내 온라인 음식료품 카테고리 성장률 (전년동월대비, %)", fontsize=18, fontweight='bold')
ax2.set_ylabel("전년동월대비(%)")
ax2.set_ylim(min(food_growth) - 2, max(food_growth) * 1.2)

for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle("9월 피크 이후 10월 급락 — 우리 회사와 업계 전체가 같은 시기에 겪은 하락", fontsize=15, y=1.01, color=GREY)
plt.tight_layout()
OUT = r"C:\Users\aidan\OneDrive\바탕 화면\종합실습\data\processed\run20_charts\05_매출추이_비교.png"
plt.savefig(OUT, dpi=150, facecolor='white', bbox_inches='tight')
print("saved:", OUT)
