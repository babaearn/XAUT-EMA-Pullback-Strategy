"""
Institutional Strategy — FIXED RISK (Non-Compounding) Statistics
Risk is always calculated on the FIXED initial $10,000 — never grows.
Score 5 = 0.8% = $80 risk/trade  → win = $200
Score 6 = 1.2% = $120 risk/trade → win = $300
Score 7 = 1.5% = $150 risk/trade → win = $375
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, seaborn as sns, calendar
from collections import Counter

# ── Load & fix PnL ────────────────────────────────────────────────────────────
df = pd.read_csv('xauusd_institutional_trades.csv')
df['exit_time']  = pd.to_datetime(df['exit_time'])
df['entry_time'] = pd.to_datetime(df['entry_time'])

INITIAL  = 10_000.0
RR       = 2.5
RISK_MAP = {5: 0.008, 6: 0.012, 7: 0.015}  # fixed % of INITIAL

# Recompute PnL flat
def fixed_pnl(row):
    r = RISK_MAP.get(int(row['score']), 0.008) * INITIAL
    return r * RR if row['status'] == 'win' else -r

df['pnl_fixed'] = df.apply(fixed_pnl, axis=1)

# Time cols
df['year']        = df['exit_time'].dt.year
df['month']       = df['exit_time'].dt.month
df['week_num']    = df['exit_time'].dt.isocalendar().week.astype(int)
df['year_week']   = df['exit_time'].dt.to_period('W')
df['day_of_week'] = df['exit_time'].dt.day_name()
df['hour']        = df['exit_time'].dt.hour

def session(h):
    if 0 <= h < 8:   return 'Asian\n(00-08)'
    if 8 <= h < 13:  return 'London Open\n(08-13)'
    if 13 <= h < 17: return 'NY/London\nOverlap (13-17)'
    if 17 <= h < 19: return 'NY Session\n(17-19)'
    return 'Other'

df['session'] = df['hour'].apply(session)

# ═══════════════════════════════ STATS ════════════════════════════════════════
wins   = df[df['status']=='win']
losses = df[df['status']=='loss']
total  = len(df)
wr     = len(wins) / total * 100
gp     = wins['pnl_fixed'].sum()
gl     = abs(losses['pnl_fixed'].sum())
pf     = gp / gl if gl > 0 else float('inf')
net    = df['pnl_fixed'].sum()

# Running equity (non-compounding)
df_s              = df.sort_values('exit_time').copy()
df_s['equity']    = INITIAL + df_s['pnl_fixed'].cumsum()
df_s['peak']      = df_s['equity'].cummax()
df_s['dd']        = (df_s['peak'] - df_s['equity']) / df_s['peak'] * 100
max_dd            = df_s['dd'].max()
final_eq          = df_s['equity'].iloc[-1]

# Yearly PnL
yearly            = df.groupby('year')['pnl_fixed'].sum()
yearly_pct        = (yearly / INITIAL * 100).round(2)
yearly_pct_ann    = yearly_pct  # each year IS one year

# Sharpe (daily)
daily_r = df_s.set_index('exit_time')['pnl_fixed'].resample('D').sum() / INITIAL * 100
sharpe  = (daily_r.mean() / daily_r.std()) * np.sqrt(252) if daily_r.std() > 0 else 0

avg_win  = wins['pnl_fixed'].mean()
avg_loss = losses['pnl_fixed'].mean()
expectancy = (wr/100 * avg_win) + ((1-wr/100) * abs(avg_loss))

# ── Weekly trade frequency ────────────────────────────────────────────────────
weekly_counts    = df.groupby('year_week').size()
avg_trades_week  = weekly_counts.mean()
median_trades_wk = weekly_counts.median()
max_trades_week  = weekly_counts.max()
min_trades_week  = weekly_counts.min()

# By day within a week
per_day_counts   = df.groupby('day_of_week').size()
day_order        = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
per_day_counts   = per_day_counts.reindex(day_order).fillna(0)

# Average trades per week per year
yr_wk = df.groupby(['year','year_week']).size().groupby('year').mean().round(1)

print(f"""
╔══════════════════════════════════════════════════════════════╗
  INSTITUTIONAL CONFLUENCE — NON-COMPOUNDING STATS
  Fixed Risk: Score5=$80/trade · Score6=$120 · Score7=$150
╠══════════════════════════════════════════════════════════════╣
  Total Trades:      {total:,}
  Win Rate:          {wr:.2f}%
  Profit Factor:     {pf:.2f}x
  Avg Win:           ${avg_win:,.2f}
  Avg Loss:          ${avg_loss:,.2f}
  Expectancy/Trade:  ${expectancy:+.2f}
╠══════════════════════════════════════════════════════════════╣
  Gross Profit:      ${gp:,.2f}
  Gross Loss:        ${gl:,.2f}
  Net PnL:           ${net:,.2f}  ({net/INITIAL*100:.2f}%)
  Final Equity:      ${final_eq:,.2f}
  Max Drawdown:      {max_dd:.2f}%
  Sharpe Ratio:      {sharpe:.2f}
╠══════════════════════════════════════════════════════════════╣
  Yearly Net % (flat $10k):""")
for yr, v in yearly_pct.items():
    sign = '+' if v >= 0 else ''
    bar  = '▓' * min(35, int(abs(v)/2))
    print(f"    {yr}:  {sign}{v:.2f}%  {bar}")

print(f"""╠══════════════════════════════════════════════════════════════╣
  WEEKLY TRADE FREQUENCY:
    Total weeks active:    {len(weekly_counts)}
    Avg trades/week:       {avg_trades_week:.1f}
    Median trades/week:    {median_trades_wk:.1f}
    Max trades in 1 week:  {int(max_trades_week)}
    Min trades in 1 week:  {int(min_trades_week)}

  Avg trades/week by year:""")
for yr, v in yr_wk.items():
    print(f"    {yr}:  {v} trades/week")

print(f"""
  Trades per day of week:""")
for day in day_order:
    cnt = int(per_day_counts.get(day, 0))
    pct_day = cnt/total*100
    print(f"    {day:<12}: {cnt:>4} trades  ({pct_day:.1f}%)")

print(f"╚══════════════════════════════════════════════════════════════╝\n")

# ═══════════════════════════════ CHARTS ═══════════════════════════════════════
def calc_pct(g): return round(g['pnl_fixed'].sum() / INITIAL * 100, 2)

def bar_c(vals):
    arr = np.array(vals, dtype=float)
    if len(arr) == 0: return []
    if arr.max() == arr.min(): return ['#4CAF50']*len(arr)
    norm = plt.Normalize(arr.min(), arr.max())
    return [plt.cm.RdYlGn(norm(v)) for v in arr]

def styled_bar(ax, x, y, title, rot=0, fs=8):
    y = list(y)
    b = ax.bar(x, y, color=bar_c(y), edgecolor='#444', lw=0.5, zorder=3)
    ax.axhline(0, color='white', lw=0.9, ls='--', alpha=0.5, zorder=2)
    ax.set_title(title, fontsize=9, fontweight='bold', color='white', pad=6)
    ax.set_ylabel('Net %', color='#aaa', fontsize=7)
    ax.tick_params(colors='#bbb', labelsize=fs)
    ax.set_facecolor('#0d0d1f'); ax.spines[:].set_color('#333')
    ax.grid(axis='y', alpha=0.15, zorder=1)
    if rot: plt.setp(ax.get_xticklabels(), rotation=rot, ha='right', fontsize=fs-1)
    for bar, val in zip(b, y):
        va = 'bottom' if val >= 0 else 'top'
        ax.annotate(f"{val:+.1f}%",
                    xy=(bar.get_x()+bar.get_width()/2, val+(0.06 if val>=0 else -0.06)),
                    ha='center', va=va, fontsize=6, color='white', fontweight='bold')

sns.set_theme(style='whitegrid')
fig = plt.figure(figsize=(22, 24), facecolor='#1a1a2e')
fig.suptitle(
    f"XAU/USD Institutional Confluence — Fixed Risk Non-Compounding Report\n"
    f"$10,000 Base · Score5=0.8% · Score6=1.2% · Score7=1.5% · 2.5×RR\n"
    f"Net: {net/INITIAL*100:+.2f}% (${net:,.0f})  ·  Max DD: {max_dd:.1f}%  ·  "
    f"Sharpe: {sharpe:.2f}  ·  WR: {wr:.1f}%  ·  PF: {pf:.2f}×  ·  Trades: {total:,}",
    fontsize=12, fontweight='bold', color='white', y=0.995
)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.32)

# ── Equity curve ─────────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(df_s['exit_time'], df_s['equity'], color='#00e676', lw=1.2)
ax0.fill_between(df_s['exit_time'], INITIAL, df_s['equity'],
                 where=df_s['equity']>=INITIAL, alpha=0.18, color='#00e676')
ax0.fill_between(df_s['exit_time'], INITIAL, df_s['equity'],
                 where=df_s['equity']< INITIAL, alpha=0.18, color='#ff5252')
ax0.axhline(INITIAL, color='white', lw=0.8, ls='--', alpha=0.4)
for yr in df_s['year'].unique():
    first = df_s[df_s['year']==yr]['exit_time'].iloc[0]
    ax0.axvline(first, color='#ffffff18', lw=0.7, ls=':')
    yr_net = yearly_pct.get(yr, 0)
    ax0.text(first, INITIAL*1.01, f"{yr}\n{yr_net:+.0f}%", fontsize=7.5,
             color='#00e676' if yr_net>=0 else '#ff5252', fontweight='bold', va='bottom')
ax0.set_title(f'Equity Curve (Non-Compounding)  ·  Start ${INITIAL:,.0f}  →  Final ${final_eq:,.0f}  '
              f'·  Max DD {max_dd:.1f}%', fontsize=10, fontweight='bold', color='white', pad=7)
ax0.set_facecolor('#0d0d1f'); ax0.spines[:].set_color('#333')
ax0.tick_params(colors='#bbb', labelsize=8)
ax0.set_ylabel('Equity ($)', color='#aaa', fontsize=9)
ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax0.grid(alpha=0.12)

# ── Session ───────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, :])
sess_order = ['Asian\n(00-08)', 'London Open\n(08-13)',
              'NY/London\nOverlap (13-17)', 'NY Session\n(17-19)', 'Other']
sd = df.groupby('session').apply(calc_pct).reindex(sess_order).fillna(0).reset_index()
sd.columns = ['session','pct']
styled_bar(ax1, sd['session'], sd['pct'].tolist(),
           'Net Profit (%) by Market Session  [Fixed Risk]', fs=9)
for idx, (bar, cnt) in enumerate(
        zip(ax1.patches, df.groupby('session').size().reindex(sess_order).fillna(0))):
    ax1.text(bar.get_x()+bar.get_width()/2, -2.0,
             f"{int(cnt)} trades", ha='center', va='top', fontsize=8, color='#aaa')

# ── Day of week ───────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[2, 0])
dow_d = df.groupby('day_of_week').apply(calc_pct).reindex(day_order).fillna(0).reset_index()
dow_d.columns = ['day','pct']
styled_bar(ax2, dow_d['day'], dow_d['pct'].tolist(),
           'Net Profit (%) by Day of Week', rot=30)

# ── Monthly ───────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 1])
md = df.groupby('month').apply(calc_pct).reset_index()
md.columns = ['month','pct']
md['mn'] = md['month'].apply(lambda x: calendar.month_abbr[x])
styled_bar(ax3, md['mn'], md['pct'].tolist(),
           'Net Profit (%) by Month')

# ── Yearly ────────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3, 0])
yd = yearly_pct.reset_index(); yd.columns=['year','pct']
styled_bar(ax4, yd['year'].astype(str), yd['pct'].tolist(),
           'Net Profit (%) by Year  [Fixed Risk — True Annual %]')

# ── Weekly avg trades (bar per year showing avg trades/week) ─────────────────
ax5 = fig.add_subplot(gs[3, 1])
# Distribution of how many trades happen per week
wk_dist = weekly_counts.value_counts().sort_index()
ax5.bar(wk_dist.index.astype(str), wk_dist.values,
        color=[plt.cm.Blues(0.4 + 0.6*v/wk_dist.max()) for v in wk_dist.values],
        edgecolor='#444', lw=0.5, zorder=3)
ax5.axvline(str(int(round(avg_trades_week))), color='#00e676', lw=2, ls='--', alpha=0.8, label=f'Avg={avg_trades_week:.1f}/wk')
ax5.set_title(f'Weekly Trade Frequency Distribution\n'
              f'Avg={avg_trades_week:.1f}/wk · Median={median_trades_wk:.0f}/wk · '
              f'Range={int(min_trades_week)}-{int(max_trades_week)}',
              fontsize=9, fontweight='bold', color='white', pad=6)
ax5.set_xlabel('Trades in that week', color='#aaa', fontsize=8)
ax5.set_ylabel('Number of weeks', color='#aaa', fontsize=8)
ax5.tick_params(colors='#bbb', labelsize=8)
ax5.set_facecolor('#0d0d1f'); ax5.spines[:].set_color('#333')
ax5.grid(axis='y', alpha=0.15, zorder=1)
ax5.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

# Annotate counts
for x, (idx, v) in zip(ax5.patches, wk_dist.items()):
    ax5.text(x.get_x()+x.get_width()/2, v+0.3, str(v),
             ha='center', va='bottom', fontsize=7, color='white')

out = '/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/institutional_fixed_risk_report.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved → {out}")
