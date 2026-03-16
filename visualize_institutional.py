"""
Full visual report for Institutional Confluence Strategy
Same layout as before: Session · Day · Month · Year · Weekly Heatmap
+ Equity Curve + Drawdown + Score Breakdown
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, seaborn as sns, calendar

# ── Load trades ───────────────────────────────────────────────────────────────
df = pd.read_csv('xauusd_institutional_trades.csv')
df['exit_time']  = pd.to_datetime(df['exit_time'])
df['entry_time'] = pd.to_datetime(df['entry_time'])

df['year']        = df['exit_time'].dt.year
df['month']       = df['exit_time'].dt.month
df['week']        = df['exit_time'].dt.isocalendar().week.astype(int)
df['day_of_week'] = df['exit_time'].dt.day_name()
df['hour']        = df['exit_time'].dt.hour

def session(h):
    if 0 <= h < 8:   return 'Asian\n(00-08 UTC)'
    if 8 <= h < 13:  return 'London Open\n(08-13 UTC)'
    if 13 <= h < 17: return 'NY/London\nOverlap (13-17)'
    if 17 <= h < 19: return 'NY Session\n(17-19 UTC)'
    return 'Other'

df['session'] = df['hour'].apply(session)

INITIAL = 10000.0

def pct(g): return round(g['pnl'].sum() / INITIAL * 100, 2)

def bar_colors(vals):
    arr = np.array(vals, dtype=float)
    if len(arr) == 0: return []
    if arr.max() == arr.min(): return ['#4CAF50'] * len(arr)
    norm = plt.Normalize(arr.min(), arr.max())
    return [plt.cm.RdYlGn(norm(v)) for v in arr]

def styled_bar(ax, x_vals, y_vals, title, xlabel='', ylabel='Net %', rotation=0, fontsize=8):
    y = list(y_vals)
    colors = bar_colors(y)
    bars = ax.bar(x_vals, y, color=colors, edgecolor='#444', lw=0.5, zorder=3)
    ax.axhline(0, color='white', lw=1.0, ls='--', alpha=0.5, zorder=2)
    ax.set_title(title, fontsize=10, fontweight='bold', color='white', pad=7)
    ax.set_xlabel(xlabel, color='#aaa', fontsize=8)
    ax.set_ylabel(ylabel, color='#aaa', fontsize=8)
    ax.tick_params(colors='#bbb', labelsize=fontsize)
    ax.set_faceground = ax.set_facecolor('#0d0d1f')
    ax.spines[:].set_color('#333')
    ax.grid(axis='y', alpha=0.15, zorder=1)
    if rotation:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=fontsize)
    for bar, val in zip(bars, y):
        va     = 'bottom' if val >= 0 else 'top'
        offset = 0.08 if val >= 0 else -0.08
        ax.annotate(f"{val:+.1f}%",
                    xy=(bar.get_x() + bar.get_width()/2, val + offset),
                    ha='center', va=va, fontsize=6.5, color='white', fontweight='bold')

# ── Layout ────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid')
wins   = df[df['status']=='win']
losses = df[df['status']=='loss']
wr     = len(wins)/len(df)*100
pf     = wins['pnl'].sum()/abs(losses['pnl'].sum()) if len(losses)>0 else float('inf')
pnl    = df['pnl'].sum()

# Running equity
df_sorted = df.sort_values('exit_time').copy()
df_sorted['running_equity'] = INITIAL + df_sorted['pnl'].cumsum()
df_sorted['peak'] = df_sorted['running_equity'].cummax()
df_sorted['dd']   = (df_sorted['peak'] - df_sorted['running_equity']) / df_sorted['peak'] * 100
max_dd = df_sorted['dd'].max()
final_eq = df_sorted['running_equity'].iloc[-1]

fig = plt.figure(figsize=(22, 22), facecolor='#1a1a2e')
fig.suptitle(
    "XAUUSD Institutional Confluence Strategy — 5-Year Quant Report (2021–2026)\n"
    "7-Signal Scoring · ATR Dynamic Stops · 2.5×RR · Variable Risk (0.8–1.5%) · "
    f"Trades={len(df)} · WR={wr:.1f}% · PF={pf:.2f}x · MaxDD={max_dd:.1f}%",
    fontsize=13, fontweight='bold', color='white', y=0.99
)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.32)

# ─── Row 0: Equity Curve (full width) ────────────────────────────────────────
ax_eq = fig.add_subplot(gs[0, :])
ax_eq.plot(df_sorted['exit_time'], df_sorted['running_equity'], color='#00e676', lw=1.2, label='Equity')
ax_eq.fill_between(df_sorted['exit_time'], INITIAL,
                   df_sorted['running_equity'],
                   where=df_sorted['running_equity'] >= INITIAL,
                   alpha=0.18, color='#00e676')
ax_eq.fill_between(df_sorted['exit_time'], INITIAL,
                   df_sorted['running_equity'],
                   where=df_sorted['running_equity'] < INITIAL,
                   alpha=0.18, color='#ff5252')
ax_eq.axhline(INITIAL, color='white', lw=0.8, ls='--', alpha=0.4)

# Annotate yearly milestones
for yr in df_sorted['year'].unique():
    grp = df_sorted[df_sorted['year']==yr]
    if len(grp):
        yr_pnl = grp['pnl'].sum()/INITIAL*100
        ax_eq.axvline(grp['exit_time'].iloc[0], color='#ffffff22', lw=0.7, ls=':')
        ax_eq.text(grp['exit_time'].iloc[0], ax_eq.get_ylim()[0] if ax_eq.get_ylim()[0] > 0 else INITIAL*0.95,
                   f"{yr}", fontsize=7, color='#aaa', va='bottom')

ax_eq.set_title(f'Equity Curve  ·  Start: ${INITIAL:,.0f}  →  Final: ${final_eq:,.0f}  '
                f'·  Max Drawdown: {max_dd:.1f}%',
                fontsize=11, fontweight='bold', color='white', pad=8)
ax_eq.set_facecolor('#0d0d1f'); ax_eq.spines[:].set_color('#333')
ax_eq.tick_params(colors='#bbb', labelsize=8)
ax_eq.set_ylabel('Equity ($)', color='#aaa', fontsize=9)
ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax_eq.grid(alpha=0.12)

# ─── Row 1: Session (full width) ─────────────────────────────────────────────
ax_sess = fig.add_subplot(gs[1, :])
sess_order = ['Asian\n(00-08 UTC)', 'London Open\n(08-13 UTC)',
              'NY/London\nOverlap (13-17)', 'NY Session\n(17-19 UTC)', 'Other']
sess_d = df.groupby('session').apply(pct).reindex(sess_order).fillna(0).reset_index()
sess_d.columns = ['session', 'pct']
styled_bar(ax_sess, sess_d['session'], sess_d['pct'].tolist(),
           'Net Profit (%) by Market Session', ylabel='Net Profit (%)', fontsize=9)
trade_counts = df.groupby('session').size().reindex(sess_order).fillna(0)
for idx, (bar, cnt) in enumerate(zip(ax_sess.patches, trade_counts)):
    ax_sess.text(bar.get_x() + bar.get_width()/2, -2.0,
                 f"{int(cnt)} trades", ha='center', va='top', fontsize=8, color='#aaa')

# ─── Row 2 left: Day of week ──────────────────────────────────────────────────
ax_dow = fig.add_subplot(gs[2, 0])
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_d = df.groupby('day_of_week').apply(pct).reindex(day_order).fillna(0).reset_index()
dow_d.columns = ['day','pct']
styled_bar(ax_dow, dow_d['day'], dow_d['pct'].tolist(),
           'Net Profit (%) by Day of Week', rotation=30, fontsize=8)

# ─── Row 2 right: Monthly seasonality ────────────────────────────────────────
ax_mo = fig.add_subplot(gs[2, 1])
mo_d = df.groupby('month').apply(pct).reset_index()
mo_d.columns = ['month','pct']
mo_d['mn'] = mo_d['month'].apply(lambda x: calendar.month_abbr[x])
styled_bar(ax_mo, mo_d['mn'], mo_d['pct'].tolist(),
           'Net Profit (%) by Month — Seasonality', fontsize=8)

# ─── Row 3 left: Yearly ───────────────────────────────────────────────────────
ax_yr = fig.add_subplot(gs[3, 0])
yr_d = df.groupby('year').apply(pct).reset_index()
yr_d.columns = ['year','pct']
styled_bar(ax_yr, yr_d['year'].astype(str), yr_d['pct'].tolist(),
           'Net Profit (%) by Year', fontsize=9)

# ─── Row 3 right: Weekly heatmap ─────────────────────────────────────────────
ax_hm = fig.add_subplot(gs[3, 1])
wk_matrix = df.groupby(['year','week']).apply(pct).unstack(level=1)
wk_matrix.index = wk_matrix.index.astype(str)
# Fill NaN with 0, limit columns to 1-53
wk_matrix = wk_matrix.fillna(0)
vmax = max(abs(wk_matrix.max().max()), abs(wk_matrix.min().min()), 0.01)
sns.heatmap(wk_matrix, cmap='RdYlGn', center=0, vmin=-vmax, vmax=vmax,
            ax=ax_hm, linewidths=0.2,
            cbar_kws={'label': 'Net Profit (%)', 'shrink': 0.8},
            annot=False, yticklabels=True)
ax_hm.set_title('Weekly Net Profit (%) — Heatmap by Year × Week',
                fontsize=10, fontweight='bold', color='white', pad=7)
ax_hm.set_xlabel('Week of Year', color='#aaa', fontsize=8)
ax_hm.set_ylabel('Year', color='#aaa', fontsize=8)
ax_hm.tick_params(colors='#bbb', labelsize=7)
ax_hm.set_facecolor('#0d0d1f')

# ─── Save ─────────────────────────────────────────────────────────────────────
out = '/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/institutional_full_report.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Report saved → {out}")
