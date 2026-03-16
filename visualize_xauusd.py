import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import calendar

# Load real XAUUSD trade log
df = pd.read_csv('xauusd_filtered_trades.csv')
df['exit_time'] = pd.to_datetime(df['exit_time'])

df['year']        = df['exit_time'].dt.year
df['month']       = df['exit_time'].dt.month
df['day_of_week'] = df['exit_time'].dt.day_name()
df['week']        = df['exit_time'].dt.isocalendar().week
df['hour_utc']    = df['exit_time'].dt.hour

def assign_session(hour):
    if 0 <= hour < 8:
        return 'Asian\n(00–08 UTC)'
    elif 8 <= hour < 13:
        return 'London Morning\n(08–13 UTC)'
    elif 13 <= hour < 17:
        return 'NY/London Overlap\n(13–17 UTC)'
    elif 17 <= hour < 22:
        return 'NY Afternoon\n(17–22 UTC)'
    else:
        return 'Late NY/Sydney\n(22–24 UTC)'

df['session'] = df['hour_utc'].apply(assign_session)

INITIAL_EQUITY = 10000.0

def calc_pct(group):
    return round(group['pnl'].sum() / INITIAL_EQUITY * 100, 2)

def bar_colors(values):
    norm = plt.Normalize(values.min(), values.max())
    sm   = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    return [sm.to_rgba(v) for v in values]

# ── Layout ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(20, 16), facecolor="#1a1a2e")
fig.suptitle("XAU/USD EMA Pullback — 5-Year Backtest  |  Net Profit % Analysis\n(Mon–Thu · 13:00–17:00 UTC · $10,000 Equity · 1% Risk/Trade · 2×RR)",
             fontsize=14, fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

def styled_bar(ax, x_vals, y_vals, title, xlabel, ylabel, rotation=0):
    colors = bar_colors(np.array(y_vals))
    bars = ax.bar(x_vals, y_vals, color=colors, edgecolor='#444', linewidth=0.5)
    ax.axhline(0, color='white', linewidth=1.2, linestyle='--', alpha=0.6)
    ax.set_title(title, fontsize=12, fontweight='bold', color='white', pad=8)
    ax.set_xlabel(xlabel, color='#aaa', fontsize=9)
    ax.set_ylabel(ylabel, color='#aaa', fontsize=9)
    ax.tick_params(colors='#bbb', labelsize=8)
    ax.set_facecolor('#12122a')
    ax.spines[:].set_color('#333')
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right' if rotation else 'center')
    for bar, val in zip(bars, y_vals):
        va    = 'bottom' if val >= 0 else 'top'
        offset = 0.15 if val >= 0 else -0.15
        ax.annotate(f"{val:+.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, val + offset),
                    ha='center', va=va, fontsize=8, fontweight='bold',
                    color='white')
    return ax

# ── 1. Session (full width, top) ────────────────────────────────────────────
ax_sess = fig.add_subplot(gs[0, :])
session_order = ['Asian\n(00–08 UTC)', 'London Morning\n(08–13 UTC)',
                 'NY/London Overlap\n(13–17 UTC)', 'NY Afternoon\n(17–22 UTC)',
                 'Late NY/Sydney\n(22–24 UTC)']
sess_pct = df.groupby('session').apply(calc_pct).reindex(session_order).fillna(0).reset_index()
sess_pct.columns = ['session', 'pct']
styled_bar(ax_sess, sess_pct['session'], sess_pct['pct'],
           '📌 Net Profit (%) by Trading Session', 'Market Session', 'Net Profit (%)')

trade_counts = df.groupby('session').size().reindex(session_order).fillna(0)
for idx, (bar, cnt) in enumerate(zip(ax_sess.patches, trade_counts)):
    ax_sess.text(bar.get_x() + bar.get_width()/2, -3.5,
                 f"{int(cnt)} trades", ha='center', va='top', fontsize=8, color='#aaa')

# ── 2. Day of Week ───────────────────────────────────────────────────────────
ax_dow = fig.add_subplot(gs[1, 0])
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_pct = df.groupby('day_of_week').apply(calc_pct).reindex(day_order).fillna(0).reset_index()
dow_pct.columns = ['day', 'pct']
styled_bar(ax_dow, dow_pct['day'], dow_pct['pct'],
           '📅 Net Profit (%) by Day of Week', 'Day', 'Net Profit (%)', rotation=30)

# ── 3. Monthly Seasonality ───────────────────────────────────────────────────
ax_mo = fig.add_subplot(gs[1, 1])
mo_pct = df.groupby('month').apply(calc_pct).reset_index()
mo_pct.columns = ['month', 'pct']
mo_pct['month_name'] = mo_pct['month'].apply(lambda x: calendar.month_abbr[x])
styled_bar(ax_mo, mo_pct['month_name'], mo_pct['pct'],
           '📆 Net Profit (%) by Month (Seasonality)', 'Month', 'Net Profit (%)')

# ── 4. Yearly ────────────────────────────────────────────────────────────────
ax_yr = fig.add_subplot(gs[2, 0])
yr_pct = df.groupby('year').apply(calc_pct).reset_index()
yr_pct.columns = ['year', 'pct']
styled_bar(ax_yr, yr_pct['year'].astype(str), yr_pct['pct'],
           '📈 Net Profit (%) by Year', 'Year', 'Net Profit (%)')

# ── 5. Weekly Heatmap ────────────────────────────────────────────────────────
ax_hm = fig.add_subplot(gs[2, 1])
wk_matrix = df.groupby(['year','week']).apply(calc_pct).unstack(level=1)
wk_matrix.index = wk_matrix.index.astype(str)
vmax = max(abs(wk_matrix.max().max()), abs(wk_matrix.min().min()))
sns.heatmap(wk_matrix, cmap="RdYlGn", center=0, vmin=-vmax, vmax=vmax,
            ax=ax_hm, linewidths=0.3, cbar_kws={'label':'Net Profit (%)', 'shrink':0.8},
            annot=False)
ax_hm.set_title('🗓️ Weekly Net Profit (%) Heatmap', fontsize=12, fontweight='bold', color='white', pad=8)
ax_hm.set_xlabel('Week of Year', color='#aaa', fontsize=9)
ax_hm.set_ylabel('Year', color='#aaa', fontsize=9)
ax_hm.tick_params(colors='#bbb', labelsize=7)
ax_hm.set_facecolor('#12122a')

output_path = '/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/xauusd_profit_dashboard.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Dashboard saved → {output_path}")
