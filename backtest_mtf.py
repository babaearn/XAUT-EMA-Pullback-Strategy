"""
Multi-Timeframe Backtest: XAU/USD
Filters: Tue/Wed/Thu only · 08:00-19:00 UTC · Skip June
Timeframes: 5m, 15m, 1H
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import calendar
from pathlib import Path

# ── Indicator helpers ─────────────────────────────────────────────
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = (-delta).where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))

# ── Backtest engine ────────────────────────────────────────────────
def run_backtest(csv_path: str, label: str,
                 ema_period: int = 21,
                 tap_pct: float = 0.2,
                 sl_pct: float  = 0.5,
                 rr: float      = 2.0,
                 risk_pct: float= 1.0,
                 hour_from: int = 8,
                 hour_to: int   = 19,
                 allowed_days   = (1, 2, 3),   # Tue=1, Wed=2, Thu=3
                 skip_months    = (6,)) -> dict:

    df = pd.read_csv(csv_path)
    df['dt']  = pd.to_datetime(df['timestamp'], unit='ms')
    df['ema'] = ema(df['close'], ema_period)
    df['rsi'] = rsi(df['close'], 14)

    hours   = df['dt'].dt.hour.values
    dow     = df['dt'].dt.dayofweek.values   # Mon=0 … Sun=6
    months  = df['dt'].dt.month.values
    closes  = df['close'].values
    emas    = df['ema'].values
    rsis    = df['rsi'].values
    highs   = df['high'].values
    lows    = df['low'].values
    times   = df['timestamp'].values

    equity       = 10000.0
    initial_eq   = equity
    position     = None
    sl = tp = 0.0
    entry_equity = equity
    entry_time   = None
    in_trend     = False
    trades       = []
    eq_curve     = []

    for i in range(50, len(df)):
        eq_curve.append({'time': times[i], 'equity': equity})

        close   = closes[i]
        high    = highs[i]
        low     = lows[i]
        ema_val = emas[i]

        # Exit logic (runs regardless of time filter)
        if position == "long":
            risk_amt = entry_equity * risk_pct / 100
            if low <= sl:
                equity -= risk_amt
                trades.append({'entry_time': entry_time, 'exit_time': times[i],
                               'type': 'long', 'status': 'loss', 'pnl': -risk_amt})
                position = None
            elif high >= tp:
                equity += risk_amt * rr
                trades.append({'entry_time': entry_time, 'exit_time': times[i],
                               'type': 'long', 'status': 'win', 'pnl': risk_amt * rr})
                position = None

        elif position == "short":
            risk_amt = entry_equity * risk_pct / 100
            if high >= sl:
                equity -= risk_amt
                trades.append({'entry_time': entry_time, 'exit_time': times[i],
                               'type': 'short', 'status': 'loss', 'pnl': -risk_amt})
                position = None
            elif low <= tp:
                equity += risk_amt * rr
                trades.append({'entry_time': entry_time, 'exit_time': times[i],
                               'type': 'short', 'status': 'win', 'pnl': risk_amt * rr})
                position = None

        # Entry filter
        if position is None:
            prev_close = closes[i-1]
            prev_ema   = emas[i-1]

            trend_flip = ((prev_close <= prev_ema and close > ema_val) or
                          (prev_close >= prev_ema and close < ema_val))
            if trend_flip:
                in_trend = True

            valid = (dow[i] in allowed_days and
                     hour_from <= hours[i] < hour_to and
                     months[i] not in skip_months)

            if valid:
                tap_zone   = ema_val * (tap_pct / 100)
                long_tap   = (close > ema_val and close <= ema_val + tap_zone) and in_trend
                short_tap  = (close < ema_val and close >= ema_val - tap_zone) and in_trend
                rsi_long   = rsis[i] > 50
                rsi_short  = rsis[i] < 50

                if long_tap and rsi_long:
                    position     = "long"
                    sl           = ema_val * (1 - sl_pct / 100)
                    risk         = abs(close - sl)
                    tp           = close + risk * rr
                    entry_equity = equity
                    entry_time   = times[i]
                    in_trend     = False
                elif short_tap and rsi_short:
                    position     = "short"
                    sl           = ema_val * (1 + sl_pct / 100)
                    risk         = abs(sl - close)
                    tp           = close - risk * rr
                    entry_equity = equity
                    entry_time   = times[i]
                    in_trend     = False

    # ── Stats ──────────────────────────────────────────────────────
    tdf = pd.DataFrame(trades)
    if tdf.empty:
        print(f"[{label}] No trades generated!")
        return {}

    tdf['entry_time'] = pd.to_datetime(tdf['entry_time'], unit='ms')
    tdf['exit_time']  = pd.to_datetime(tdf['exit_time'],  unit='ms')
    tdf.to_csv(f"xauusd_{label}_trades.csv", index=False)

    eq_df = pd.DataFrame(eq_curve)
    eq_df['time']   = pd.to_datetime(eq_df['time'], unit='ms')
    eq_df['cummax'] = eq_df['equity'].cummax()
    eq_df['dd']     = (eq_df['cummax'] - eq_df['equity']) / eq_df['cummax'] * 100
    max_dd = eq_df['dd'].max()

    wins   = tdf[tdf['status']=='win']
    losses = tdf[tdf['status']=='loss']
    pnl    = tdf['pnl'].sum()
    gp     = wins['pnl'].sum()
    gl     = abs(losses['pnl'].sum())
    pf     = gp / gl if gl > 0 else float('inf')
    wr     = len(wins) / len(tdf) * 100

    print(f"\n===== {label} =====")
    print(f"  Trades: {len(tdf)}  |  Wins: {len(wins)}  |  Losses: {len(losses)}")
    print(f"  Win Rate: {wr:.2f}%  |  Profit Factor: {pf:.2f}x")
    print(f"  Total PnL: ${pnl:.2f} ({pnl/initial_eq*100:.2f}%)")
    print(f"  Gross Profit: ${gp:.2f}  |  Gross Loss: -${gl:.2f}")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"  Final Equity: ${equity:.2f}")

    return dict(label=label, trades=tdf, equity=equity, pnl=pnl,
                pct=pnl/initial_eq*100, max_dd=max_dd, wr=wr, pf=pf,
                gp=gp, gl=gl)

# ── Run all three ────────────────────────────────────────────────
FILTER = dict(hour_from=8, hour_to=19, allowed_days=(1,2,3), skip_months=(6,))

r5m  = run_backtest("data/xauusd-m5-bid-2021-03-01-2026-03-15.csv",  "5m",  **FILTER)
r15m = run_backtest("data/xauusd-m15-bid-2021-03-01-2026-03-15.csv", "15m", **FILTER)
r1h  = run_backtest("data/xauusd-h1-bid-2021-03-01-2026-03-15.csv",  "1H",  **FILTER)

results = [r for r in [r5m, r15m, r1h] if r]

# ── Charting ──────────────────────────────────────────────────────
INITIAL_EQUITY = 10000.0

def calc_pct(group):
    return round(group['pnl'].sum() / INITIAL_EQUITY * 100, 2)

def bar_colors(values):
    arr  = np.array(values, dtype=float)
    norm = plt.Normalize(arr.min(), arr.max())
    sm   = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    return [sm.to_rgba(v) for v in arr]

def styled_bar(ax, x_vals, y_vals, title, xlabel='', rotation=0, fontsize=8):
    colors = bar_colors(y_vals)
    bars   = ax.bar(x_vals, y_vals, color=colors, edgecolor='#444', linewidth=0.5)
    ax.axhline(0, color='white', linewidth=1.0, linestyle='--', alpha=0.6)
    ax.set_title(title, fontsize=10, fontweight='bold', color='white', pad=6)
    ax.set_xlabel(xlabel, color='#aaa', fontsize=8)
    ax.set_ylabel('Net %', color='#aaa', fontsize=8)
    ax.tick_params(colors='#bbb', labelsize=7)
    ax.set_facecolor('#12122a')
    ax.spines[:].set_color('#333')
    plt.setp(ax.get_xticklabels(), rotation=rotation,
             ha='right' if rotation else 'center', fontsize=fontsize)
    for bar, val in zip(bars, y_vals):
        va = 'bottom' if val >= 0 else 'top'
        ax.annotate(f"{val:+.2f}%",
                    xy=(bar.get_x() + bar.get_width()/2, val + (0.1 if val>=0 else -0.1)),
                    ha='center', va=va, fontsize=7, color='white', fontweight='bold')

sns.set_theme(style='whitegrid')
ROWS = len(results)
fig  = plt.figure(figsize=(22, 7 * ROWS + 2), facecolor='#1a1a2e')
fig.suptitle("XAU/USD — Multi-Timeframe Backtest (5m / 15m / 1H)\n"
             "Filter: Tue/Wed/Thu · 08:00–19:00 UTC · Skip June · $10k · 1% Risk · 2x RR",
             fontsize=13, fontweight='bold', color='white', y=0.99)

day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

for row_idx, res in enumerate(results):
    tdf   = res['trades']
    label = res['label']

    tdf['year']    = tdf['exit_time'].dt.year
    tdf['month']   = tdf['exit_time'].dt.month
    tdf['dow']     = tdf['exit_time'].dt.day_name()
    tdf['hour']    = tdf['exit_time'].dt.hour

    gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=
         gridspec.GridSpec(ROWS, 1, figure=fig, hspace=0.55)[row_idx],
         wspace=0.35)

    # Summary annotation
    ax_info = fig.add_subplot(gs[0])
    ax_info.axis('off')
    ax_info.set_facecolor('#12122a')
    summary = (f"Timeframe: {label}\n\n"
               f"Trades:    {res['trades'].shape[0]}\n"
               f"Win Rate:  {res['wr']:.1f}%\n"
               f"Net PnL:   {res['pct']:+.2f}%\n"
               f"Profit Factor: {res['pf']:.2f}x\n"
               f"Max DD:    {res['max_dd']:.2f}%\n"
               f"Final Eq:  ${res['equity']:,.2f}")
    color = '#00e676' if res['pct'] >= 0 else '#ff5252'
    ax_info.text(0.5, 0.5, summary, ha='center', va='center',
                 fontsize=10, color=color, fontweight='bold',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#0d0d1f',
                           edgecolor=color, linewidth=1.5),
                 transform=ax_info.transAxes)
    ax_info.set_title(f"[{label}] Summary", fontsize=10, fontweight='bold', color='white')

    # Month
    ax_mo = fig.add_subplot(gs[1])
    mo = tdf.groupby('month').apply(calc_pct).reset_index()
    mo.columns = ['month','pct']
    mo['mn'] = mo['month'].apply(lambda x: calendar.month_abbr[x])
    styled_bar(ax_mo, mo['mn'], mo['pct'].tolist(), f'[{label}] Monthly', rotation=30)

    # Yearly
    ax_yr = fig.add_subplot(gs[2])
    yr = tdf.groupby('year').apply(calc_pct).reset_index()
    yr.columns = ['year','pct']
    styled_bar(ax_yr, yr['year'].astype(str), yr['pct'].tolist(), f'[{label}] Yearly')

    # Hourly (instead of session — since window is 8-19 it's more granular)
    ax_hr = fig.add_subplot(gs[3])
    hr = tdf.groupby('hour').apply(calc_pct).reset_index()
    hr.columns = ['hour','pct']
    styled_bar(ax_hr, hr['hour'].astype(str), hr['pct'].tolist(), f'[{label}] By Hour (UTC)', rotation=45)

out = '/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/xauusd_mtf_dashboard.png'
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nDashboard saved -> {out}")
