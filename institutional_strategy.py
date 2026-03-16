"""
GOLD INSTITUTIONAL CONFLUENCE STRATEGY v2
==========================================
Clean, vectorized implementation.
Signals:
  1. EMA8 > EMA21 > EMA50 aligned (5m)
  2. RSI 5m 50-70 long / 30-50 short
  3. MACD histogram positive/negative (5m)
  4. 1H EMA21 > 1H EMA200 (macro trend)
  5. ADX(1H) > 20 (real trend, not chop)
  6. Volume > 20-bar MA  
  7. 1H RSI > 50 (higher TF momentum)

Entry: Price pulls into 21 EMA (0.2% zone) when score >= 5
Exit:  ATR-based SL (1.5×ATR) | 2.5×RR TP | no trailing (clean first)
Risk:  score7=1.5%, score6=1.2%, score5=0.8%
Filter: Tue/Wed/Thu · 08:00-19:00 UTC · skip June
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, seaborn as sns, calendar
from collections import Counter

# ─── Indicators ──────────────────────────────────────────────────────────────
def ema_c(s, p): return s.ewm(span=p, adjust=False).mean()

def rsi_c(s, p=14):
    d = s.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/p, adjust=False).mean()
    al = l.ewm(alpha=1/p, adjust=False).mean()
    return 100 - 100/(1 + ag/al.replace(0, 1e-9))

def atr_c(h, l, c, p=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def adx_c(h, l, c, p=14):
    up = h.diff(); dn = -l.diff()
    pdm = up.where((up>dn)&(up>0), 0.0)
    ndm = dn.where((dn>up)&(dn>0), 0.0)
    at  = atr_c(h,l,c,p)
    pdi = 100*pdm.ewm(alpha=1/p,adjust=False).mean()/at.replace(0,1e-9)
    ndi = 100*ndm.ewm(alpha=1/p,adjust=False).mean()/at.replace(0,1e-9)
    dx  = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0,1e-9)
    return dx.ewm(alpha=1/p,adjust=False).mean(), pdi, ndi

def macd_c(s, f=12, sl=26, sg=9):
    ml = ema_c(s,f) - ema_c(s,sl)
    sig = ema_c(ml, sg)
    return ml, sig

# ─── Load & Compute 5m ───────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/xauusd-m5-bid-2021-03-01-2026-03-15.csv")
df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
df['hour']  = df['dt'].dt.hour
df['dow']   = df['dt'].dt.dayofweek   # Mon=0, Sun=6
df['month'] = df['dt'].dt.month

print("Computing 5m indicators...")
df['ema8']  = ema_c(df['close'], 8)
df['ema21'] = ema_c(df['close'], 21)
df['ema50'] = ema_c(df['close'], 50)
df['rsi']   = rsi_c(df['close'], 14)
df['atr']   = atr_c(df['high'], df['low'], df['close'], 14)
ml, sl_m    = macd_c(df['close'])
df['macd']  = ml; df['macd_s'] = sl_m
df['macd_hist'] = ml - sl_m
df['vol_ma']= df['volume'].rolling(20).mean()

# ─── Resample to 1H & Compute ────────────────────────────────────────────────
print("Computing 1H indicators...")
df.set_index('dt', inplace=True)
h1 = df[['open','high','low','close','volume']].resample('1H').agg(
    {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

h1['ema21']  = ema_c(h1['close'], 21)
h1['ema200'] = ema_c(h1['close'], 200)
h1['rsi']    = rsi_c(h1['close'], 14)
h1_adx, h1_pdi, h1_ndi = adx_c(h1['high'], h1['low'], h1['close'], 14)
h1['adx'] = h1_adx; h1['pdi'] = h1_pdi; h1['ndi'] = h1_ndi

# Forward-fill 1H values into 5m
df['h1_ema21']  = h1['ema21'].reindex(df.index, method='ffill')
df['h1_ema200'] = h1['ema200'].reindex(df.index, method='ffill')
df['h1_rsi']    = h1['rsi'].reindex(df.index, method='ffill')
df['h1_adx']    = h1['adx'].reindex(df.index, method='ffill')
df['h1_pdi']    = h1['pdi'].reindex(df.index, method='ffill')
df['h1_ndi']    = h1['ndi'].reindex(df.index, method='ffill')

df.reset_index(inplace=True)

# ─── Vectorized Signal Scoring (LONG) ────────────────────────────────────────
d = df
s1L = (d['ema8'] > d['ema21']) & (d['ema21'] > d['ema50'])
s2L = (d['rsi'] > 50) & (d['rsi'] < 70)
s3L = d['macd_hist'] > 0
s4L = d['h1_ema21'] > d['h1_ema200']
s5L = (d['h1_adx'] > 20) & (d['h1_pdi'] > d['h1_ndi'])
s6  = d['volume'] > d['vol_ma']
s7L = d['h1_rsi'] > 50

df['score_long']  = s1L.astype(int)+s2L.astype(int)+s3L.astype(int)+\
                    s4L.astype(int)+s5L.astype(int)+s6.astype(int)+s7L.astype(int)

s1S = (d['ema8'] < d['ema21']) & (d['ema21'] < d['ema50'])
s2S = (d['rsi'] > 30) & (d['rsi'] < 50)
s3S = d['macd_hist'] < 0
s4S = d['h1_ema21'] < d['h1_ema200']
s5S = (d['h1_adx'] > 20) & (d['h1_ndi'] > d['h1_pdi'])
s7S = d['h1_rsi'] < 50

df['score_short'] = s1S.astype(int)+s2S.astype(int)+s3S.astype(int)+\
                    s4S.astype(int)+s5S.astype(int)+s6.astype(int)+s7S.astype(int)

# ─── Entry Conditions ─────────────────────────────────────────────────────────
TAP = 0.20   # 0.20% tap zone
df['tap_zone'] = df['ema21'] * TAP / 100

df['in_long_tap']  = (df['close'] > df['ema21']) & (df['close'] <= df['ema21'] + df['tap_zone'])
df['in_short_tap'] = (df['close'] < df['ema21']) & (df['close'] >= df['ema21'] - df['tap_zone'])

df['valid_session'] = ((df['dow'].isin([1,2,3])) &
                       (df['hour'] >= 8) & (df['hour'] < 19) &
                       (df['month'] != 6))

df['entry_long']  = df['in_long_tap']  & (df['score_long']  >= 5) & df['valid_session']
df['entry_short'] = df['in_short_tap'] & (df['score_short'] >= 5) & df['valid_session']

# ─── Simulation Loop (state machine) ─────────────────────────────────────────
print("Simulating trades...")

RISK_MAP   = {7: 1.5, 6: 1.2, 5: 0.8}
RR         = 2.5
ATR_SL_MUL = 1.5
INITIAL    = 10000.0
equity     = INITIAL

pos = None; sl = tp = 0.0; risk_pct = 1.0; entry_score = 5
trades = []; eq_curve = []; score_log = []

entry_long_arr  = df['entry_long'].values
entry_short_arr = df['entry_short'].values
close_arr = df['close'].values; high_arr = df['high'].values
low_arr   = df['low'].values;   atr_arr  = df['atr'].values
ts_arr    = df['timestamp'].values
sL_arr    = df['score_long'].values; sS_arr = df['score_short'].values

for i in range(300, len(df)):
    eq_curve.append(equity)
    if equity <= 100: break                             # protect from ruin

    c = close_arr[i]; h = high_arr[i]; lo = low_arr[i]
    at = atr_arr[i] if (not np.isnan(atr_arr[i]) and atr_arr[i] > 0) else c * 0.003

    risk_amt = entry_eq * risk_pct / 100 if pos else 0.0

    if pos == "long":
        if lo <= sl:
            equity -= risk_amt
            trades.append({'entry_time': entry_t, 'exit_time': ts_arr[i],
                           'status': 'loss', 'pnl': -risk_amt, 'score': entry_score, 'type': 'long'})
            pos = None
        elif h >= tp:
            equity += risk_amt * RR
            trades.append({'entry_time': entry_t, 'exit_time': ts_arr[i],
                           'status': 'win', 'pnl': risk_amt * RR, 'score': entry_score, 'type': 'long'})
            pos = None

    elif pos == "short":
        if h >= sl:
            equity -= risk_amt
            trades.append({'entry_time': entry_t, 'exit_time': ts_arr[i],
                           'status': 'loss', 'pnl': -risk_amt, 'score': entry_score, 'type': 'short'})
            pos = None
        elif lo <= tp:
            equity += risk_amt * RR
            trades.append({'entry_time': entry_t, 'exit_time': ts_arr[i],
                           'status': 'win', 'pnl': risk_amt * RR, 'score': entry_score, 'type': 'short'})
            pos = None

    if pos is None:
        if entry_long_arr[i]:
            sc = int(sL_arr[i])
            if sc >= 5:
                pos         = "long"
                entry_score = sc
                risk_pct    = RISK_MAP.get(sc, 0.8)
                entry_eq    = equity
                entry_t     = ts_arr[i]
                sl          = c - ATR_SL_MUL * at
                tp          = c + (c - sl) * RR
                score_log.append(sc)

        elif entry_short_arr[i]:
            sc = int(sS_arr[i])
            if sc >= 5:
                pos         = "short"
                entry_score = sc
                risk_pct    = RISK_MAP.get(sc, 0.8)
                entry_eq    = equity
                entry_t     = ts_arr[i]
                sl          = c + ATR_SL_MUL * at
                tp          = c - (sl - c) * RR
                score_log.append(sc)

# ─── Statistics ──────────────────────────────────────────────────────────────
tdf = pd.DataFrame(trades)
if tdf.empty: print("No trades generated."); exit()

tdf['entry_time'] = pd.to_datetime(tdf['entry_time'], unit='ms')
tdf['exit_time']  = pd.to_datetime(tdf['exit_time'],  unit='ms')
tdf.to_csv('xauusd_institutional_trades.csv', index=False)

eq_s = pd.Series(eq_curve)
peak = eq_s.cummax(); dd = (peak - eq_s)/peak * 100
max_dd = dd.max()

wins   = tdf[tdf['status']=='win']
losses = tdf[tdf['status']=='loss']
pnl    = tdf['pnl'].sum()
gp     = wins['pnl'].sum(); gl = abs(losses['pnl'].sum())
wr     = len(wins)/len(tdf)*100
pf     = gp/gl if gl > 0 else float('inf')

tdf['year']  = tdf['exit_time'].dt.year
tdf['month'] = tdf['exit_time'].dt.month
tdf['dow']   = tdf['exit_time'].dt.day_name()
tdf['hour']  = tdf['exit_time'].dt.hour

yearly_pct = (tdf.groupby('year')['pnl'].sum() / INITIAL * 100).round(2)
daily_r    = tdf.set_index('exit_time')['pnl'].resample('D').sum() / INITIAL * 100
sharpe     = (daily_r.mean()/daily_r.std())*np.sqrt(252) if daily_r.std() > 0 else 0

print(f"""
╔══════════════════════════════════════════════════════╗
  INSTITUTIONAL CONFLUENCE STRATEGY v2 — XAUUSD 5Y
╠══════════════════════════════════════════════════════╣
  Total Trades:    {len(tdf)}
  Win Rate:        {wr:.2f}%
  Profit Factor:   {pf:.2f}x
  Gross Profit:    ${gp:,.2f}
  Gross Loss:      ${gl:,.2f}
  Net PnL:         ${pnl:,.2f}  ({pnl/INITIAL*100:.2f}%)
  Final Equity:    ${equity:,.2f}
  Max Drawdown:    {max_dd:.2f}%
  Sharpe Ratio:    {sharpe:.2f}
╠══════════════════════════════════════════════════════╣
  Yearly Breakdown:""")
for yr, v in yearly_pct.items():
    bar = '▓' * min(40, int(abs(v)/1)) + ('  ↑ PROFIT' if v>0 else '  ↓ LOSS')
    print(f"    {yr}:  {v:+.2f}%   {bar}")
print(f"""╠══════════════════════════════════════════════════════╣
  Score Distribution: {dict(sorted(Counter(score_log).items()))}
╚══════════════════════════════════════════════════════╝
""")

# ─── Charts ──────────────────────────────────────────────────────────────────
def calc_pct(g): return round(g['pnl'].sum()/INITIAL*100, 2)
def bar_c(vals):
    arr = np.array(vals, dtype=float)
    if arr.max() == arr.min() or len(arr) == 1: return ['#4CAF50']*len(arr)
    norm = plt.Normalize(arr.min(), arr.max())
    return [plt.cm.RdYlGn(norm(v)) for v in arr]

def styled_bar(ax, x, y, title, rot=0):
    b = ax.bar(x, y, color=bar_c(y), edgecolor='#333', lw=0.5, zorder=3)
    ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.5, zorder=2)
    ax.set_title(title, fontsize=9, fontweight='bold', color='white', pad=5)
    ax.set_ylabel('%', color='#aaa', fontsize=7); ax.tick_params(colors='#bbb', labelsize=7)
    ax.set_facecolor('#0d0d1f'); ax.spines[:].set_color('#333'); ax.grid(axis='y', alpha=0.15, zorder=1)
    if rot: plt.setp(ax.get_xticklabels(), rotation=rot, ha='right', fontsize=6)
    for bar, val in zip(b, y):
        ax.annotate(f"{val:+.1f}%", xy=(bar.get_x()+bar.get_width()/2, val+(0.05 if val>=0 else -0.05)),
                    ha='center', va='bottom' if val>=0 else 'top', fontsize=6, color='white', fontweight='bold')

sns.set_theme(style='whitegrid')
fig = plt.figure(figsize=(24,18), facecolor='#12122a')
fig.suptitle(f"XAUUSD Institutional Confluence Strategy (7-Signal + ATR Stops + Variable Risk)\n"
             f"Net: {pnl/INITIAL*100:+.2f}%  |  Max DD: {max_dd:.1f}%  |  Sharpe: {sharpe:.2f}  |  "
             f"Win: {wr:.1f}%  |  PF: {pf:.2f}x  |  Trades: {len(tdf)}",
             fontsize=12, fontweight='bold', color='white', y=0.99)

gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)
EQ  = pd.to_datetime(df['timestamp'].values[:len(eq_curve)], unit='ms')

# Equity curve
ax0 = fig.add_subplot(gs[0, :])
ax0.plot(EQ, eq_s.values, color='#00e676', lw=1.2)
ax0.fill_between(EQ, INITIAL, eq_s.values, where=eq_s.values>=INITIAL, alpha=0.2, color='#00e676')
ax0.fill_between(EQ, INITIAL, eq_s.values, where=eq_s.values< INITIAL, alpha=0.2, color='#ff5252')
ax0.axhline(INITIAL, color='white', lw=0.8, ls='--', alpha=0.4)
ax0.set_facecolor('#0d0d1f'); ax0.spines[:].set_color('#333'); ax0.tick_params(colors='#bbb',labelsize=8)
ax0.set_title('Equity Curve', fontsize=11, fontweight='bold', color='white', pad=6)
ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax0.grid(alpha=0.12)

# Yearly
ax1 = fig.add_subplot(gs[1,0])
yd  = yearly_pct.reset_index(); yd.columns=['year','pct']
styled_bar(ax1, yd['year'].astype(str), yd['pct'].tolist(), 'Yearly PnL (%)')

# Monthly
ax2 = fig.add_subplot(gs[1,1])
md  = tdf.groupby('month').apply(calc_pct).reset_index(); md.columns=['month','pct']
md['mn'] = md['month'].apply(lambda x: calendar.month_abbr[x])
styled_bar(ax2, md['mn'], md['pct'].tolist(), 'Monthly PnL (%) — Seasonality')

# Hourly
ax3 = fig.add_subplot(gs[1,2])
hd  = tdf.groupby('hour').apply(calc_pct).reset_index(); hd.columns=['hour','pct']
styled_bar(ax3, hd['hour'].astype(str), hd['pct'].tolist(), 'By Hour UTC', rot=45)

# Score PnL
ax4 = fig.add_subplot(gs[2,0])
sd  = tdf.groupby('score').apply(calc_pct).reset_index(); sd.columns=['score','pct']
styled_bar(ax4, sd['score'].astype(str), sd['pct'].tolist(), 'PnL by Confluence Score')

# Win rate per score
ax5 = fig.add_subplot(gs[2,1])
wr5 = tdf.groupby('score').apply(lambda g: len(g[g['status']=='win'])/len(g)*100).reset_index()
wr5.columns = ['score','wr']
b5  = ax5.bar(wr5['score'].astype(str), wr5['wr'], color=[plt.cm.RdYlGn(v/100) for v in wr5['wr']], edgecolor='#333')
ax5.axhline(50, color='white', lw=0.8, ls='--', alpha=0.5)
for bar,val in zip(b5, wr5['wr']): ax5.annotate(f"{val:.0f}%", xy=(bar.get_x()+bar.get_width()/2, val+0.5), ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
ax5.set_title('Win Rate by Confluence Score', fontsize=9, fontweight='bold', color='white', pad=5)
ax5.set_facecolor('#0d0d1f'); ax5.spines[:].set_color('#333'); ax5.tick_params(colors='#bbb', labelsize=8)
ax5.grid(axis='y', alpha=0.15)

# Drawdown
ax6 = fig.add_subplot(gs[2,2])
ax6.fill_between(EQ[:len(dd)], 0, -dd.values, color='#ff5252', alpha=0.5)
ax6.plot(EQ[:len(dd)], -dd.values, color='#ff1744', lw=0.8)
ax6.set_title(f'Drawdown Profile (Max -{max_dd:.1f}%)', fontsize=9, fontweight='bold', color='white', pad=5)
ax6.set_facecolor('#0d0d1f'); ax6.spines[:].set_color('#333'); ax6.tick_params(colors='#bbb', labelsize=7)
ax6.grid(alpha=0.12)

out = '/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/institutional_dashboard.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Dashboard saved → {out}")
