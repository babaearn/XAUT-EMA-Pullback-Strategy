"""
VERIFICATION + BIAS FIX
=======================
Key issues checked:
1. 1H Lookahead Bias: h1 indicators shift by 1 bar before ffill
2. Signal logic audit: print sample entries and check they're valid
3. Trade count sanity check
4. Re-run clean backtest and compare to previous result
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
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
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
    return ml, ema_c(ml,sg)

# ─── Load 5m ─────────────────────────────────────────────────────────────────
print("Loading 5m data...")
df = pd.read_csv("data/xauusd-m5-bid-2021-03-01-2026-03-15.csv")
df['dt']    = pd.to_datetime(df['timestamp'], unit='ms')
df['hour']  = df['dt'].dt.hour
df['dow']   = df['dt'].dt.dayofweek
df['month'] = df['dt'].dt.month

print("Computing 5m indicators...")
df['ema8']  = ema_c(df['close'], 8)
df['ema21'] = ema_c(df['close'], 21)
df['ema50'] = ema_c(df['close'], 50)
df['rsi']   = rsi_c(df['close'], 14)
df['atr']   = atr_c(df['high'], df['low'], df['close'], 14)
ml, sl_m    = macd_c(df['close'])
df['macd_hist'] = ml - sl_m
df['vol_ma'] = df['volume'].rolling(20).mean()

# ─── Resample to 1H — but SHIFT BY 1 bar to prevent lookahead ───────────────
#  BEFORE fix: h1['ema21'] for the 12:00 bar contained all 12:xx candles
#  AFTER fix:  We use .shift(1) so the 12:00 bar gets 11:00's indicator value
print("Computing 1H indicators (with lookahead fix: shift by 1 closed bar)...")
df.set_index('dt', inplace=True)
h1 = df[['open','high','low','close','volume']].resample('1H').agg(
    {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

h1['ema21']  = ema_c(h1['close'], 21)
h1['ema200'] = ema_c(h1['close'], 200)
h1['rsi']    = rsi_c(h1['close'], 14)
adx_v, pdi, ndi = adx_c(h1['high'], h1['low'], h1['close'], 14)
h1['adx'] = adx_v; h1['pdi'] = pdi; h1['ndi'] = ndi

# ✅ KEY FIX: shift h1 indicators by 1 closed bar BEFORE forward-filling
#    This guarantees we only use the PREVIOUS completed 1H bar at any 5m entry
h1_shifted = h1[['ema21','ema200','rsi','adx','pdi','ndi']].shift(1)

df['h1_ema21']  = h1_shifted['ema21'].reindex(df.index,  method='ffill')
df['h1_ema200'] = h1_shifted['ema200'].reindex(df.index, method='ffill')
df['h1_rsi']    = h1_shifted['rsi'].reindex(df.index,    method='ffill')
df['h1_adx']    = h1_shifted['adx'].reindex(df.index,    method='ffill')
df['h1_pdi']    = h1_shifted['pdi'].reindex(df.index,    method='ffill')
df['h1_ndi']    = h1_shifted['ndi'].reindex(df.index,    method='ffill')
df.reset_index(inplace=True)

# ─── Vectorized Scoring ───────────────────────────────────────────────────────
d = df
s1L = (d['ema8']>d['ema21'])&(d['ema21']>d['ema50'])
s2L = (d['rsi']>50)&(d['rsi']<70)
s3L = d['macd_hist']>0
s4L = d['h1_ema21']>d['h1_ema200']
s5L = (d['h1_adx']>20)&(d['h1_pdi']>d['h1_ndi'])
s6  = d['volume']>d['vol_ma']
s7L = d['h1_rsi']>50

s1S = (d['ema8']<d['ema21'])&(d['ema21']<d['ema50'])
s2S = (d['rsi']>30)&(d['rsi']<50)
s3S = d['macd_hist']<0
s4S = d['h1_ema21']<d['h1_ema200']
s5S = (d['h1_adx']>20)&(d['h1_ndi']>d['h1_pdi'])
s7S = d['h1_rsi']<50

df['score_long']  = s1L.astype(int)+s2L.astype(int)+s3L.astype(int)+\
                    s4L.astype(int)+s5L.astype(int)+s6.astype(int)+s7L.astype(int)
df['score_short'] = s1S.astype(int)+s2S.astype(int)+s3S.astype(int)+\
                    s4S.astype(int)+s5S.astype(int)+s6.astype(int)+s7S.astype(int)

TAP = 0.20
df['tap_zone']     = df['ema21'] * TAP / 100
df['in_long_tap']  = (df['close']>df['ema21'])&(df['close']<=df['ema21']+df['tap_zone'])
df['in_short_tap'] = (df['close']<df['ema21'])&(df['close']>=df['ema21']-df['tap_zone'])
df['valid_session']= (df['dow'].isin([1,2,3]))&(df['hour']>=8)&(df['hour']<19)&(df['month']!=6)

df['entry_long']  = df['in_long_tap'] &(df['score_long']>=5) &df['valid_session']
df['entry_short'] = df['in_short_tap']&(df['score_short']>=5)&df['valid_session']

# ─── SANITY CHECK: verify a few signals manually ──────────────────────────────
print("\n--- SIGNAL SANITY CHECK (first 5 detected long entries) ---")
sample_longs = df[df['entry_long']].head(5)[
    ['dt','close','ema21','ema8','ema50','rsi','macd_hist',
     'h1_ema21','h1_ema200','h1_rsi','h1_adx','h1_pdi','h1_ndi',
     'score_long','volume','vol_ma','dow','hour','month','valid_session']
]
for _, row in sample_longs.iterrows():
    tap_z = row['ema21'] * TAP / 100   # compute inline
    print(f"\n  [{row['dt']}] LONG entry")
    print(f"    Price={row['close']:.2f} | EMA21={row['ema21']:.2f} | "
          f"Tap zone: [{row['ema21']:.2f} – {row['ema21']+tap_z:.2f}]")
    print(f"    EMA stack: {row['ema8']:.2f} > {row['ema21']:.2f} > {row['ema50']:.2f} "
          f"= {'✅' if row['ema8']>row['ema21']>row['ema50'] else '❌'}")
    print(f"    RSI={row['rsi']:.1f} in 50-70 = {'✅' if 50<row['rsi']<70 else '❌'}")
    print(f"    MACD hist={row['macd_hist']:.4f} > 0 = {'✅' if row['macd_hist']>0 else '❌'}")
    print(f"    1H EMA21({row['h1_ema21']:.2f}) > EMA200({row['h1_ema200']:.2f}) "
          f"= {'✅' if row['h1_ema21']>row['h1_ema200'] else '❌'}")
    print(f"    ADX={row['h1_adx']:.1f}>20 & PDI({row['h1_pdi']:.1f})>NDI({row['h1_ndi']:.1f}) "
          f"= {'✅' if row['h1_adx']>20 and row['h1_pdi']>row['h1_ndi'] else '❌'}")
    print(f"    Vol={row['volume']:.3f} > Vol_MA={row['vol_ma']:.3f} "
          f"= {'✅' if row['volume']>row['vol_ma'] else '❌'}")
    print(f"    1H RSI={row['h1_rsi']:.1f} > 50 = {'✅' if row['h1_rsi']>50 else '❌'}")
    print(f"    Score={row['score_long']} | Day={row['dow']} Hour={row['hour']} "
          f"Month={row['month']} | Valid={'✅' if row['valid_session'] else '❌'}")

# ─── Simulation (clean, honest) ───────────────────────────────────────────────
print("\nRunning verified backtest...")

INITIAL  = 10_000.0
RR       = 2.5
RISK_MAP = {5: 0.008, 6: 0.012, 7: 0.015}
ATR_SL   = 1.5

equity = INITIAL; pos = None; sl = tp = 0.0
risk_pct = 0.008; entry_score = 5; entry_eq = INITIAL; entry_t = None
trades = []; eq_curve = []; score_log = []

entry_long_arr  = df['entry_long'].values
entry_short_arr = df['entry_short'].values
close_a = df['close'].values; high_a = df['high'].values
low_a   = df['low'].values;   atr_a  = df['atr'].values
ts_a    = df['timestamp'].values
sL_a    = df['score_long'].values; sS_a = df['score_short'].values

for i in range(300, len(df)):
    eq_curve.append(equity)
    if equity <= 100: break

    c = close_a[i]; h = high_a[i]; lo = low_a[i]
    at = atr_a[i] if (not np.isnan(atr_a[i]) and atr_a[i]>0) else c*0.003
    risk_amt = entry_eq * risk_pct

    if pos == "long":
        if lo <= sl:
            equity -= risk_amt
            trades.append({'entry_time':entry_t,'exit_time':ts_a[i],
                           'status':'loss','pnl':-risk_amt,'score':entry_score,'type':'long'})
            pos = None
        elif h >= tp:
            equity += risk_amt * RR
            trades.append({'entry_time':entry_t,'exit_time':ts_a[i],
                           'status':'win','pnl':risk_amt*RR,'score':entry_score,'type':'long'})
            pos = None

    elif pos == "short":
        if h >= sl:
            equity -= risk_amt
            trades.append({'entry_time':entry_t,'exit_time':ts_a[i],
                           'status':'loss','pnl':-risk_amt,'score':entry_score,'type':'short'})
            pos = None
        elif lo <= tp:
            equity += risk_amt * RR
            trades.append({'entry_time':entry_t,'exit_time':ts_a[i],
                           'status':'win','pnl':risk_amt*RR,'score':entry_score,'type':'short'})
            pos = None

    if pos is None:
        if entry_long_arr[i]:
            sc = int(sL_a[i])
            if sc >= 5:
                pos=   "long"; entry_score=sc; risk_pct=RISK_MAP.get(sc,0.008)
                entry_eq=INITIAL; entry_t=ts_a[i]   # FIXED: always use initial equity base
                sl=c-ATR_SL*at; tp=c+(c-sl)*RR; score_log.append(sc)
        elif entry_short_arr[i]:
            sc = int(sS_a[i])
            if sc >= 5:
                pos=   "short"; entry_score=sc; risk_pct=RISK_MAP.get(sc,0.008)
                entry_eq=INITIAL; entry_t=ts_a[i]   # FIXED: always use initial equity base
                sl=c+ATR_SL*at; tp=c-(sl-c)*RR; score_log.append(sc)

# ─── Results ─────────────────────────────────────────────────────────────────
tdf = pd.DataFrame(trades)
if tdf.empty: print("No trades."); exit()

tdf['entry_time'] = pd.to_datetime(tdf['entry_time'], unit='ms')
tdf['exit_time']  = pd.to_datetime(tdf['exit_time'],  unit='ms')

# FIXED risk recalculation (already fixed in entry_eq=INITIAL above)
wins   = tdf[tdf['status']=='win']
losses = tdf[tdf['status']=='loss']
gp     = wins['pnl'].sum(); gl = abs(losses['pnl'].sum())
pf     = gp/gl if gl > 0 else float('inf')
net    = tdf['pnl'].sum(); wr = len(wins)/len(tdf)*100

eq_s   = INITIAL + tdf.sort_values('exit_time')['pnl'].cumsum()
peak   = eq_s.cummax(); dd = (peak-eq_s)/peak*100; max_dd = dd.max()
final_eq = INITIAL + net

daily_r = tdf.set_index('exit_time')['pnl'].resample('D').sum()/INITIAL*100
sharpe  = (daily_r.mean()/daily_r.std())*np.sqrt(252) if daily_r.std()>0 else 0

tdf['year']  = tdf['exit_time'].dt.year
tdf['month'] = tdf['exit_time'].dt.month
tdf['dow']   = tdf['exit_time'].dt.day_name()
tdf['hour']  = tdf['exit_time'].dt.hour
tdf['week']  = tdf['exit_time'].dt.isocalendar().week.astype(int)

yearly_pct = (tdf.groupby('year')['pnl'].sum()/INITIAL*100).round(2)

wk_counts     = tdf.groupby(tdf['exit_time'].dt.to_period('W')).size()
avg_wk        = wk_counts.mean()
yr_wk_avg     = tdf.groupby(['year', tdf['exit_time'].dt.to_period('W')]).size().groupby('year').mean().round(1)

avg_win  = wins['pnl'].mean()
avg_loss = losses['pnl'].mean()
expectancy = (wr/100 * avg_win) + ((1-wr/100)*abs(avg_loss))

print(f"""
╔═══════════════════════════════════════════════════════════════╗
  VERIFIED BACKTEST — LOOKAHEAD BIAS FIXED — NON-COMPOUNDING
  (Fixed base $10,000 · 1H shift(1) fix applied)
╠═══════════════════════════════════════════════════════════════╣
  Total Trades:      {len(tdf):,}
  Win Rate:          {wr:.2f}%
  Profit Factor:     {pf:.2f}x
  Avg Win:           ${avg_win:,.2f}
  Avg Loss:          ${avg_loss:,.2f}
  Expectancy/Trade:  ${expectancy:+.2f}
╠═══════════════════════════════════════════════════════════════╣
  Gross Profit:      ${gp:,.2f}
  Gross Loss:       -${gl:,.2f}
  Net PnL:           ${net:,.2f}  ({net/INITIAL*100:.2f}%)
  Final Equity:      ${final_eq:,.2f}
  Max Drawdown:      {max_dd:.2f}%
  Sharpe Ratio:      {sharpe:.2f}
╠═══════════════════════════════════════════════════════════════╣
  Yearly Net % (fixed $10k base):""")
for yr,v in yearly_pct.items():
    neg = v < 0
    bar = ('▓' if not neg else '░') * min(40, int(abs(v)/2))
    print(f"    {yr}:  {v:+.2f}%  {bar}  {'↓ LOSS' if neg else '↑ PROFIT'}")

print(f"""╠═══════════════════════════════════════════════════════════════╣
  WEEKLY FREQUENCY:
    Avg trades/week:  {avg_wk:.1f}
    By year:""")
for yr,v in yr_wk_avg.items():
    print(f"      {yr}: {v:.1f} trades/week")

print(f"""╠═══════════════════════════════════════════════════════════════╣
  Score Distribution: {dict(sorted(Counter(score_log).items()))}
╠═══════════════════════════════════════════════════════════════╣
  VERIFICATION STATUS:
    [✅] 1H Lookahead bias: FIXED (shift(1) applied)
    [✅] Position gating: Only entry when pos is None
    [✅] Fixed base equity: entry_eq = $10,000 always
    [✅] Signal sanity: printed above — check console
    [{'✅' if wr > 20 else '❌'}] Win rate > 20%: {wr:.2f}%
    [{'✅' if pf > 1.0 else '❌'}] Profit Factor > 1.0: {pf:.2f}x
    [{'✅' if max_dd < 30 else '❌'}] Max DD < 30%: {max_dd:.2f}%
    [{'✅' if sharpe > 0.8 else '❌'}] Sharpe > 0.8: {sharpe:.2f}
╚═══════════════════════════════════════════════════════════════╝
""")

tdf.to_csv('xauusd_verified_trades.csv', index=False)
print("Verified trades saved → xauusd_verified_trades.csv")

# ─── Charts ──────────────────────────────────────────────────────────────────
def calc_pct(g): return round(g['pnl'].sum()/INITIAL*100, 2)
def bar_c(vals):
    arr=np.array(vals,dtype=float)
    if len(arr)==0: return []
    if arr.max()==arr.min(): return ['#4CAF50']*len(arr)
    norm=plt.Normalize(arr.min(),arr.max())
    return [plt.cm.RdYlGn(norm(v)) for v in arr]

def styled_bar(ax, x, y, title, rot=0, fs=8):
    y=list(y); b=ax.bar(x,y,color=bar_c(y),edgecolor='#444',lw=0.5,zorder=3)
    ax.axhline(0,color='white',lw=0.9,ls='--',alpha=0.5,zorder=2)
    ax.set_title(title,fontsize=9,fontweight='bold',color='white',pad=6)
    ax.set_ylabel('Net %',color='#aaa',fontsize=7); ax.tick_params(colors='#bbb',labelsize=fs)
    ax.set_facecolor('#0d0d1f'); ax.spines[:].set_color('#333'); ax.grid(axis='y',alpha=0.15,zorder=1)
    if rot: plt.setp(ax.get_xticklabels(),rotation=rot,ha='right',fontsize=fs-1)
    for bar,val in zip(b,y):
        ax.annotate(f"{val:+.1f}%",
                    xy=(bar.get_x()+bar.get_width()/2, val+(0.05 if val>=0 else -0.05)),
                    ha='center',va='bottom' if val>=0 else 'top',fontsize=6,color='white',fontweight='bold')

sns.set_theme(style='whitegrid')
fig=plt.figure(figsize=(22,22),facecolor='#1a1a2e')
fig.suptitle(
    f"XAU/USD Institutional Strategy — VERIFIED Report (Lookahead Fix Applied)\n"
    f"Net: {net/INITIAL*100:+.2f}% · Final: ${final_eq:,.0f} · Max DD: {max_dd:.1f}% · "
    f"Sharpe: {sharpe:.2f} · WR: {wr:.1f}% · PF: {pf:.2f}× · Trades: {len(tdf):,}",
    fontsize=12,fontweight='bold',color='white',y=0.995)

gs=gridspec.GridSpec(4,2,figure=fig,hspace=0.50,wspace=0.32)

# Equity curve
ax0=fig.add_subplot(gs[0,:])
eq_times = tdf.sort_values('exit_time')['exit_time'].values
ax0.plot(eq_times, eq_s.values, color='#00e676', lw=1.2)
ax0.fill_between(eq_times, INITIAL, eq_s.values, where=eq_s.values>=INITIAL, alpha=0.18, color='#00e676')
ax0.fill_between(eq_times, INITIAL, eq_s.values, where=eq_s.values< INITIAL, alpha=0.18, color='#ff5252')
ax0.axhline(INITIAL, color='white', lw=0.8, ls='--', alpha=0.4)
for yr in tdf['year'].unique():
    first=tdf[tdf['year']==yr]['exit_time'].iloc[0]
    yr_v=yearly_pct.get(yr,0)
    ax0.axvline(first, color='#ffffff18', lw=0.7, ls=':')
    ax0.text(first, INITIAL*1.01, f"{yr}\n{yr_v:+.0f}%", fontsize=8,
             color='#00e676' if yr_v>=0 else '#ff5252', fontweight='bold', va='bottom')
ax0.set_title(f'Equity Curve (NON-COMPOUNDING · Fixed $10K base · Lookahead Fixed)  '
              f'·  Final ${final_eq:,.0f}  ·  Max DD {max_dd:.1f}%', fontsize=10, fontweight='bold', color='white', pad=7)
ax0.set_facecolor('#0d0d1f'); ax0.spines[:].set_color('#333'); ax0.tick_params(colors='#bbb', labelsize=8)
ax0.set_ylabel('Equity ($)', color='#aaa', fontsize=9)
ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax0.grid(alpha=0.12)

# Session
ax1=fig.add_subplot(gs[1,:])
def session(h):
    if 0<=h<8: return 'Asian\n(00-08)'
    if 8<=h<13: return 'London Open\n(08-13)'
    if 13<=h<17: return 'NY/London\nOverlap (13-17)'
    if 17<=h<19: return 'NY Session\n(17-19)'
    return 'Other'
tdf['session']=tdf['hour'].apply(session)
sess_order=['Asian\n(00-08)','London Open\n(08-13)','NY/London\nOverlap (13-17)','NY Session\n(17-19)','Other']
sd=tdf.groupby('session').apply(calc_pct).reindex(sess_order).fillna(0).reset_index()
sd.columns=['session','pct']
styled_bar(ax1, sd['session'], sd['pct'].tolist(), 'Net Profit (%) by Session — VERIFIED', fs=9)
for bar,cnt in zip(ax1.patches, tdf.groupby('session').size().reindex(sess_order).fillna(0)):
    ax1.text(bar.get_x()+bar.get_width()/2, -1.5, f"{int(cnt)} trades",
             ha='center', va='top', fontsize=8, color='#aaa')

# Day of week
ax2=fig.add_subplot(gs[2,0])
day_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dd2=tdf.groupby('dow').apply(calc_pct).reindex(day_order).fillna(0).reset_index(); dd2.columns=['day','pct']
styled_bar(ax2, dd2['day'], dd2['pct'].tolist(), 'Net Profit (%) by Day',rot=30)

# Monthly
ax3=fig.add_subplot(gs[2,1])
md=tdf.groupby('month').apply(calc_pct).reset_index(); md.columns=['month','pct']
md['mn']=md['month'].apply(lambda x: calendar.month_abbr[x])
styled_bar(ax3, md['mn'], md['pct'].tolist(), 'Monthly PnL — Seasonality')

# Yearly
ax4=fig.add_subplot(gs[3,0])
yd=yearly_pct.reset_index(); yd.columns=['year','pct']
styled_bar(ax4, yd['year'].astype(str), yd['pct'].tolist(), 'Yearly Net %  (Fixed $10K · True Annual Return)')

# Weekly heatmap
ax5=fig.add_subplot(gs[3,1])
wk_m=tdf.groupby(['year','week']).apply(calc_pct).unstack(level=1).fillna(0)
wk_m.index=wk_m.index.astype(str)
vmax=max(abs(wk_m.max().max()), abs(wk_m.min().min()), 0.01)
sns.heatmap(wk_m, cmap='RdYlGn', center=0, vmin=-vmax, vmax=vmax,
            ax=ax5, linewidths=0.2, cbar_kws={'label':'Net %','shrink':0.8}, annot=False)
ax5.set_title(f'Weekly PnL Heatmap (Avg {avg_wk:.1f} trades/week)',
              fontsize=9,fontweight='bold',color='white',pad=6)
ax5.set_facecolor('#0d0d1f'); ax5.tick_params(colors='#bbb', labelsize=7)

out='/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/verified_report.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Verified dashboard → {out}")
