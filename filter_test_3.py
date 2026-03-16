"""
FILTER TEST #3 — INVERSE VOLATILITY POSITION SIZING
=====================================================
Builds on: Filter #1 (Daily EMA200) — ACCEPTED
Active baseline: Trades=1,742 · WR=30.08% · PF=1.07x · Net=+88.60% · MaxDD=33.95% · Sharpe=0.51

Filter #3 Rule:
  adjusted_risk_pct = base_risk_pct × (atr_avg50 / current_atr)
  → Clamped between 0.4× base (floor) and 2.0× base (ceiling)

Meaning:
  High ATR (volatile): ratio < 1 → smaller size → less exposure in chaotic markets
  Low ATR  (calm):     ratio > 1 → slightly larger size → more efficient use of capital

Base risks (from score):
  Score 5 → 0.8%  · Score 6 → 1.2%  · Score 7 → 1.5%

Expected outcome:
  Smoother equity curve → better Sharpe
  Reduced losses during high-ATR stop-out periods
  Net PnL might change slightly but risk-adjusted metrics should improve
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, seaborn as sns, calendar
from collections import Counter

def ema_c(s,p): return s.ewm(span=p,adjust=False).mean()
def rsi_c(s,p=14):
    d=s.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    ag=g.ewm(alpha=1/p,adjust=False).mean(); al=l.ewm(alpha=1/p,adjust=False).mean()
    return 100-100/(1+ag/al.replace(0,1e-9))
def atr_c(h,l,c,p=14):
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
    return tr.ewm(alpha=1/p,adjust=False).mean()
def adx_c(h,l,c,p=14):
    up=h.diff(); dn=-l.diff()
    pdm=up.where((up>dn)&(up>0),0.0); ndm=dn.where((dn>up)&(dn>0),0.0)
    at=atr_c(h,l,c,p)
    pdi=100*pdm.ewm(alpha=1/p,adjust=False).mean()/at.replace(0,1e-9)
    ndi=100*ndm.ewm(alpha=1/p,adjust=False).mean()/at.replace(0,1e-9)
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,1e-9)
    return dx.ewm(alpha=1/p,adjust=False).mean(),pdi,ndi
def macd_c(s,f=12,sl=26,sg=9):
    ml=ema_c(s,f)-ema_c(s,sl); return ml,ema_c(ml,sg)

PREV={'label':'F#1 (Daily EMA200)','trades':1742,'wr':30.08,'pf':1.07,
      'net_pct':88.60,'max_dd':33.95,'sharpe':0.51}

INITIAL=10_000.0; RR=2.5; ATR_SL=1.5
BASE_RISK={5:0.008, 6:0.012, 7:0.015}
VOL_FLOOR=0.4   # min multiplier on base risk
VOL_CAP  =2.0   # max multiplier on base risk

print("Loading & computing indicators...")
df=pd.read_csv("data/xauusd-m5-bid-2021-03-01-2026-03-15.csv")
df['dt']=pd.to_datetime(df['timestamp'],unit='ms')
df['hour']=df['dt'].dt.hour; df['dow']=df['dt'].dt.dayofweek; df['month']=df['dt'].dt.month

df['ema8']=ema_c(df['close'],8); df['ema21']=ema_c(df['close'],21); df['ema50']=ema_c(df['close'],50)
df['rsi']=rsi_c(df['close'],14); df['atr']=atr_c(df['high'],df['low'],df['close'],14)
ml,sl_m=macd_c(df['close']); df['macd_hist']=ml-sl_m
df['vol_ma']=df['volume'].rolling(20).mean()
df['atr_avg50']=df['atr'].rolling(50).mean()   # ← used for vol ratio

df.set_index('dt',inplace=True)

# 1H bias (shift-1)
h1=df[['open','high','low','close','volume']].resample('1H').agg(
    {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
h1['ema21']=ema_c(h1['close'],21); h1['ema200']=ema_c(h1['close'],200)
h1['rsi']=rsi_c(h1['close'],14)
adx_v,pdi,ndi=adx_c(h1['high'],h1['low'],h1['close'],14)
h1['adx']=adx_v; h1['pdi']=pdi; h1['ndi']=ndi
h1_s=h1[['ema21','ema200','rsi','adx','pdi','ndi']].shift(1)
df['h1_ema21']=h1_s['ema21'].reindex(df.index,method='ffill')
df['h1_ema200']=h1_s['ema200'].reindex(df.index,method='ffill')
df['h1_rsi']=h1_s['rsi'].reindex(df.index,method='ffill')
df['h1_adx']=h1_s['adx'].reindex(df.index,method='ffill')
df['h1_pdi']=h1_s['pdi'].reindex(df.index,method='ffill')
df['h1_ndi']=h1_s['ndi'].reindex(df.index,method='ffill')

# Daily EMA200 (Filter #1)
d1=df[['open','high','low','close','volume']].resample('1D').agg(
    {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
d1['ema200']=ema_c(d1['close'],200)
d1_s=d1[['ema200']].shift(1)
df['d1_ema200']=d1_s['ema200'].reindex(df.index,method='ffill')
df.reset_index(inplace=True)

# Scoring
d=df
s1L=(d['ema8']>d['ema21'])&(d['ema21']>d['ema50'])
s2L=(d['rsi']>50)&(d['rsi']<70)
s3L=d['macd_hist']>0
s4L=d['h1_ema21']>d['h1_ema200']
s5L=(d['h1_adx']>20)&(d['h1_pdi']>d['h1_ndi'])
s6=d['volume']>d['vol_ma']
s7L=d['h1_rsi']>50
s1S=(d['ema8']<d['ema21'])&(d['ema21']<d['ema50'])
s2S=(d['rsi']>30)&(d['rsi']<50)
s3S=d['macd_hist']<0
s4S=d['h1_ema21']<d['h1_ema200']
s5S=(d['h1_adx']>20)&(d['h1_ndi']>d['h1_pdi'])
s7S=d['h1_rsi']<50

df['score_long']=s1L.astype(int)+s2L.astype(int)+s3L.astype(int)+\
                 s4L.astype(int)+s5L.astype(int)+s6.astype(int)+s7L.astype(int)
df['score_short']=s1S.astype(int)+s2S.astype(int)+s3S.astype(int)+\
                  s4S.astype(int)+s5S.astype(int)+s6.astype(int)+s7S.astype(int)

TAP=0.20; df['tap_zone']=df['ema21']*TAP/100
df['in_long_tap']=(df['close']>df['ema21'])&(df['close']<=df['ema21']+df['tap_zone'])
df['in_short_tap']=(df['close']<df['ema21'])&(df['close']>=df['ema21']-df['tap_zone'])
df['valid_session']=(df['dow'].isin([1,2,3]))&(df['hour']>=8)&(df['hour']<19)&(df['month']!=6)
df['macro_long']=df['close']>df['d1_ema200']
df['macro_short']=df['close']<df['d1_ema200']

df['entry_long']=(df['in_long_tap']&(df['score_long']>=5)&df['valid_session']&df['macro_long'])
df['entry_short']=(df['in_short_tap']&(df['score_short']>=5)&df['valid_session']&df['macro_short'])

# ── Simulation with Inverse Vol Sizing ───────────────────────────────────────
print("Running backtest with Filter #1 + Inverse Vol Sizing...")

equity=INITIAL; pos=None; sl=tp=0.0
risk_pct=0.008; entry_score=5; entry_t=None
trades=[]; score_log=[]; risk_used_log=[]

eL=df['entry_long'].values; eS=df['entry_short'].values
c_a=df['close'].values; h_a=df['high'].values
lo_a=df['low'].values; at_a=df['atr'].values
avg50_a=df['atr_avg50'].values
ts_a=df['timestamp'].values
sL_a=df['score_long'].values; sS_a=df['score_short'].values

for i in range(300,len(df)):
    if equity<=100: break
    c=c_a[i]; h=h_a[i]; lo=lo_a[i]
    at=at_a[i] if (not np.isnan(at_a[i]) and at_a[i]>0) else c*0.003
    risk_amt=INITIAL*risk_pct   # always off fixed INITIAL

    if pos=="long":
        if lo<=sl:
            equity-=risk_amt
            trades.append({'entry_time':entry_t,'exit_time':ts_a[i],
                           'status':'loss','pnl':-risk_amt,'score':entry_score,
                           'type':'long','risk_pct':risk_pct})
            pos=None
        elif h>=tp:
            equity+=risk_amt*RR
            trades.append({'entry_time':entry_t,'exit_time':ts_a[i],
                           'status':'win','pnl':risk_amt*RR,'score':entry_score,
                           'type':'long','risk_pct':risk_pct})
            pos=None
    elif pos=="short":
        if h>=sl:
            equity-=risk_amt
            trades.append({'entry_time':entry_t,'exit_time':ts_a[i],
                           'status':'loss','pnl':-risk_amt,'score':entry_score,
                           'type':'short','risk_pct':risk_pct})
            pos=None
        elif lo<=tp:
            equity+=risk_amt*RR
            trades.append({'entry_time':entry_t,'exit_time':ts_a[i],
                           'status':'win','pnl':risk_amt*RR,'score':entry_score,
                           'type':'short','risk_pct':risk_pct})
            pos=None

    if pos is None:
        sc_entry = None
        if eL[i]: sc_entry = int(sL_a[i]); direction = "long"
        elif eS[i]: sc_entry = int(sS_a[i]); direction = "short"

        if sc_entry and sc_entry >= 5:
            # ✅ FILTER #3: Compute inverse volatility-adjusted risk
            base_r = BASE_RISK.get(sc_entry, 0.008)
            atr_avg = avg50_a[i] if (not np.isnan(avg50_a[i]) and avg50_a[i]>0) else at
            if atr_avg > 0:
                vol_ratio = atr_avg / at          # < 1 when volatile, > 1 when calm
                vol_ratio = np.clip(vol_ratio, VOL_FLOOR, VOL_CAP)
            else:
                vol_ratio = 1.0
            risk_pct  = base_r * vol_ratio        # ← adjusted risk

            entry_score=sc_entry; entry_t=ts_a[i]
            if direction=="long":
                pos="long"; sl=c-ATR_SL*at; tp=c+(c-sl)*RR
            else:
                pos="short"; sl=c+ATR_SL*at; tp=c-(sl-c)*RR
            score_log.append(sc_entry); risk_used_log.append(risk_pct)

# ── Stats ─────────────────────────────────────────────────────────────────────
tdf=pd.DataFrame(trades)
tdf['entry_time']=pd.to_datetime(tdf['entry_time'],unit='ms')
tdf['exit_time']=pd.to_datetime(tdf['exit_time'],unit='ms')
tdf['year']=tdf['exit_time'].dt.year; tdf['month']=tdf['exit_time'].dt.month
tdf['dow']=tdf['exit_time'].dt.day_name(); tdf['hour']=tdf['exit_time'].dt.hour
tdf['week']=tdf['exit_time'].dt.isocalendar().week.astype(int)
tdf.to_csv('xauusd_filter3_trades.csv',index=False)

wins=tdf[tdf['status']=='win']; losses=tdf[tdf['status']=='loss']
gp=wins['pnl'].sum(); gl=abs(losses['pnl'].sum())
pf=gp/gl if gl>0 else float('inf')
net=tdf['pnl'].sum(); wr=len(wins)/len(tdf)*100

eq_s=INITIAL+tdf.sort_values('exit_time')['pnl'].cumsum()
peak=eq_s.cummax(); dd=(peak-eq_s)/peak*100; max_dd=dd.max()
final_eq=INITIAL+net

daily_r=tdf.set_index('exit_time')['pnl'].resample('D').sum()/INITIAL*100
sharpe=(daily_r.mean()/daily_r.std())*np.sqrt(252) if daily_r.std()>0 else 0

yearly_pct=(tdf.groupby('year')['pnl'].sum()/INITIAL*100).round(2)
wk_counts=tdf.groupby(tdf['exit_time'].dt.to_period('W')).size()

avg_risk_used = np.mean(risk_used_log) * 100

def chk(new,old,hb=True):
    d=new-old; icon=('✅' if d>0 else '❌') if hb else ('✅' if d<0 else '❌')
    return f"{icon} {d:+.2f}"

print(f"""
╔══════════════════════════════════════════════════════════════════╗
  FILTER #3: INVERSE VOLATILITY SIZING
  avg_risk_pct actually used: {avg_risk_used:.3f}%  (base Score5=0.8% Score6=1.2% Score7=1.5%)
  Vol clamp: floor={VOL_FLOOR}× base | cap={VOL_CAP}× base
╠══════════════════════╦═════════════╦════════════════════════════╣
  Metric              │ F#1 Baseline│ F#1 + F#3 Result
╠══════════════════════╬═════════════╬════════════════════════════╣
  Total Trades        │  1,742      │  {len(tdf):,}  (same entries)
  Win Rate            │  30.08%     │  {wr:.2f}%  {chk(wr,30.08)}
  Profit Factor       │  1.07x      │  {pf:.2f}x  {chk(pf,1.07)}
  Net PnL             │  +88.60%    │  {net/INITIAL*100:+.2f}%  {chk(net/INITIAL*100,88.60)}
  Final Equity        │  $18,860    │  ${final_eq:,.0f}
  Max Drawdown        │  33.95%     │  {max_dd:.2f}%  {chk(max_dd,33.95,False)}
  Sharpe Ratio        │  0.51       │  {sharpe:.2f}  {chk(sharpe,0.51)}
  Avg Trades/Week     │  8.3        │  {wk_counts.mean():.1f}  (same)
╠══════════════════════╩═════════════╩════════════════════════════╣
  Yearly Breakdown:""")

prev_yr={'2021':-5.60,'2022':24.55,'2023':-1.45,'2024':54.60,'2025':23.25,'2026':-6.75}
for yr,v in yearly_pct.items():
    bv=prev_yr.get(str(yr),0)
    icon='✅' if v>bv else '❌'
    print(f"    {yr}: {v:+.2f}%  (was {bv:+.2f}%)  {icon}")

print(f"""╠══════════════════════════════════════════════════════════════════╣
  Risk Distribution (actual % used per trade):
    min={min(risk_used_log)*100:.3f}%  max={max(risk_used_log)*100:.3f}%  
    avg={avg_risk_used:.3f}%  median={np.median(risk_used_log)*100:.3f}%
  Score distribution: {dict(sorted(Counter(score_log).items()))}
╚══════════════════════════════════════════════════════════════════╝
""")

# ── Chart ─────────────────────────────────────────────────────────────────────
def calc_pct(g): return round(g['pnl'].sum()/INITIAL*100,2)
def bar_c(vals):
    arr=np.array(vals,dtype=float)
    if len(arr)==0: return []
    if arr.max()==arr.min(): return ['#4CAF50']*len(arr)
    norm=plt.Normalize(arr.min(),arr.max())
    return [plt.cm.RdYlGn(norm(v)) for v in arr]

def styled_bar(ax,x,y,title,rot=0,fs=8):
    y=list(y); b=ax.bar(x,y,color=bar_c(y),edgecolor='#444',lw=0.5,zorder=3)
    ax.axhline(0,color='white',lw=0.9,ls='--',alpha=0.5,zorder=2)
    ax.set_title(title,fontsize=9,fontweight='bold',color='white',pad=6)
    ax.set_ylabel('Net %',color='#aaa',fontsize=7); ax.tick_params(colors='#bbb',labelsize=fs)
    ax.set_facecolor('#0d0d1f'); ax.spines[:].set_color('#333'); ax.grid(axis='y',alpha=0.15,zorder=1)
    if rot: plt.setp(ax.get_xticklabels(),rotation=rot,ha='right',fontsize=fs-1)
    for bar,val in zip(b,y):
        ax.annotate(f"{val:+.1f}%",
                    xy=(bar.get_x()+bar.get_width()/2,val+(0.05 if val>=0 else -0.05)),
                    ha='center',va='bottom' if val>=0 else 'top',fontsize=6,color='white',fontweight='bold')

sns.set_theme(style='whitegrid')
fig=plt.figure(figsize=(22,20),facecolor='#1a1a2e')
fig.suptitle(
    f"Filter #3: Inverse Volatility Sizing  [avg_risk={avg_risk_used:.3f}%  floor={VOL_FLOOR}× cap={VOL_CAP}×]\n"
    f"Net: {net/INITIAL*100:+.2f}%  ·  MaxDD: {max_dd:.1f}%  ·  Sharpe: {sharpe:.2f}  ·  "
    f"WR: {wr:.1f}%  ·  PF: {pf:.2f}×  ·  Trades: {len(tdf):,}\n"
    f"vs F#1 Baseline → Net: +88.60%  ·  MaxDD: 33.95%  ·  Sharpe: 0.51",
    fontsize=12,fontweight='bold',color='white',y=0.995)

gs=gridspec.GridSpec(4,2,figure=fig,hspace=0.50,wspace=0.32)

ax0=fig.add_subplot(gs[0,:])
eq_times=tdf.sort_values('exit_time')['exit_time'].values
ax0.plot(eq_times,eq_s.values,color='#00e676',lw=1.2)
ax0.fill_between(eq_times,INITIAL,eq_s.values,where=eq_s.values>=INITIAL,alpha=0.18,color='#00e676')
ax0.fill_between(eq_times,INITIAL,eq_s.values,where=eq_s.values< INITIAL,alpha=0.18,color='#ff5252')
ax0.axhline(INITIAL,color='white',lw=0.8,ls='--',alpha=0.4)
for yr in tdf['year'].unique():
    first=tdf[tdf['year']==yr]['exit_time'].iloc[0]; yv=yearly_pct.get(yr,0)
    ax0.axvline(first,color='#ffffff18',lw=0.7,ls=':')
    ax0.text(first,INITIAL*1.005,f"{yr}\n{yv:+.0f}%",fontsize=8,
             color='#00e676' if yv>=0 else '#ff5252',fontweight='bold',va='bottom')
ax0.set_title(f'Equity Curve — F#1+F#3  |  Final ${final_eq:,.0f}  |  Max DD {max_dd:.1f}%',
              fontsize=10,fontweight='bold',color='white',pad=7)
ax0.set_facecolor('#0d0d1f'); ax0.spines[:].set_color('#333'); ax0.tick_params(colors='#bbb',labelsize=8)
ax0.set_ylabel('Equity ($)',color='#aaa'); ax0.grid(alpha=0.12)
ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))

ax1=fig.add_subplot(gs[1,:])
def session(h):
    if 0<=h<8: return 'Asian\n(00-08)'
    if 8<=h<13: return 'London Open\n(08-13)'
    if 13<=h<17: return 'NY/London\nOverlap (13-17)'
    if 17<=h<19: return 'NY Session\n(17-19)'
    return 'Other'
tdf['session']=tdf['hour'].apply(session)
sess_order=['Asian\n(00-08)','London Open\n(08-13)','NY/London\nOverlap (13-17)','NY Session\n(17-19)','Other']
sd=tdf.groupby('session').apply(calc_pct).reindex(sess_order).fillna(0).reset_index(); sd.columns=['s','pct']
styled_bar(ax1,sd['s'],sd['pct'].tolist(),'Session Net Profit (%)',fs=9)

ax2=fig.add_subplot(gs[2,0])
day_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dd2=tdf.groupby('dow').apply(calc_pct).reindex(day_order).fillna(0).reset_index(); dd2.columns=['d','pct']
styled_bar(ax2,dd2['d'],dd2['pct'].tolist(),'Day of Week Net %',rot=30)

ax3=fig.add_subplot(gs[2,1])
md=tdf.groupby('month').apply(calc_pct).reset_index(); md.columns=['m','pct']
md['mn']=md['m'].apply(lambda x: calendar.month_abbr[x])
styled_bar(ax3,md['mn'],md['pct'].tolist(),'Monthly Seasonality (%)')

ax4=fig.add_subplot(gs[3,0])
yd=yearly_pct.reset_index(); yd.columns=['y','pct']
styled_bar(ax4,yd['y'].astype(str),yd['pct'].tolist(),'Yearly Net %')

# Risk distribution histogram
ax5=fig.add_subplot(gs[3,1])
r_arr=np.array(risk_used_log)*100
ax5.hist(r_arr,bins=40,color='#4fc3f7',edgecolor='#333',alpha=0.8)
ax5.axvline(avg_risk_used,color='#ff9800',lw=2,ls='--',label=f'Mean={avg_risk_used:.3f}%')
ax5.axvline(1.0,color='#ff5252',lw=1.5,ls=':',label='Fixed 1.0% reference')
ax5.set_title('Risk % Distribution (Inverse Vol Adjusted)',fontsize=9,fontweight='bold',color='white',pad=6)
ax5.set_xlabel('Risk % per Trade',color='#aaa',fontsize=8); ax5.set_ylabel('# Trades',color='#aaa',fontsize=8)
ax5.tick_params(colors='#bbb',labelsize=8); ax5.set_facecolor('#0d0d1f')
ax5.spines[:].set_color('#333'); ax5.grid(alpha=0.15); ax5.legend(fontsize=8,facecolor='#1a1a2e',labelcolor='white')

out='/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/filter3_inv_vol.png'
plt.savefig(out,dpi=150,bbox_inches='tight',facecolor=fig.get_facecolor())
print(f"Chart saved → {out}")
