"""
FILTER TEST #6 — DXY DIRECTION (US DOLLAR INDEX)
=================================================
Builds on: Filter #1 (Daily EMA200) + Filter #3 (Inv Vol Sizing)
Active baseline: Trades=1,742 · WR=30.08% · PF=1.08x · Net=+86.81% · MaxDD=28.41% · Sharpe=0.56

Filter #6 Rule:
  Long gold only when DXY daily close < DXY EMA21 (dollar WEAKENING)
  Short gold only when DXY daily close > DXY EMA21 (dollar STRENGTHENING)
  DXY EMA21 shifted by 1 bar → no lookahead

Scientific basis:
  Gold vs USD has -0.85 average correlation over 20 years.
  Every institutional gold desk uses DXY direction as primary macro overlay.
  When USD is strengthening, gold faces fundamental headwind regardless of technicals.
  When USD is weakening, gold has tailwind — EMA pullbacks are more likely to hold.

Data: yfinance DX-Y.NYB (free, real-time delayed)
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, seaborn as sns, calendar
from collections import Counter
import yfinance as yf

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

PREV={'trades':1742,'wr':30.08,'pf':1.08,'net_pct':86.81,'max_dd':28.41,'sharpe':0.56}
INITIAL=10_000.0; RR=2.5; ATR_SL=1.5
BASE_RISK={5:0.008,6:0.012,7:0.015}
VOL_FLOOR=0.4; VOL_CAP=2.0

# ── Download DXY Data ─────────────────────────────────────────────────────────
print("Downloading DXY daily data from Yahoo Finance...")
dxy_raw = yf.download("DX-Y.NYB", start="2020-01-01", end="2026-03-20",
                      progress=False, auto_adjust=True)
if dxy_raw.empty:
    # Fallback ticker
    dxy_raw = yf.download("UUP", start="2020-01-01", end="2026-03-20",
                          progress=False, auto_adjust=True)
    print("  (Using UUP as DXY proxy)")

dxy = dxy_raw[['Close']].copy()
dxy.columns = ['dxy_close']
dxy.index = pd.to_datetime(dxy.index).tz_localize(None)
dxy['dxy_ema21'] = ema_c(dxy['dxy_close'], 21)
# Shift by 1 day — use YESTERDAY's DXY signal to avoid lookahead
dxy_shifted = dxy[['dxy_close','dxy_ema21']].shift(1)
dxy_shifted.columns = ['dxy_prev_close','dxy_prev_ema21']
print(f"  DXY data: {dxy.index[0].date()} → {dxy.index[-1].date()}  ({len(dxy)} days)")
print(f"  DXY range: {dxy['dxy_close'].min():.2f} – {dxy['dxy_close'].max():.2f}")

# ── Load 5m Gold Data ─────────────────────────────────────────────────────────
print("Loading 5m gold data & computing indicators...")
df = pd.read_csv("data/xauusd-m5-bid-2021-03-01-2026-03-15.csv")
df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
df['hour']  = df['dt'].dt.hour
df['dow']   = df['dt'].dt.dayofweek
df['month'] = df['dt'].dt.month
df['date']  = df['dt'].dt.normalize()   # daily date for DXY merge

df['ema8']  = ema_c(df['close'],8); df['ema21'] = ema_c(df['close'],21)
df['ema50'] = ema_c(df['close'],50); df['rsi']  = rsi_c(df['close'],14)
df['atr']   = atr_c(df['high'],df['low'],df['close'],14)
ml,sl_m     = macd_c(df['close']); df['macd_hist'] = ml-sl_m
df['vol_ma']   = df['volume'].rolling(20).mean()
df['atr_avg50']= df['atr'].rolling(50).mean()

# Merge DXY onto 5m by date
df = df.merge(dxy_shifted, left_on='date', right_index=True, how='left')
df['dxy_prev_close'] = df['dxy_prev_close'].ffill()
df['dxy_prev_ema21'] = df['dxy_prev_ema21'].ffill()

# DXY direction: True = dollar weak = gold bullish
df['dxy_weak']   = df['dxy_prev_close'] < df['dxy_prev_ema21']   # for longs
df['dxy_strong'] = df['dxy_prev_close'] > df['dxy_prev_ema21']   # for shorts

print(f"  DXY weak (dollar below EMA21): {df['dxy_weak'].sum():,} / {len(df):,} 5m bars "
      f"({df['dxy_weak'].mean()*100:.1f}%)")
print(f"  DXY strong (dollar above EMA21): {df['dxy_strong'].sum():,} 5m bars "
      f"({df['dxy_strong'].mean()*100:.1f}%)")

df.set_index('dt', inplace=True)

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

# Daily EMA200 (F#1)
d1=df[['open','high','low','close','volume']].resample('1D').agg(
    {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
d1['ema200']=ema_c(d1['close'],200)
df['d1_ema200']=d1[['ema200']].shift(1)['ema200'].reindex(df.index,method='ffill')
df.reset_index(inplace=True)

# Scoring
d=df
s1L=(d['ema8']>d['ema21'])&(d['ema21']>d['ema50'])
s2L=(d['rsi']>50)&(d['rsi']<70); s3L=d['macd_hist']>0
s4L=d['h1_ema21']>d['h1_ema200']
s5L=(d['h1_adx']>20)&(d['h1_pdi']>d['h1_ndi'])
s6=d['volume']>d['vol_ma']; s7L=d['h1_rsi']>50
s1S=(d['ema8']<d['ema21'])&(d['ema21']<d['ema50'])
s2S=(d['rsi']>30)&(d['rsi']<50); s3S=d['macd_hist']<0
s4S=d['h1_ema21']<d['h1_ema200']
s5S=(d['h1_adx']>20)&(d['h1_ndi']>d['h1_pdi']); s7S=d['h1_rsi']<50

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

# ✅ Filter #6: DXY Direction Gate
df['entry_long'] =(df['in_long_tap']  &(df['score_long']>=5)  &df['valid_session']&
                   df['macro_long']   &df['dxy_weak'])
df['entry_short']=(df['in_short_tap'] &(df['score_short']>=5) &df['valid_session']&
                   df['macro_short']  &df['dxy_strong'])

# ── Simulation (F#1 + F#3 + F#6) ────────────────────────────────────────────
print("Running backtest F#1 + F#3 + F#6...")
equity=INITIAL; pos=None; sl=tp=0.0
risk_pct=0.008; entry_score=5; entry_t=None
trades=[]; score_log=[]; risk_log=[]

eL=df['entry_long'].values; eS=df['entry_short'].values
c_a=df['close'].values; h_a=df['high'].values
lo_a=df['low'].values; at_a=df['atr'].values
avg50_a=df['atr_avg50'].values; ts_a=df['timestamp'].values
sL_a=df['score_long'].values; sS_a=df['score_short'].values

for i in range(300,len(df)):
    if equity<=100: break
    c=c_a[i]; h=h_a[i]; lo=lo_a[i]
    at=at_a[i] if (not np.isnan(at_a[i]) and at_a[i]>0) else c*0.003
    risk_amt=INITIAL*risk_pct

    if pos=="long":
        if lo<=sl:
            equity-=risk_amt
            trades.append({'entry_time':entry_t,'exit_time':ts_a[i],'status':'loss',
                           'pnl':-risk_amt,'score':entry_score,'type':'long'})
            pos=None
        elif h>=tp:
            equity+=risk_amt*RR
            trades.append({'entry_time':entry_t,'exit_time':ts_a[i],'status':'win',
                           'pnl':risk_amt*RR,'score':entry_score,'type':'long'})
            pos=None
    elif pos=="short":
        if h>=sl:
            equity-=risk_amt
            trades.append({'entry_time':entry_t,'exit_time':ts_a[i],'status':'loss',
                           'pnl':-risk_amt,'score':entry_score,'type':'short'})
            pos=None
        elif lo<=tp:
            equity+=risk_amt*RR
            trades.append({'entry_time':entry_t,'exit_time':ts_a[i],'status':'win',
                           'pnl':risk_amt*RR,'score':entry_score,'type':'short'})
            pos=None

    if pos is None:
        sc_entry=None
        if eL[i]: sc_entry=int(sL_a[i]); direction="long"
        elif eS[i]: sc_entry=int(sS_a[i]); direction="short"
        if sc_entry and sc_entry>=5:
            base_r=BASE_RISK.get(sc_entry,0.008)
            atr_avg=avg50_a[i] if (not np.isnan(avg50_a[i]) and avg50_a[i]>0) else at
            vol_ratio=np.clip(atr_avg/at if atr_avg>0 else 1.0,VOL_FLOOR,VOL_CAP)
            risk_pct=base_r*vol_ratio; entry_score=sc_entry; entry_t=ts_a[i]
            if direction=="long": pos="long"; sl=c-ATR_SL*at; tp=c+(c-sl)*RR
            else: pos="short"; sl=c+ATR_SL*at; tp=c-(sl-c)*RR
            score_log.append(sc_entry); risk_log.append(risk_pct)

# ── Stats ─────────────────────────────────────────────────────────────────────
tdf=pd.DataFrame(trades)
tdf['entry_time']=pd.to_datetime(tdf['entry_time'],unit='ms')
tdf['exit_time']=pd.to_datetime(tdf['exit_time'],unit='ms')
tdf['year']=tdf['exit_time'].dt.year; tdf['month']=tdf['exit_time'].dt.month
tdf['dow']=tdf['exit_time'].dt.day_name(); tdf['hour']=tdf['exit_time'].dt.hour
tdf['week']=tdf['exit_time'].dt.isocalendar().week.astype(int)
tdf.to_csv('xauusd_filter6_trades.csv',index=False)

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

def chk(new,old,hb=True):
    d=new-old; icon=('✅' if d>0 else '❌') if hb else ('✅' if d<0 else '❌')
    return f"{icon} {d:+.2f}"

print(f"""
╔══════════════════════════════════════════════════════════════════╗
  FILTER #6: DXY DIRECTION GATE
  Long: DXY < EMA21 (dollar weak) | Short: DXY > EMA21 (dollar strong)
╠══════════════════════╦══════════════╦═══════════════════════════╣
  Metric              │ F#1+F#3 Base │ F#1+F#3+F#6 Result
╠══════════════════════╬══════════════╬═══════════════════════════╣
  Total Trades        │  1,742       │  {len(tdf):,}  {chk(len(tdf),1742,False)} ({1742-len(tdf)} blocked)
  Win Rate            │  30.08%      │  {wr:.2f}%  {chk(wr,30.08)}
  Profit Factor       │  1.08x       │  {pf:.2f}x  {chk(pf,1.08)}
  Net PnL             │  +86.81%     │  {net/INITIAL*100:+.2f}%  {chk(net/INITIAL*100,86.81)}
  Final Equity        │  $18,681     │  ${final_eq:,.0f}
  Max Drawdown        │  28.41%      │  {max_dd:.2f}%  {chk(max_dd,28.41,False)}
  Sharpe Ratio        │  0.56        │  {sharpe:.2f}  {chk(sharpe,0.56)}
  Avg Trades/Week     │  8.3         │  {wk_counts.mean():.1f}
╠══════════════════════╩══════════════╩═══════════════════════════╣
  Yearly Breakdown:""")

prev_yr={'2021':-5.67,'2022':19.93,'2023':-0.08,'2024':53.37,'2025':26.66,'2026':-7.40}
for yr,v in yearly_pct.items():
    bv=prev_yr.get(str(yr),0)
    print(f"    {yr}: {v:+.2f}%  (was {bv:+.2f}%)  {'✅' if v>bv else '❌'}")

print(f"""╠══════════════════════════════════════════════════════════════════╣
  Score: {dict(sorted(Counter(score_log).items()))}
╚══════════════════════════════════════════════════════════════════╝""")

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
    f"Filter #6: DXY Direction Gate (Gold vs USD Correlation)  [F#1 + F#3 + F#6]\n"
    f"Net: {net/INITIAL*100:+.2f}%  ·  MaxDD: {max_dd:.1f}%  ·  Sharpe: {sharpe:.2f}  ·  "
    f"WR: {wr:.1f}%  ·  PF: {pf:.2f}×  ·  Trades: {len(tdf):,}\n"
    f"vs F#1+F#3 → Net: +86.81%  ·  MaxDD: 28.41%  ·  Sharpe: 0.56",
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
ax0.set_title(f'Equity Curve — F#1+F#3+F#6 (DXY)  |  Final ${final_eq:,.0f}  |  Max DD {max_dd:.1f}%',
              fontsize=10,fontweight='bold',color='white',pad=7)
ax0.set_facecolor('#0d0d1f');ax0.spines[:].set_color('#333');ax0.tick_params(colors='#bbb',labelsize=8)
ax0.set_ylabel('Equity ($)',color='#aaa'); ax0.grid(alpha=0.12)
ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))

ax1=fig.add_subplot(gs[1,:])
def session(h):
    if 0<=h<8: return 'Asian\n(00-08)'
    if 8<=h<13: return 'London Open\n(08-13)'
    if 13<=h<17: return 'NY/London\nOverlap (13-17)'
    if 17<=h<19: return 'NY Session\n(17-19)'; return 'Other'
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

ax5=fig.add_subplot(gs[3,1])
wk_m=tdf.groupby(['year','week']).apply(calc_pct).unstack(level=1).fillna(0)
wk_m.index=wk_m.index.astype(str)
vmax=max(abs(wk_m.max().max()),abs(wk_m.min().min()),0.01)
sns.heatmap(wk_m,cmap='RdYlGn',center=0,vmin=-vmax,vmax=vmax,
            ax=ax5,linewidths=0.2,cbar_kws={'label':'Net %','shrink':0.8})
ax5.set_title(f'Weekly PnL Heatmap (Avg {wk_counts.mean():.1f}/wk)',
              fontsize=9,fontweight='bold',color='white',pad=6)
ax5.set_facecolor('#0d0d1f'); ax5.tick_params(colors='#bbb',labelsize=7)

out='/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/filter6_dxy.png'
plt.savefig(out,dpi=150,bbox_inches='tight',facecolor=fig.get_facecolor())
print(f"\nChart saved → {out}")
