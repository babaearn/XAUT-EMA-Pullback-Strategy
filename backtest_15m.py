"""
15-MINUTE TIMEFRAME BACKTEST — FULL STACK (F#1 + F#3 + F#8 + F#11 ML)
=======================================================================
Uses native 15m Dukascopy data (not resampled from 5m).
All indicator periods adapted for 15m timeframe:
  - EMA 8/21/50 same (covers similar price memory)
  - RSI 14 same
  - ATR 14 same
  - 1H bias from real H1 data (fewer bars → stronger signal per bar)
  - Daily EMA200 from D1 data
  - Lunar cycle (date-based, same logic)
  - ML: trained fresh on 15m forward-simulated candidates

15m vs 5m key differences:
  - Each bar = 15 minutes (3× less noise)
  - 0.25% tap zone same (price-relative)
  - SL = 1.5 × ATR(14) on 15m bars (ATR will be naturally 1.7× larger than 5m)
  - Session: same Tue-Thu 08-19 UTC
  - 15m bars mean ~44 bars/session (vs ~132 in 5m)
  - Expected trades/week: ~2-3 (fewer but higher quality)
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, seaborn as sns, calendar
from collections import Counter
import ephem, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

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
def bolb(s,p=20,k=2):
    mid=s.rolling(p).mean(); std=s.rolling(p).std()
    return mid+k*std,mid-k*std,mid

INITIAL=10_000.0; RR=2.5; ATR_SL=1.5
BASE_RISK={5:0.008,6:0.012,7:0.015}
VOL_FLOOR=0.4; VOL_CAP=2.0
FULL_MOON_THRESHOLD=85.0
ML_THRESH=0.35
MAX_BARS_FORWARD=100   # 100×15m = 25 hours (slightly more)

# ── Baseline comparison (from 5m test) ────────────────────────────────────────
PREV_5M={'net':109.68,'dd':14.52,'sharpe':0.87,'wr':31.49,'pf':1.17,'trades':1032}

FEATURE_COLS=[
    's1L','s2L','s3L','s4L','s5L','s6','s7L','score_long',
    'ret_1b','ret_4b','ret_8b','ret_16b',
    'ema21_slope','ema50_slope','rsi','rsi_slope','rsi_ma','bb_pos',
    'atr_ratio','atr_pctile','vol_ratio','body_ratio',
    'h1_adx','h1_rsi','h1_adx_slope','h1_rsi_slope',
    'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','lunar_sin',
]

# ── Lunar ─────────────────────────────────────────────────────────────────────
print("Computing lunar phases...")
def get_lunar(d_str):
    m=ephem.Moon(); m.compute(d_str); return m.phase
dates=pd.date_range('2021-01-01','2026-04-01',freq='D')
lunar_df=pd.DataFrame({'date':dates})
lunar_df['lunar_pct']=lunar_df['date'].apply(lambda d: get_lunar(d.strftime('%Y/%m/%d')))
lunar_df['full_moon_avoid']=(lunar_df['lunar_pct']>FULL_MOON_THRESHOLD).shift(1).fillna(False)
lunar_df['lunar_prev']=lunar_df['lunar_pct'].shift(1).fillna(50)
lunar_df.set_index('date',inplace=True)

# ── Load native 15m data ──────────────────────────────────────────────────────
print("Loading native 15m data from Dukascopy...")
df=pd.read_csv("data/xauusd-m15-bid-2021-03-01-2026-03-15.csv")
df['dt']=pd.to_datetime(df['timestamp'],unit='ms')
df['hour']=df['dt'].dt.hour; df['dow']=df['dt'].dt.dayofweek
df['month']=df['dt'].dt.month; df['date']=df['dt'].dt.normalize()
df['year']=df['dt'].dt.year
print(f"  15m bars: {len(df):,}  |  Range: {df['dt'].iloc[0].date()} → {df['dt'].iloc[-1].date()}")

# ── Indicators ────────────────────────────────────────────────────────────────
print("Computing indicators on 15m data...")
df['ema8']=ema_c(df['close'],8); df['ema21']=ema_c(df['close'],21)
df['ema50']=ema_c(df['close'],50)
df['rsi']=rsi_c(df['close'],14); df['atr']=atr_c(df['high'],df['low'],df['close'],14)
ml_v=ema_c(df['close'],12)-ema_c(df['close'],26)
sl_m=ema_c(ml_v,9); df['macd_hist']=ml_v-sl_m
df['vol_ma']=df['volume'].rolling(20).mean()
df['atr_avg50']=df['atr'].rolling(50).mean()
df['atr_ratio']=(df['atr']/df['atr_avg50'].replace(0,np.nan)).fillna(1.0)
df['tap_zone']=df['ema21']*0.25/100  # 0.25% tap zone for 15m (slightly wider)

# 15m momentum features (adapted for 15m periods ≈ 5m bar equivalents ×3)
for n,label in [(1,'1b'),(4,'4b'),(8,'8b'),(16,'16b')]:  # 15m, 1H, 2H, 4H
    df[f'ret_{label}']=(df['close']-df['close'].shift(n))/df['close'].shift(n)*100
df['ema21_slope']=(df['ema21']-df['ema21'].shift(4))/df['ema21'].shift(4)*100
df['ema50_slope']=(df['ema50']-df['ema50'].shift(8))/df['ema50'].shift(8)*100
df['rsi_slope']=df['rsi']-df['rsi'].shift(4); df['rsi_ma']=df['rsi'].rolling(14).mean()
bb_up,bb_dn,_=bolb(df['close'],20,2)
df['bb_pos']=(df['close']-bb_dn)/(bb_up-bb_dn).replace(0,np.nan)
df['vol_ratio']=df['volume']/df['vol_ma'].replace(0,np.nan)
df['atr_pctile']=df['atr'].rolling(252).rank(pct=True)
df['body_ratio']=(df['close']-df['open']).abs()/(df['high']-df['low']).replace(0,np.nan)
df['hour_sin']=np.sin(2*np.pi*df['hour']/24); df['hour_cos']=np.cos(2*np.pi*df['hour']/24)
df['dow_sin']=np.sin(2*np.pi*df['dow']/7);    df['dow_cos']=np.cos(2*np.pi*df['dow']/7)
df['month_sin']=np.sin(2*np.pi*df['month']/12)

df=df.merge(lunar_df[['full_moon_avoid','lunar_prev']],left_on='date',right_index=True,how='left')
df['full_moon_avoid']=df['full_moon_avoid'].fillna(False)
df['lunar_sin']=np.sin(2*np.pi*df['lunar_prev']/100)

df.set_index('dt',inplace=True)

# 1H indicators (from 15m bars resampled to 1H)
h1=df[['open','high','low','close','volume']].resample('1H').agg(
    {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
h1['ema21']=ema_c(h1['close'],21); h1['ema200']=ema_c(h1['close'],200)
h1['rsi']=rsi_c(h1['close'],14)
adx_v,pdi,ndi=adx_c(h1['high'],h1['low'],h1['close'],14)
h1['adx']=adx_v; h1['pdi']=pdi; h1['ndi']=ndi
h1['adx_slope']=h1['adx']-h1['adx'].shift(3); h1['rsi_slope']=h1['rsi']-h1['rsi'].shift(3)
h1_s=h1[['ema21','ema200','rsi','adx','pdi','ndi','adx_slope','rsi_slope']].shift(1)
for col in h1_s.columns: df[f'h1_{col}']=h1_s[col].reindex(df.index,method='ffill')

# Daily EMA200
d1=df[['open','high','low','close','volume']].resample('1D').agg(
    {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
d1['ema200']=ema_c(d1['close'],200)
df['d1_ema200']=d1[['ema200']].shift(1)['ema200'].reindex(df.index,method='ffill')
df.reset_index(inplace=True)

# Signals
d=df
df['s1L']=((d['ema8']>d['ema21'])&(d['ema21']>d['ema50'])).astype(int)
df['s2L']=((d['rsi']>50)&(d['rsi']<70)).astype(int)
df['s3L']=(d['macd_hist']>0).astype(int)
df['s4L']=(d['h1_ema21']>d['h1_ema200']).astype(int)
df['s5L']=((d['h1_adx']>20)&(d['h1_pdi']>d['h1_ndi'])).astype(int)
df['s6']=(d['volume']>d['vol_ma']).astype(int)
df['s7L']=(d['h1_rsi']>50).astype(int)
df['score_long']=df[['s1L','s2L','s3L','s4L','s5L','s6','s7L']].sum(axis=1)
df['s1S']=((d['ema8']<d['ema21'])&(d['ema21']<d['ema50'])).astype(int)
df['s2S']=((d['rsi']>30)&(d['rsi']<50)).astype(int)
df['s3S']=(d['macd_hist']<0).astype(int)
df['s4S']=(d['h1_ema21']<d['h1_ema200']).astype(int)
df['s5S']=((d['h1_adx']>20)&(d['h1_ndi']>d['h1_pdi'])).astype(int)
df['s7S']=(d['h1_rsi']<50).astype(int)
df['score_short']=df[['s1S','s2S','s3S','s4S','s5S','s6','s7S']].sum(axis=1)

df['tap_long'] =(d['close']>d['ema21'])&(d['close']<=d['ema21']+df['tap_zone'])
df['tap_short']=(d['close']<d['ema21'])&(d['close']>=d['ema21']-df['tap_zone'])
df['valid_session']=(d['dow'].isin([1,2,3]))&(d['hour']>=8)&(d['hour']<19)&(d['month']!=6)
df['macro_long']=d['close']>d['d1_ema200']
df['macro_short']=d['close']<d['d1_ema200']
df['lunar_ok']=~d['full_moon_avoid']
df['cand_long'] =(df['tap_long'] &(df['score_long']>=5) &df['valid_session']&df['macro_long'] &df['lunar_ok'])
df['cand_short']=(df['tap_short']&(df['score_short']>=5)&df['valid_session']&df['macro_short']&df['lunar_ok'])

n_cand=df['cand_long'].sum()+df['cand_short'].sum()
print(f"  Total candidate bars (score>=5 + all gates): {n_cand:,}")

# ── Forward simulation labeling ───────────────────────────────────────────────
print("Labeling 15m candidates via forward simulation...")
c_arr=df['close'].values; h_arr=df['high'].values; lo_arr=df['low'].values; at_arr=df['atr'].values
cL=df['cand_long'].values; cS=df['cand_short'].values
sL=df['score_long'].values; sS=df['score_short'].values

labeled_rows=[]
for i in range(300, len(df)-MAX_BARS_FORWARD):
    is_long=cL[i]; is_short=cS[i]
    if not (is_long or is_short): continue
    c=c_arr[i]; at=at_arr[i]
    if np.isnan(at) or at<=0: at=c*0.003
    direction='long' if is_long else 'short'
    score=int(sL[i]) if is_long else int(sS[i])
    if direction=='long': sl_p=c-ATR_SL*at; tp_p=c+(c-sl_p)*RR
    else:                 sl_p=c+ATR_SL*at; tp_p=c-(sl_p-c)*RR
    label=None
    for j in range(i+1, min(i+MAX_BARS_FORWARD, len(df))):
        hj=h_arr[j]; lj=lo_arr[j]
        if direction=='long':
            if lj<=sl_p: label=0; break
            if hj>=tp_p: label=1; break
        else:
            if hj>=sl_p: label=0; break
            if lj<=tp_p: label=1; break
    if label is None: continue
    row={'label':label,'direction':direction,'score':score,'entry_idx':i,
         'ts':df['timestamp'].iloc[i],'year':df['year'].iloc[i]}
    for f in FEATURE_COLS:
        v=df[f].iloc[i] if f in df.columns else 0.0
        if pd.isna(v): v=0.0
        if direction=='short' and f in ['s1L','s2L','s3L','s4L','s5L','s7L','score_long',
                                          'ret_1b','ret_4b','ret_8b','ret_16b',
                                          'ema21_slope','ema50_slope','rsi_slope','bb_pos']:
            map_={'s1L':'s1S','s2L':'s2S','s3L':'s3S','s4L':'s4S','s5L':'s5S','s7L':'s7S'}
            v=df[map_[f]].iloc[i] if f in map_ else (df['score_short'].iloc[i] if f=='score_long' else -v)
        row[f]=v
    labeled_rows.append(row)

ML_df=pd.DataFrame(labeled_rows)
ML_df['entry_dt']=pd.to_datetime(ML_df['ts'],unit='ms')
print(f"  Labeled: {len(ML_df):,}  wins={ML_df['label'].sum():,}  losses={(ML_df['label']==0).sum():,}  WR={ML_df['label'].mean()*100:.1f}%")

# ── Walk-forward ML ───────────────────────────────────────────────────────────
print("\nWalk-forward ML validation on 15m data...")
folds=[
    ('2021','2022', pd.Timestamp('2021-03-01'), pd.Timestamp('2022-01-01'), pd.Timestamp('2023-01-01')),
    ('2021-22','2023', pd.Timestamp('2021-03-01'), pd.Timestamp('2023-01-01'), pd.Timestamp('2024-01-01')),
    ('2021-23','2024', pd.Timestamp('2021-03-01'), pd.Timestamp('2024-01-01'), pd.Timestamp('2025-01-01')),
    ('2021-24','2025', pd.Timestamp('2021-03-01'), pd.Timestamp('2025-01-01'), pd.Timestamp('2026-01-01')),
    ('2021-25','2026', pd.Timestamp('2021-03-01'), pd.Timestamp('2026-01-01'), pd.Timestamp('2027-01-01')),
]
ML_df['ml_prob']=np.nan; auc_scores=[]

for train_label,test_label,train_start,test_start,test_end in folds:
    train_mask=(ML_df['entry_dt']>=train_start)&(ML_df['entry_dt']<test_start)
    test_mask =(ML_df['entry_dt']>=test_start) &(ML_df['entry_dt']<test_end)
    if train_mask.sum()<50 or test_mask.sum()<5: continue
    X_tr=ML_df[train_mask][FEATURE_COLS].astype(float).values
    y_tr=ML_df[train_mask]['label'].values
    X_te=ML_df[test_mask][FEATURE_COLS].astype(float).values
    y_te=ML_df[test_mask]['label'].values
    sc=StandardScaler(); X_tr=sc.fit_transform(X_tr); X_te=sc.transform(X_te)
    model=RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=15,
        max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    probs=model.predict_proba(X_te)[:,1]
    ML_df.loc[test_mask,'ml_prob']=probs
    if len(np.unique(y_te))>1:
        auc=roc_auc_score(y_te,probs); auc_scores.append(auc)
        n35=(probs>=0.35).sum(); wr35=y_te[probs>=0.35].mean()*100 if n35>0 else 0
        print(f"  Fold {test_label}: AUC={auc:.4f}  P>=0.35: n={n35}, WR={wr35:.1f}%")

avg_auc=np.mean(auc_scores) if auc_scores else 0
oos=ML_df[ML_df['ml_prob'].notna()]
print(f"\n  Avg AUC: {avg_auc:.4f}  |  OOS candidates: {len(oos):,}")

# ── Backtest simulation ───────────────────────────────────────────────────────
print(f"\nRunning 15m backtest (ML gate P>={ML_THRESH})...")
idx_to_prob_long={}; idx_to_prob_short={}
for _,row in oos[oos['ml_prob']>=ML_THRESH].iterrows():
    idx=int(row['entry_idx']); p=row['ml_prob']
    if row['direction']=='long': idx_to_prob_long[idx]=p
    else: idx_to_prob_short[idx]=p

equity=INITIAL; pos=None; sl=tp=0.0
risk_pct=0.008; entry_score=5; entry_t=None
trades=[]; score_log=[]

c_a=df['close'].values; h_a=df['high'].values; lo_a=df['low'].values
at_a=df['atr'].values; avg50_a=df['atr_avg50'].values; ts_a=df['timestamp'].values
sL_a=df['score_long'].values; sS_a=df['score_short'].values

for i in range(300,len(df)):
    if equity<=100: break
    c=c_a[i]; h=h_a[i]; lo=lo_a[i]
    at=at_a[i] if (not np.isnan(at_a[i]) and at_a[i]>0) else c*0.003
    risk_amt=INITIAL*risk_pct

    if pos=="long":
        if lo<=sl:
            equity-=risk_amt; trades.append({'entry_time':entry_t,'exit_time':ts_a[i],
                'status':'loss','pnl':-risk_amt,'score':entry_score,'type':'long'}); pos=None
        elif h>=tp:
            equity+=risk_amt*RR; trades.append({'entry_time':entry_t,'exit_time':ts_a[i],
                'status':'win','pnl':risk_amt*RR,'score':entry_score,'type':'long'}); pos=None
    elif pos=="short":
        if h>=sl:
            equity-=risk_amt; trades.append({'entry_time':entry_t,'exit_time':ts_a[i],
                'status':'loss','pnl':-risk_amt,'score':entry_score,'type':'short'}); pos=None
        elif lo<=tp:
            equity+=risk_amt*RR; trades.append({'entry_time':entry_t,'exit_time':ts_a[i],
                'status':'win','pnl':risk_amt*RR,'score':entry_score,'type':'short'}); pos=None

    if pos is None:
        if i in idx_to_prob_long: direction="long"; sc_entry=int(sL_a[i])
        elif i in idx_to_prob_short: direction="short"; sc_entry=int(sS_a[i])
        else: continue
        if sc_entry>=5:
            base_r=BASE_RISK.get(sc_entry,0.008)
            atr_avg=avg50_a[i] if (not np.isnan(avg50_a[i]) and avg50_a[i]>0) else at
            vol_ratio=np.clip(atr_avg/at if at>0 else 1.0,VOL_FLOOR,VOL_CAP)
            risk_pct=base_r*vol_ratio; entry_score=sc_entry; entry_t=ts_a[i]
            if direction=="long": pos="long"; sl=c-ATR_SL*at; tp=c+(c-sl)*RR
            else: pos="short"; sl=c+ATR_SL*at; tp=c-(sl-c)*RR
            score_log.append(sc_entry)

# ── Stats ─────────────────────────────────────────────────────────────────────
tdf=pd.DataFrame(trades)
tdf['entry_time']=pd.to_datetime(tdf['entry_time'],unit='ms')
tdf['exit_time']=pd.to_datetime(tdf['exit_time'],unit='ms')
tdf['year']=tdf['exit_time'].dt.year; tdf['month']=tdf['exit_time'].dt.month
tdf['dow']=tdf['exit_time'].dt.day_name(); tdf['hour']=tdf['exit_time'].dt.hour
tdf['week']=tdf['exit_time'].dt.isocalendar().week.astype(int)
tdf.to_csv('xauusd_15m_trades.csv',index=False)

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
  15-MINUTE TIMEFRAME BACKTEST — F#1 + F#3 + F#8 + ML Walk-Forward
  Native 15m Dukascopy data  |  ML Avg AUC: {avg_auc:.4f}
╠════════════════════════╦═════════════╦═══════════════════════════╣
  Metric                │  5m Stack   │  15m Stack
╠════════════════════════╬═════════════╬═══════════════════════════╣
  Total Trades          │  1,032      │  {len(tdf):,}  {chk(len(tdf),1032,False)}
  Win Rate              │  31.49%     │  {wr:.2f}%  {chk(wr,31.49)}
  Profit Factor         │  1.17x      │  {pf:.2f}x  {chk(pf,1.17)}
  Net PnL               │  +104.61%   │  {net/INITIAL*100:+.2f}%  {chk(net/INITIAL*100,104.61)}
  Final Equity          │  $20,461    │  ${final_eq:,.0f}
  Max Drawdown          │  14.52%     │  {max_dd:.2f}%  {chk(max_dd,14.52,False)}
  Sharpe Ratio          │  0.87       │  {sharpe:.2f}  {chk(sharpe,0.87)}
  Avg Trades/Week       │  7.6        │  {wk_counts.mean():.1f}
╠════════════════════════╩═════════════╩═══════════════════════════╣
  Yearly Breakdown (OOS from 2022):""")

prev_yr={'2022':8.73,'2023':12.40,'2024':58.36,'2025':28.07,'2026':-2.94}
for yr,v in yearly_pct.items():
    bv=prev_yr.get(str(yr),0)
    oos_flag='[OOS]' if yr>=2022 else '[train]'
    print(f"    {yr} {oos_flag}: {v:+.2f}%  (5m: {bv:+.2f}%)  {'✅' if v>bv else '❌'}")

print(f"""╠══════════════════════════════════════════════════════════════════╣
  Score dist: {dict(sorted(Counter(score_log).items()))}
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
fig=plt.figure(figsize=(22,20),facecolor='#0a0a1a')
fig.suptitle(
    f"15-MINUTE BACKTEST — Institutional ML Walk-Forward  [F#1+F#3+F#8+ML]\n"
    f"Net: {net/INITIAL*100:+.2f}%  ·  MaxDD: {max_dd:.1f}%  ·  Sharpe: {sharpe:.2f}  ·  "
    f"WR: {wr:.1f}%  ·  PF: {pf:.2f}×  ·  Trades: {len(tdf):,}  ·  Avg AUC: {avg_auc:.4f}\n"
    f"vs 5m Stack → Net: +104.61%  ·  MaxDD: 14.52%  ·  Sharpe: 0.87",
    fontsize=12,fontweight='bold',color='white',y=0.998)

gs=gridspec.GridSpec(4,2,figure=fig,hspace=0.50,wspace=0.32)

ax0=fig.add_subplot(gs[0,:])
eq_times=tdf.sort_values('exit_time')['exit_time'].values
ax0.plot(eq_times,eq_s.values,color='#ffd740',lw=1.5,label='15m ML Strategy')
ax0.fill_between(eq_times,INITIAL,eq_s.values,where=eq_s.values>=INITIAL,alpha=0.15,color='#ffd740')
ax0.fill_between(eq_times,INITIAL,eq_s.values,where=eq_s.values< INITIAL,alpha=0.15,color='#ff4444')
ax0.axhline(INITIAL,color='white',lw=0.8,ls='--',alpha=0.4)
for yr in tdf['year'].unique():
    if tdf[tdf['year']==yr].empty: continue
    first=tdf[tdf['year']==yr]['exit_time'].iloc[0]; yv=yearly_pct.get(yr,0)
    ax0.axvline(first,color='#ffffff18',lw=0.7,ls=':')
    ax0.text(first,INITIAL*1.01,f"{yr}\n{yv:+.0f}%",fontsize=8,
             color='#ffd740' if yv>=0 else '#ff4444',fontweight='bold',va='bottom')
ax0.set_title(f'15m Equity  |  Final ${final_eq:,.0f}  |  Max DD {max_dd:.1f}%  |  Sharpe {sharpe:.2f}',
              fontsize=10,fontweight='bold',color='white',pad=7)
ax0.set_facecolor('#0a0a1f');ax0.spines[:].set_color('#333');ax0.tick_params(colors='#bbb',labelsize=8)
ax0.set_ylabel('Equity ($)',color='#aaa'); ax0.grid(alpha=0.10)
ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax0.legend(fontsize=8,facecolor='#0a0a1a',labelcolor='white')

ax1=fig.add_subplot(gs[1,:])
def session(h):
    if 0<=h<8: return 'Asian (00-08)'
    if 8<=h<13: return 'London Open\n(08-13)'
    if 13<=h<17: return 'NY/London\nOverlap (13-17)'
    if 17<=h<19: return 'NY Session\n(17-19)'; return 'Other'
tdf['session']=tdf['hour'].apply(session)
sess_order=['Asian (00-08)','London Open\n(08-13)','NY/London\nOverlap (13-17)','NY Session\n(17-19)','Other']
sd=tdf.groupby('session').apply(calc_pct).reindex(sess_order).fillna(0).reset_index(); sd.columns=['s','pct']
styled_bar(ax1,sd['s'],sd['pct'].tolist(),'15m Session Net Profit (%)',fs=9)

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
styled_bar(ax4,yd['y'].astype(str),yd['pct'].tolist(),'Yearly Net % (OOS from 2022)')

ax5=fig.add_subplot(gs[3,1])
wk_m=tdf.groupby(['year','week']).apply(calc_pct).unstack(level=1).fillna(0)
wk_m.index=wk_m.index.astype(str)
vmax=max(abs(wk_m.max().max()),abs(wk_m.min().min()),0.01)
sns.heatmap(wk_m,cmap='RdYlGn',center=0,vmin=-vmax,vmax=vmax,ax=ax5,
            linewidths=0.2,cbar_kws={'label':'Net %','shrink':0.8})
ax5.set_title(f'Weekly Heatmap (Avg {wk_counts.mean():.1f}/wk)',
              fontsize=9,fontweight='bold',color='white',pad=6)
ax5.set_facecolor('#0d0d1f'); ax5.tick_params(colors='#bbb',labelsize=7)

out='/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/backtest_15m.png'
plt.savefig(out,dpi=150,bbox_inches='tight',facecolor=fig.get_facecolor())
print(f"\nChart saved → {out}")
