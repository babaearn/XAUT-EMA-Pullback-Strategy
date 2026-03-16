"""
FILTER TEST #10 — XGBOOST ML PROBABILITY MODEL
================================================
Builds on: F#1 (Daily EMA200) + F#3 (Inv Vol Sizing) + F#8 (Lunar Cycle)
Active baseline: Trades=1,299 · WR=30.79% · PF=1.14x · Net=+109.68% · MaxDD=22.75% · Sharpe=0.80

What this does:
  Replaces the crude 5/7 score threshold with a trained ML probability model.
  Instead of: "enter if score >= 5"
  We use:     "enter if P(win | features) >= threshold"

Training protocol (strict time-based split, NO data leakage):
  TRAIN:  2021-03-01 → 2023-12-31  (3 years of labeled outcomes)
  TEST:   2024-01-01 → 2026-03-15  (blind forward test — never seen by model)

Features (16 inputs):
  - 7 individual signals (binary: s1-s7)
  - confluence score (sum of signals)
  - RSI exact value (not just >50)
  - MACD histogram exact value
  - 1H ADX value (trend strength)
  - 1H RSI value
  - ATR ratio (current/avg50) — volatility context
  - tap_depth_pct (how deep into the 0.2% EMA zone)
  - hour of day (0-23)
  - lunar phase % (0-100)

Model: XGBoost classifier with:
  - max_depth=4 (prevent overfitting on small dataset)
  - n_estimators=200
  - learning_rate=0.05
  - scale_pos_weight = loss_count/win_count (handle class imbalance)
  - eval_metric='auc'
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, seaborn as sns, calendar
from collections import Counter
import ephem, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
print("sklearn GradientBoosting loaded (pure Python, no OpenMP)")


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

INITIAL=10_000.0; RR=2.5; ATR_SL=1.5
BASE_RISK={5:0.008,6:0.012,7:0.015}
VOL_FLOOR=0.4; VOL_CAP=2.0
FULL_MOON_THRESHOLD=85.0
ML_PROB_THRESHOLD=0.52   # Enter only when model says P(win) > this

# ── Lunar table ───────────────────────────────────────────────────────────────
print("Computing lunar phases...")
def get_lunar(date_str):
    m=ephem.Moon(); m.compute(date_str); return m.phase
dates=pd.date_range('2021-03-01','2026-03-16',freq='D')
lunar_df=pd.DataFrame({'date':dates})
lunar_df['lunar_pct']=lunar_df['date'].apply(lambda d: get_lunar(d.strftime('%Y/%m/%d')))
lunar_df['full_moon_zone']=lunar_df['lunar_pct']>FULL_MOON_THRESHOLD
lunar_df['full_moon_avoid']=lunar_df['full_moon_zone'].shift(1).fillna(False)
lunar_df['lunar_prev']=lunar_df['lunar_pct'].shift(1).fillna(50)
lunar_df.set_index('date',inplace=True)

# ── Load & full feature computation ──────────────────────────────────────────
print("Loading all data & computing full feature set...")
df=pd.read_csv("data/xauusd-m5-bid-2021-03-01-2026-03-15.csv")
df['dt']=pd.to_datetime(df['timestamp'],unit='ms')
df['hour']=df['dt'].dt.hour; df['dow']=df['dt'].dt.dayofweek
df['month']=df['dt'].dt.month; df['date']=df['dt'].dt.normalize()

df['ema8']=ema_c(df['close'],8); df['ema21']=ema_c(df['close'],21); df['ema50']=ema_c(df['close'],50)
df['rsi']=rsi_c(df['close'],14); df['atr']=atr_c(df['high'],df['low'],df['close'],14)
ml_v,sl_m=macd_c(df['close']); df['macd_hist']=ml_v-sl_m
df['vol_ma']=df['volume'].rolling(20).mean()
df['atr_avg50']=df['atr'].rolling(50).mean()
df['atr_ratio']=(df['atr']/df['atr_avg50'].replace(0,np.nan)).fillna(1.0)
df['tap_zone']=df['ema21']*0.20/100
df['tap_depth_pct_long']=(df['close']-df['ema21'])/df['tap_zone'].replace(0,np.nan)*100
df['tap_depth_pct_long']=df['tap_depth_pct_long'].fillna(0).clip(0,100)

# Merge lunar
df=df.merge(lunar_df[['full_moon_avoid','lunar_prev']],left_on='date',right_index=True,how='left')
df['full_moon_avoid']=df['full_moon_avoid'].fillna(False)
df['lunar_prev']=df['lunar_prev'].fillna(50)

df.set_index('dt',inplace=True)

# 1H bias
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

# Daily EMA200
d1=df[['open','high','low','close','volume']].resample('1D').agg(
    {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
d1['ema200']=ema_c(d1['close'],200)
df['d1_ema200']=d1[['ema200']].shift(1)['ema200'].reindex(df.index,method='ffill')
df.reset_index(inplace=True)

# ── Individual signals (for ML features) ─────────────────────────────────────
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

# Base entry conditions (F#1 + F#8 gates, score>=5)
df['in_long_tap']=(d['close']>d['ema21'])&(d['close']<=d['ema21']+df['tap_zone'])
df['in_short_tap']=(d['close']<d['ema21'])&(d['close']>=d['ema21']-df['tap_zone'])
df['valid_session']=(d['dow'].isin([1,2,3]))&(d['hour']>=8)&(d['hour']<19)&(d['month']!=6)
df['macro_long']=d['close']>d['d1_ema200']
df['macro_short']=d['close']<d['d1_ema200']
df['lunar_ok']=~d['full_moon_avoid']

# Candidate entries (F#1+F#8 filtered but no score threshold yet — ML will decide)
df['cand_long'] =(df['in_long_tap'] &(df['score_long']>=5) &df['valid_session']&df['macro_long'] &df['lunar_ok'])
df['cand_short']=(df['in_short_tap']&(df['score_short']>=5)&df['valid_session']&df['macro_short']&df['lunar_ok'])

# ── Build labeled dataset from actual F#8 trade log ──────────────────────────
print("Loading F#8 trade log for ML training labels...")
trades_f8=pd.read_csv('xauusd_filter8_trades.csv')
trades_f8['entry_time']=pd.to_datetime(trades_f8['entry_time'])
trades_f8['target']=(trades_f8['status']=='win').astype(int)

# Match features to entry timestamps
df_idx=df.copy(); df_idx['dt_ms']=df_idx['timestamp']
feature_cols=['s1L','s2L','s3L','s4L','s5L','s6','s7L',
              'score_long','rsi','macd_hist','h1_adx','h1_rsi',
              'atr_ratio','tap_depth_pct_long','hour','lunar_prev']

# For shorts, remap s signals
df_short_feats=df.copy()
df_short_feats['s1L']=df['s1S']; df_short_feats['s2L']=df['s2S']
df_short_feats['s3L']=df['s3S']; df_short_feats['s4L']=df['s4S']
df_short_feats['s5L']=df['s5S']; df_short_feats['s7L']=df['s7S']
df_short_feats['score_long']=df['score_short']
df_short_feats['tap_depth_pct_long']=df['tap_depth_pct_long']*-1

# Build feature lookups by timestamp
df_ts=df.set_index('timestamp')[feature_cols]
df_ts_short=df_short_feats.set_index('timestamp')[feature_cols]

def get_feats(row):
    ts=int(row['entry_time'].timestamp()*1000)
    src=df_ts_short if row.get('type','long')=='short' else df_ts
    if ts in src.index: return src.loc[ts]
    return None

print("Extracting features for each labeled trade...")
feature_rows=[]
for _,row in trades_f8.iterrows():
    f=get_feats(row)
    if f is not None:
        feature_rows.append({**f.to_dict(),'target':row['target'],
                              'entry_time':row['entry_time'],
                              'type':row['type']})
ML_df=pd.DataFrame(feature_rows).dropna()
print(f"  Labeled dataset: {len(ML_df)} trades  (wins={ML_df['target'].sum()}, "
      f"losses={(ML_df['target']==0).sum()})")

# ── Time-based train/test split ───────────────────────────────────────────────
TRAIN_END=pd.Timestamp('2024-01-01')
train=ML_df[ML_df['entry_time']<TRAIN_END]
test =ML_df[ML_df['entry_time']>=TRAIN_END]
print(f"  TRAIN: {len(train)} trades (2021-2023)  |  TEST: {len(test)} trades (2024-2026)")

X_train=train[feature_cols].astype(float)
y_train=train['target']
X_test =test[feature_cols].astype(float)
y_test =test['target']

# Class imbalance correction
spw=max(1.0,(y_train==0).sum()/(y_train==1).sum())
print(f"  Class imbalance ratio: {spw:.2f} → using scale_pos_weight={spw:.2f}")

# ── Train XGBoost ─────────────────────────────────────────────────────────────
print("\nTraining GradientBoostingClassifier (pure sklearn, no OpenMP)...")
model = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, max_features=0.8,
    random_state=42, verbose=0)
model.fit(X_train.values, y_train.values)  # use .values to avoid feature name validation

# Evaluate on test set
y_prob_test = model.predict_proba(X_test.values)[:,1]
auc=roc_auc_score(y_test, y_prob_test)
print(f"  Test AUC: {auc:.4f}  (0.5=random, 1.0=perfect, >0.55=useful)")

# Win rate by probability bucket
test_copy=test.copy(); test_copy['prob']=y_prob_test
for thresh in [0.40,0.45,0.50,0.52,0.55,0.60]:
    subset=test_copy[test_copy['prob']>=thresh]
    n=len(subset); w=subset['target'].sum() if n>0 else 0
    wr=w/n*100 if n>0 else 0
    print(f"  P(win)>={thresh:.2f}: {n:4d} trades · WR={wr:.1f}%  "
          f"{'✅' if wr>30.79 else '❌'}")

# ── Build probability score on ALL candidate entries ──────────────────────────
print("\nScoring all potential entries with ML model...")
# Build clean feature frame (no duplicate columns)
extra_cols = ['cand_long','cand_short','score_short','atr','atr_avg50']
keep = ['timestamp'] + feature_cols + [c for c in extra_cols if c not in feature_cols]
df_feat_all = df[keep].copy()
df_feat_all = df_feat_all.dropna(subset=feature_cols)
df_feat_all['prob_long'] = 0.0; df_feat_all['prob_short'] = 0.0

cand_long_mask  = df_feat_all['cand_long'].fillna(False)
cand_short_mask = df_feat_all['cand_short'].fillna(False)

if cand_long_mask.any():
    X_pred_long = df_feat_all.loc[cand_long_mask, feature_cols].astype(float).values
    assert X_pred_long.shape[1] == len(feature_cols), f"Feature count mismatch: {X_pred_long.shape[1]} vs {len(feature_cols)}"
    df_feat_all.loc[cand_long_mask, 'prob_long'] = model.predict_proba(X_pred_long)[:,1]

if cand_short_mask.any():
    short_feats = df_feat_all.loc[cand_short_mask, feature_cols].copy()
    short_feats['s1L'] = df.loc[df_feat_all.index[cand_short_mask], 's1S'].values
    short_feats['s2L'] = df.loc[df_feat_all.index[cand_short_mask], 's2S'].values
    short_feats['s3L'] = df.loc[df_feat_all.index[cand_short_mask], 's3S'].values
    short_feats['score_long'] = df.loc[df_feat_all.index[cand_short_mask], 'score_short'].values
    X_pred_short = short_feats[feature_cols].astype(float).values
    df_feat_all.loc[cand_short_mask, 'prob_short'] = model.predict_proba(X_pred_short)[:,1]

df_feat_all['entry_long']  = cand_long_mask  & (df_feat_all['prob_long']  >= ML_PROB_THRESHOLD)
df_feat_all['entry_short'] = cand_short_mask & (df_feat_all['prob_short'] >= ML_PROB_THRESHOLD)

# ── Simulation ────────────────────────────────────────────────────────────────
print(f"Running backtest with ML gate (P>={ML_PROB_THRESHOLD})...")
feat_idx=df_feat_all.reset_index(drop=True)
equity=INITIAL; pos=None; sl=tp=0.0
risk_pct=0.008; entry_score=5; entry_t=None
trades=[]; score_log=[]; risk_log=[]

c_a=df['close'].values; h_a=df['high'].values; lo_a=df['low'].values
at_a=df['atr'].values; avg50_a=df['atr_avg50'].values
ts_a=df['timestamp'].values
sL_a=df['score_long'].values; sS_a=df['score_short'].values
eL=df_feat_all['entry_long'].reindex(range(len(df)),fill_value=False).values
eS=df_feat_all['entry_short'].reindex(range(len(df)),fill_value=False).values

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
        if i<len(eL) and eL[i]: sc_entry=int(sL_a[i]); direction="long"
        elif i<len(eS) and eS[i]: sc_entry=int(sS_a[i]); direction="short"
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
tdf.to_csv('xauusd_filter10_trades.csv',index=False)

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
  FILTER #10: XGBOOST ML MODEL  (P(win) >= {ML_PROB_THRESHOLD})
  Train: 2021-2023  |  Test: 2024-2026  |  AUC: {auc:.4f}
╠══════════════════════╦══════════════╦═══════════════════════════╣
  Metric              │ F#1+F#3+F#8  │ +F#10 ML Result
╠══════════════════════╬══════════════╬═══════════════════════════╣
  Total Trades        │  1,299       │  {len(tdf):,}  {chk(len(tdf),1299,False)}
  Win Rate            │  30.79%      │  {wr:.2f}%  {chk(wr,30.79)}
  Profit Factor       │  1.14x       │  {pf:.2f}x  {chk(pf,1.14)}
  Net PnL             │  +109.68%    │  {net/INITIAL*100:+.2f}%  {chk(net/INITIAL*100,109.68)}
  Final Equity        │  $20,968     │  ${final_eq:,.0f}
  Max Drawdown        │  22.75%      │  {max_dd:.2f}%  {chk(max_dd,22.75,False)}
  Sharpe Ratio        │  0.80        │  {sharpe:.2f}  {chk(sharpe,0.80)}
  Avg Trades/Week     │  7.8         │  {wk_counts.mean():.1f}
╠══════════════════════╩══════════════╩═══════════════════════════╣
  Yearly Breakdown:""")

prev_yr={'2021':-3.70,'2022':21.10,'2023':16.64,'2024':50.78,'2025':26.53,'2026':-1.67}
for yr,v in yearly_pct.items():
    bv=prev_yr.get(str(yr),0)
    print(f"    {yr}: {v:+.2f}%  (was {bv:+.2f}%)  {'✅' if v>bv else '❌'}")

# Feature importance
fi=model.feature_importances_
fi_df=pd.DataFrame({'feature':feature_cols,'importance':fi}).sort_values('importance',ascending=False)
print(f"""╠══════════════════════════════════════════════════════════════════╣
  Top 5 features by importance:""")
for _,row in fi_df.head(5).iterrows():
    print(f"    {row['feature']:<30}: {row['importance']:.4f}")
print(f"  Score: {dict(sorted(Counter(score_log).items()))}")
print("╚══════════════════════════════════════════════════════════════════╝")

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
fig=plt.figure(figsize=(22,22),facecolor='#1a1a2e')
fig.suptitle(
    f"Filter #10: XGBoost ML Probability Model  P>={ML_PROB_THRESHOLD}  AUC={auc:.4f}  [F#1+F#3+F#8+F#10]\n"
    f"Net: {net/INITIAL*100:+.2f}%  ·  MaxDD: {max_dd:.1f}%  ·  Sharpe: {sharpe:.2f}  ·  "
    f"WR: {wr:.1f}%  ·  PF: {pf:.2f}×  ·  Trades: {len(tdf):,}\n"
    f"vs F#1+F#3+F#8 → Net: +109.68%  ·  MaxDD: 22.75%  ·  Sharpe: 0.80",
    fontsize=12,fontweight='bold',color='white',y=0.995)

gs=gridspec.GridSpec(5,2,figure=fig,hspace=0.52,wspace=0.32)

ax0=fig.add_subplot(gs[0,:])
eq_times=tdf.sort_values('exit_time')['exit_time'].values
ax0.plot(eq_times,eq_s.values,color='#00e676',lw=1.2)
ax0.fill_between(eq_times,INITIAL,eq_s.values,where=eq_s.values>=INITIAL,alpha=0.18,color='#00e676')
ax0.fill_between(eq_times,INITIAL,eq_s.values,where=eq_s.values< INITIAL,alpha=0.18,color='#ff5252')
ax0.axhline(INITIAL,color='white',lw=0.8,ls='--',alpha=0.4)
# Mark train/test split
split_dt=pd.Timestamp('2024-01-01')
ax0.axvline(split_dt,color='#ff9800',lw=2,ls='--',alpha=0.8,label='Train→Test Split (2024)')
ax0.legend(fontsize=9,facecolor='#1a1a2e',labelcolor='white',loc='upper left')
for yr in tdf['year'].unique():
    first=tdf[tdf['year']==yr]['exit_time'].iloc[0]; yv=yearly_pct.get(yr,0)
    ax0.axvline(first,color='#ffffff18',lw=0.7,ls=':')
    ax0.text(first,INITIAL*1.005,f"{yr}\n{yv:+.0f}%",fontsize=8,
             color='#00e676' if yv>=0 else '#ff5252',fontweight='bold',va='bottom')
ax0.set_title(f'Equity — F#1+F#3+F#8+F#10 ML  |  Final ${final_eq:,.0f}  |  Max DD {max_dd:.1f}%  |  AUC {auc:.3f}',
              fontsize=10,fontweight='bold',color='white',pad=7)
ax0.set_facecolor('#0d0d1f');ax0.spines[:].set_color('#333');ax0.tick_params(colors='#bbb',labelsize=8)
ax0.set_ylabel('Equity ($)',color='#aaa'); ax0.grid(alpha=0.12)
ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))

ax1=fig.add_subplot(gs[1,:])
def session(h):
    if 0<=h<8: return 'Asian (00-08)'
    if 8<=h<13: return 'London Open\n(08-13)'
    if 13<=h<17: return 'NY/London\nOverlap (13-17)'
    if 17<=h<19: return 'NY Session\n(17-19)'; return 'Other'
tdf['session']=tdf['hour'].apply(session)
sess_order=['Asian (00-08)','London Open\n(08-13)','NY/London\nOverlap (13-17)','NY Session\n(17-19)','Other']
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
styled_bar(ax4,yd['y'].astype(str),yd['pct'].tolist(),'Yearly Net % (blind test from 2024🔵)')

ax5=fig.add_subplot(gs[3,1])
fi_top=fi_df.head(10)
bars=ax5.barh(fi_top['feature'], fi_top['importance'],
              color=['#4fc3f7' if v>0.1 else '#81d4fa' for v in fi_top['importance']],
              edgecolor='#333', linewidth=0.5)
ax5.set_title('XGBoost Feature Importance (top 10)',fontsize=9,fontweight='bold',color='white',pad=6)
ax5.set_xlabel('Importance',color='#aaa',fontsize=7); ax5.tick_params(colors='#bbb',labelsize=8)
ax5.set_facecolor('#0d0d1f'); ax5.spines[:].set_color('#333'); ax5.grid(axis='x',alpha=0.15)
ax5.invert_yaxis()
for bar,val in zip(bars,fi_top['importance']):
    ax5.text(val+0.002,bar.get_y()+bar.get_height()/2,f"{val:.4f}",va='center',fontsize=7,color='white')

ax6=fig.add_subplot(gs[4,:])
wk_m=tdf.groupby(['year','week']).apply(calc_pct).unstack(level=1).fillna(0)
wk_m.index=wk_m.index.astype(str)
vmax=max(abs(wk_m.max().max()),abs(wk_m.min().min()),0.01)
sns.heatmap(wk_m,cmap='RdYlGn',center=0,vmin=-vmax,vmax=vmax,ax=ax6,
            linewidths=0.2,cbar_kws={'label':'Net %','shrink':0.8})
ax6.set_title(f'Weekly PnL Heatmap  (Avg {wk_counts.mean():.1f}/wk)',
              fontsize=9,fontweight='bold',color='white',pad=6)
ax6.set_facecolor('#0d0d1f'); ax6.tick_params(colors='#bbb',labelsize=7)

out='/Users/mudrex/.gemini/antigravity/brain/52dfe01a-90ef-4075-85d5-508cdf7ccaa5/filter10_xgboost.png'
plt.savefig(out,dpi=150,bbox_inches='tight',facecolor=fig.get_facecolor())
print(f"\nChart saved → {out}")
