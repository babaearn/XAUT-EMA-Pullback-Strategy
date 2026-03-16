import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timedelta
from typing import Optional

def fetch_klines(symbol: str, interval: str, limit: int, end_ms: Optional[int] = None):
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
    if end_ms: params["end"] = end_ms
    resp = requests.get(url, params=params, timeout=15).json()
    if resp.get("retCode") == 0: return resp.get("result", {}).get("list", [])
    return []

def fetch_historical_custom(symbol: str, interval: str, days: int):
    end_ms = int(datetime.now().timestamp() * 1000)
    start_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_candles = []
    current_end = end_ms
    while current_end > start_ms:
        candles = fetch_klines(symbol, interval, 1000, current_end)
        if not candles: break
        all_candles.extend(candles)
        current_end = int(candles[-1][0]) - 1
        if int(candles[-1][0]) <= start_ms: break
        time.sleep(0.2)
    df = pd.DataFrame(all_candles, columns=["open_time", "open", "high", "low", "close", "volume", "turnover"])
    df = df.sort_values("open_time").reset_index(drop=True)
    for col in ["open_time", "open", "high", "low", "close", "volume"]: df[col] = df[col].astype(float)
    return df

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, sign=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, sign)
    return macd_line, signal_line, macd_line - signal_line

def run_quant_backtest():
    symbol = "PAXGUSDT"
    days = 1825
    print(f"Fetching {days} days of {symbol} 5m data from Bybit...")
    df = fetch_historical_custom(symbol, interval="5", days=days)
    print(f"Loaded {len(df)} candles. Computing indicators...")
    
    # Needs to extract hour and day of week to filter *entry*
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Calculate indicators vectorized
    df['ema'] = ema(df['close'], 21)
    df['rsi'] = rsi(df['close'], 14)
    macd_line, signal_line, _ = macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    
    print("Simulating trades with applied session and weekend filters...")
    
    # Strategy params
    tap_threshold_pct = 0.2
    stop_loss_buffer_pct = 0.5
    take_profit_rr = 2.0
    risk_per_trade_pct = 1.0
    use_rsi_filter = True
    first_tap_only = True
    rsi_long_min = 50.0
    rsi_short_max = 50.0
    
    initial_equity = 10000.0
    equity = initial_equity
    
    position = None
    sl = 0.0
    tp = 0.0
    entry_equity = initial_equity
    
    trades = []
    equity_curve = []
    
    in_trend = False
    
    closes = df['close'].values
    emas = df['ema'].values
    rsis = df['rsi'].values
    macds = df['macd'].values
    macd_sigs = df['macd_signal'].values
    highs = df['high'].values
    lows = df['low'].values
    times = df['open_time'].values
    hours = df['datetime'].dt.hour.values
    days_of_week = df['datetime'].dt.dayofweek.values # Monday=0, Sunday=6
    
    for i in range(50, len(df)):
        equity_curve.append({'time': times[i], 'equity': equity})
        
        close = closes[i]
        high = highs[i]
        low = lows[i]
        ema_val = emas[i]
        hour = hours[i]
        day_of_week = days_of_week[i]
        
        # Position management
        if position == "long":
            risk_amount = entry_equity * (risk_per_trade_pct / 100)
            if low <= sl:
                equity -= risk_amount
                trades.append({'entry_time': entry_time, 'exit_time': times[i], 'type': 'long', 'status': 'loss', 'pnl': -risk_amount})
                position = None
            elif high >= tp:
                equity += risk_amount * take_profit_rr
                trades.append({'entry_time': entry_time, 'exit_time': times[i], 'type': 'long', 'status': 'win', 'pnl': risk_amount * take_profit_rr})
                position = None
                
        elif position == "short":
            risk_amount = entry_equity * (risk_per_trade_pct / 100)
            if high >= sl:
                equity -= risk_amount
                trades.append({'entry_time': entry_time, 'exit_time': times[i], 'type': 'short', 'status': 'loss', 'pnl': -risk_amount})
                position = None
            elif low <= tp:
                equity += risk_amount * take_profit_rr
                trades.append({'entry_time': entry_time, 'exit_time': times[i], 'type': 'short', 'status': 'win', 'pnl': risk_amount * take_profit_rr})
                position = None
                
        # Signal evaluation (Only allowed Monday to Thursday, 13:00 - 16:59 UTC)
        # Friday (4), Saturday (5), Sunday (6) are rejected
        if position is None:
            prev_close = closes[i-1]
            prev_ema = emas[i-1]
            
            trend_flip = (prev_close <= prev_ema and close > ema_val) or (prev_close >= prev_ema and close < ema_val)
            if trend_flip:
                in_trend = True
                
            is_valid_time = (day_of_week < 4) and (13 <= hour < 17)
            
            if is_valid_time:
                tap_zone = ema_val * (tap_threshold_pct / 100)
                long_tap = (close > ema_val and close <= ema_val + tap_zone) and (not first_tap_only or in_trend)
                rsi_ok_long = rsis[i] > rsi_long_min
                
                short_tap = (close < ema_val and close >= ema_val - tap_zone) and (not first_tap_only or in_trend)
                rsi_ok_short = rsis[i] < rsi_short_max
                
                if long_tap and rsi_ok_long:
                    position = "long"
                    sl = ema_val * (1 - stop_loss_buffer_pct / 100)
                    risk = abs(close - sl)
                    tp = close + risk * take_profit_rr
                    entry_equity = equity
                    entry_time = times[i]
                    in_trend = False
                elif short_tap and rsi_ok_short:
                    position = "short"
                    sl = ema_val * (1 + stop_loss_buffer_pct / 100)
                    risk = abs(sl - close)
                    tp = close - risk * take_profit_rr
                    entry_equity = equity
                    entry_time = times[i]
                    in_trend = False
                
    print(f"Finished simulating {len(trades)} trades.")
    
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'], unit='ms')
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'], unit='ms')
        trades_df.to_csv('paxg_filtered_trades.csv', index=False)
        
        eq_df = pd.DataFrame(equity_curve)
        eq_df['time'] = pd.to_datetime(eq_df['time'], unit='ms')
        eq_df['cummax'] = eq_df['equity'].cummax()
        eq_df['drawdown'] = (eq_df['cummax'] - eq_df['equity']) / eq_df['cummax'] * 100
        max_drawdown = eq_df['drawdown'].max()
        
        wins = trades_df[trades_df['status'] == 'win']
        losses = trades_df[trades_df['status'] == 'loss']
        
        print(f"\n===== RESULTS (FILTERED) =====")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Wins: {len(wins)}, Losses: {len(losses)}")
        print(f"Win Rate: {len(wins)/len(trades_df)*100:.2f}%")
        print(f"Gross Profit: {wins['pnl'].sum():.2f}")
        print(f"Gross Loss: {losses['pnl'].sum():.2f}")
        total_pnl = trades_df['pnl'].sum()
        print(f"Total PnL: {total_pnl:.2f} ({(total_pnl/initial_equity)*100:.2f}%)")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Final Equity: {equity:.2f}")
        print("============================\n")
    else:
        print("No trades generated")

if __name__ == "__main__":
    run_quant_backtest()
