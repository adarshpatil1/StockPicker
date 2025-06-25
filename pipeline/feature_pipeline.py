import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from pathlib import Path

def download_data(ticker, start="2018-01-01", end=None):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

def add_technical_indicators(df):
    close = df[['Close']].squeeze()  # ✅ ensures 1D

    df['rsi'] = RSIIndicator(close).rsi()

    macd = MACD(close)
    df['macd'] = macd.macd_diff()

    bb = BollingerBands(close)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = bb.bollinger_wband()

    return df

def label_data(df, horizon=1):
    df['return'] = df['Close'].pct_change(horizon).shift(-horizon)
    df['target'] = (df['return'] > 0).astype(int)
    return df

def preprocess_ticker(ticker):
    df = download_data(ticker)
    df = add_technical_indicators(df)
    df = label_data(df)
    df.dropna(inplace=True)
    return df

def run_pipeline(tickers, output_path="data/processed"):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        df = preprocess_ticker(ticker)
        df.to_csv(f"{output_path}/{ticker}.csv")
        print(f"Saved: {ticker} - {len(df)} rows")

if __name__ == "__main__":
    run_pipeline(["RELIANCE.NS", "INFY.NS", "TCS.NS", 
                  "ICICIBANK.NS", "SBIN.NS", "HDFCBANK.NS", 
                  "TATMOTORS.NS", "HINDUNILVR.NS"
                  ])
