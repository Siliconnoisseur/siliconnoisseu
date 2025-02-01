#!/usr/bin/env python3
"
multi_indicator_sp100.py

A comprehensive Python script that:

1. Downloads daily data for the S&P 100 tickers using yfinance.
2. Computes five technical indicators (SMA200 with reversed logic, RSI, MACD, Bollinger Bands, and Stochastic with in-zone logic).
3. Generates a daily Buy/Sell signal if >=4/5 indicators agree.
4. Runs a simple long-only backtest (single share).
5. Optionally plots Matplotlib charts for any ticker that triggers a Buy or Sell signal on the current day.

Required libraries:
    pip install yfinance pandas numpy matplotlib

Author: ChatGPT
Date: 2025-01-01
"

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

###############################################################################
# GLOBAL CONFIGURATION
###############################################################################
CONFIG = {
    # Data lookback
    "DATA_LOOKBACK_DAYS": 365,       # How many days of history to pull
    "MIN_BARS_REQUIRED": 200,        # Minimum rows needed (for SMA200, etc.)

    # Indicators
    "SMA200_WINDOW": 200,
    "RSI_PERIOD": 14,
    "RSI_LOWER": 30,
    "RSI_UPPER": 70,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "BB_WINDOW": 20,
    "BB_STD": 2,
    "STOCH_K_PERIOD": 14,
    "STOCH_D_PERIOD": 3,  # We'll use both %K and %D in "in/out of zone" logic
    "STOCH_LOWER": 20,
    "STOCH_UPPER": 80,

    # Signal Combining
    "BUY_SIGNAL_THRESHOLD": 4,  # How many indicators (out of 5) must say Buy
    "SELL_SIGNAL_THRESHOLD": 4, # How many indicators (out of 5) must say Sell

    # Plotting
    "SHOW_PLOTS": True,  # Set to False to disable Matplotlib chart display

    # Backtest Execution Timing
    # Options:
    #   "TRADE_EXECUTION": "next_open"  => trades at the *next day's* Open
    #   "TRADE_EXECUTION": "same_close" => trades at the *same day's* Close
    "TRADE_EXECUTION": "next_open",

    # Logging
    "VERBOSE": True
}

###############################################################################
# FULL S&P 100 TICKER LIST
###############################################################################
sp100_tickers = [
    "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMGN","AMT","AMZN","AVGO","AXP","BA","BAC","BK",
    "BKNG","BLK","BMY","BRK-B","C","CAT","CHTR","CI","CL","CMCSA","COF","COP","COST","CRM",
    "CSCO","CVS","CVX","DHR","DIS","DOW","DUK","EMR","EXC","F","FDX","GD","GE","GILD","GM",
    "GOOG","GOOGL","GS","HD","HON","IBM","INTC","JNJ","JPM","KHC","KO","LIN","LLY","LMT","LOW",
    "MA","MCD","MDLZ","MDT","MET","META","MMM","MO","MRK","MS","MSFT","NEE","NFLX","NVDA","ORCL",
    "PEP","PFE","PG","PM","PYPL","QCOM","RTX","SBUX","SO","SPG","T","TGT","TMO","TMUS","TSLA",
    "TXN","UNH","UNP","UPS","USB","V","VZ","WBA","WELL","WFC","WMT","XOM"
]

###############################################################################
# INDICATOR FUNCTIONS
###############################################################################
def sma(series, window=20):
    return series.rolling(window=window).mean()

def ema(series, window=20):
    return series.ewm(span=window, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(price, slow=26, fast=12, signal=9):
    exp1 = ema(price, fast)
    exp2 = ema(price, slow)
    macd_line = exp1 - exp2
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def bollinger_bands(series, window=20, num_std=2):
    mid_band = sma(series, window)
    std = series.rolling(window).std()
    upper_band = mid_band + (num_std * std)
    lower_band = mid_band - (num_std * std)
    return upper_band, mid_band, lower_band

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k_line = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_line = k_line.rolling(d_period).mean()
    return k_line, d_line
