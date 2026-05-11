import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start="2019-01-01"):

    print(f"Downloading data for {ticker}...")

    df = yf.download(ticker, start=start)

    if df.empty:
        raise ValueError("No data found. Check ticker symbol.")

    df.dropna(inplace=True)

    return df