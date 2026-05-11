import pandas as pd
import numpy as np


def calculate_rsi(data, window=14):

    close = data['Close'].squeeze()

    delta = close.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_features(df):

    df['Return'] = df['Close'].pct_change()

    df['Lag_1'] = df['Return'].shift(1)
    df['Lag_2'] = df['Return'].shift(2)
    df['Lag_5'] = df['Return'].shift(5)

    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    df['Volatility'] = df['Return'].rolling(window=10).std()

    df['RSI'] = calculate_rsi(df)

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = ema_12 - ema_26

    df['Volume_Change'] = df['Volume'].pct_change()

    df['Target'] = (
        df['Close'].shift(-1) > df['Close']
    ).astype(int)

    df.dropna(inplace=True)

    return df