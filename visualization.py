import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_stock_price(df):

    plt.figure(figsize=(12, 6))

    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['MA_10'], label='MA 10')
    plt.plot(df['MA_20'], label='MA 20')
    plt.plot(df['MA_50'], label='MA 50')

    plt.title('Stock Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.legend()
    plt.grid(True)

    plt.show()


def plot_feature_importance(model, feature_columns):

    if not hasattr(model, 'feature_importances_'):
        return

    importance = model.feature_importances_

    plt.figure(figsize=(10, 6))

    plt.barh(feature_columns, importance)

    plt.title('Feature Importance')

    plt.show()


def plot_equity_curve(test_df):

    plt.figure(figsize=(12, 6))

    plt.plot(test_df['Cumulative_Strategy'], label='Strategy')
    plt.plot(test_df['Cumulative_BuyHold'], label='Buy & Hold')

    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')

    plt.legend()
    plt.grid(True)

    plt.show()


def monte_carlo_simulation(df, simulations=100, days=100):

    returns = df['Return'].dropna()

    mean_return = returns.mean()
    volatility = returns.std()

    last_price = df['Close'].iloc[-1]

    plt.figure(figsize=(12, 6))

    for _ in range(simulations):

        prices = [last_price]

        for _ in range(days):
            next_price = prices[-1] * (
                1 + np.random.normal(mean_return, volatility)
            )

            prices.append(next_price)

        plt.plot(prices)

    plt.title('Monte Carlo Simulation')
    plt.xlabel('Days')
    plt.ylabel('Simulated Price')

    plt.grid(True)

    plt.show()


def plot_correlation_heatmap(df):

    plt.figure(figsize=(12, 8))

    sns.heatmap(df.corr(), cmap='coolwarm')

    plt.title('Correlation Heatmap')

    plt.show()