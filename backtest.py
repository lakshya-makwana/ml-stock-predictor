import pandas as pd
import numpy as np


def backtest_strategy(df, predictions, split_index):

    test_df = df.iloc[split_index:].copy()

    test_df['Prediction'] = predictions

    # Strategy Return
    transaction_cost = 0.001

    test_df['Trade'] = (
        test_df['Prediction'].diff().abs().fillna(0)
    )

    test_df['Strategy_Return'] = (
        test_df['Return'] * test_df['Prediction']
    ) - (
        test_df['Trade'] * transaction_cost
    )

    # Buy and Hold Return
    test_df['Buy_Hold_Return'] = test_df['Return']

    # Cumulative Returns
    test_df['Cumulative_Strategy'] = (
        1 + test_df['Strategy_Return']
    ).cumprod()

    test_df['Cumulative_BuyHold'] = (
        1 + test_df['Buy_Hold_Return']
    ).cumprod()

    sharpe_ratio = (
        test_df['Strategy_Return'].mean() /
        test_df['Strategy_Return'].std()
    ) * np.sqrt(252)

    rolling_max = test_df['Cumulative_Strategy'].cummax()

    drawdown = (
        test_df['Cumulative_Strategy'] - rolling_max
    ) / rolling_max

    max_drawdown = drawdown.min()
    strategy_return = test_df['Cumulative_Strategy'].iloc[-1] - 1
    buy_hold_return = test_df['Cumulative_BuyHold'].iloc[-1] - 1

    number_of_trades = int(test_df['Trade'].sum())
    trading_days = test_df[test_df['Prediction'] == 1]
    winning_days = trading_days[trading_days['Strategy_Return'] > 0]

    if len(trading_days) > 0:
        win_rate = len(winning_days) / len(trading_days)
    else:
        win_rate = 0

    excess_return = strategy_return - buy_hold_return


    print("\nBacktesting Results")
    print("-" * 30)

    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Strategy Return: {strategy_return:.2%}")
    print(f"Buy & Hold Return: {buy_hold_return:.2%}")
    print(f"Number of Trades: {number_of_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Excess Return vs Buy & Hold: {excess_return:.2%}")

    return test_df