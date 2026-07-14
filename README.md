# QuantVision AI

QuantVision AI is a machine-learning stock analysis dashboard that downloads historical market data, creates technical indicators, trains prediction models, and visualizes stock forecasts.

## Features

- Download historical stock data using Yahoo Finance
- Display candlestick charts and moving averages
- Generate machine-learning price predictions
- Show actual vs predicted prices
- Calculate basic volatility
- Provide bullish or bearish model signals
- Run a separate command-line ML pipeline for direction prediction and backtesting

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- yfinance
- Plotly
- Matplotlib
- Seaborn
- XGBoost

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt

```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

Run the command-line ML pipeline:

```bash
python main.py
```

## Project Structure

```text
app.py                  Streamlit dashboard
main.py                 Command-line ML pipeline
data_loader.py          Downloads stock data
feature_engineering.py  Creates technical indicators and target labels
model.py                Trains and evaluates ML models
backtest.py             Backtests model-based trading strategy
visualization.py        Creates charts and plots
```

## Disclaimer

This project is for educational purposes only and is not financial advice.
