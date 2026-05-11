import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Page configuration
st.set_page_config(
    page_title="ML Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 ML Stock Predictor")
st.markdown("Predict stock closing prices using Machine Learning")

# Sidebar
st.sidebar.header("Settings")

stock = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Download stock data
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

try:
    data = load_data(stock, start_date, end_date)

    if data.empty:
        st.error("No data found. Please enter a valid stock ticker.")
        st.stop()

    st.subheader(f"Stock Data for {stock}")
    st.dataframe(data.tail())

    # Stock price chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price'
    ))

    fig.update_layout(
        title=f"{stock} Closing Price",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Feature Engineering
    df = data.copy()

    df['Prev Close'] = df['Close'].shift(1)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    df.dropna(inplace=True)

    features = ['Prev Close', 'MA5', 'MA10', 'MA20']
    X = df[features]
    y = df['Close']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)

    st.subheader("Model Performance")
    st.metric("Mean Absolute Error", f"{mae:.2f}")

    # Prediction graph
    pred_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    }, index=y_test.index)

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=pred_df.index,
        y=pred_df['Actual'],
        mode='lines',
        name='Actual Price'
    ))

    fig2.add_trace(go.Scatter(
        x=pred_df.index,
        y=pred_df['Predicted'],
        mode='lines',
        name='Predicted Price'
    ))

    fig2.update_layout(
        title="Actual vs Predicted Prices",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Future prediction
    st.subheader("Next Day Prediction")

    latest_data = df[features].iloc[-1:]
    future_prediction = model.predict(latest_data)[0]

    st.success(f"Predicted Next Closing Price for {stock}: ${future_prediction:.2f}")

except Exception as e:
    st.error(f"Error: {e}")