import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="QuantVision AI",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

/* Main app */
.stApp {
    background-color: #0b0f19;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #1a1f2e, #121826);
    border: 1px solid #2a3245;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background-color: #1f2937;
    color: white;
    border-radius: 10px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("📈 QuantVision AI")
st.markdown("### ML-Powered Financial Analytics Dashboard")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Dashboard Settings")

stock = st.sidebar.text_input(
    "Stock Ticker",
    "TSLA"
)

prediction_days = st.sidebar.selectbox(
    "Prediction Window",
    [1, 7, 30]
)

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2023-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime("today")
)

# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

try:

    data = load_data(stock, start_date, end_date)

    if data.empty:
        st.error("No stock data found.")
        st.stop()

    # Fix multidimensional issue
    close_prices = data['Close'].squeeze()

    # ---------------------------------------------------
    # KPI CARDS
    # ---------------------------------------------------

    current_price = float(close_prices.iloc[-1])
    previous_price = float(close_prices.iloc[-2])

    price_change = current_price - previous_price
    percent_change = (price_change / previous_price) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Current Price",
        f"${current_price:.2f}",
        f"{percent_change:.2f}%"
    )

    col2.metric(
        "Volume",
        f"{int(data['Volume'].iloc[-1]):,}"
    )

    col3.metric(
        "52W High",
        f"${float(close_prices.max()):.2f}"
    )

    col4.metric(
        "52W Low",
        f"${float(close_prices.min()):.2f}"
    )

    # ---------------------------------------------------
    # TABS
    # ---------------------------------------------------

    tab1, tab2, tab3 = st.tabs([
        "Overview",
        "Predictions",
        "Technical Analysis"
    ])

    # ===================================================
    # TAB 1 - OVERVIEW
    # ===================================================

    with tab1:

        st.subheader(f"{stock} Market Overview")

        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=close_prices,
            name='Market Data'
        )])

        # Moving averages
        fig.add_trace(go.Scatter(
            x=data.index,
            y=close_prices.rolling(20).mean(),
            mode='lines',
            name='MA20'
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=close_prices.rolling(50).mean(),
            mode='lines',
            name='MA50'
        ))

        fig.update_layout(
            title=f"{stock} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=700,

            template="plotly_dark",

            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",

            font=dict(color="white")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Raw data
        with st.expander("View Raw Stock Data"):
            st.dataframe(data.tail())

    # ===================================================
    # FEATURE ENGINEERING
    # ===================================================

    df = data.copy()

    df['Close'] = close_prices

    df['Prev Close'] = close_prices.shift(1)
    df['MA5'] = close_prices.rolling(5).mean()
    df['MA10'] = close_prices.rolling(10).mean()
    df['MA20'] = close_prices.rolling(20).mean()

    df.dropna(inplace=True)

    features = [
        'Prev Close',
        'MA5',
        'MA10',
        'MA20'
    ]

    X = df[features]
    y = df['Close']

    # ---------------------------------------------------
    # TRAIN TEST SPLIT
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # ===================================================
    # TAB 2 - PREDICTIONS
    # ===================================================

    with tab2:

        st.subheader("ML Prediction Dashboard")

        st.metric(
            "Mean Absolute Error",
            f"{mae:.2f}"
        )

        # Prediction dataframe
        pred_df = pd.DataFrame({
            'Actual': y_test.values.ravel(),
            'Predicted': predictions.ravel()
        }, index=y_test.index)

        # Prediction chart
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
            height=600,

            template="plotly_dark",

            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",

            font=dict(color="white")
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # Future predictions
        st.subheader("Future Price Prediction")

        latest_data = df[features].iloc[-1:]

        future_predictions = []

        current_input = latest_data.copy()

        for _ in range(prediction_days):

            pred = model.predict(current_input)[0]

            future_predictions.append(pred)

            current_input['Prev Close'] = pred
            current_input['MA5'] = pred
            current_input['MA10'] = pred
            current_input['MA20'] = pred

        future_dates = pd.date_range(
            start=pd.to_datetime("today") + pd.Timedelta(days=1),
            periods=prediction_days
        )

        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_predictions
        })

        st.dataframe(future_df)

        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            x=future_df['Date'],
            y=future_df['Predicted Price'],
            mode='lines+markers',
            name='Future Prediction'
        ))

        fig3.update_layout(
            title=f"{prediction_days}-Day Future Forecast",
            xaxis_title="Date",
            yaxis_title="Predicted Price",
            height=600,

            template="plotly_dark",

            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",

            font=dict(color="white")
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # Buy/Sell Signal
        if future_predictions[-1] > current_price:
            st.success("BUY SIGNAL")
        else:
            st.error("SELL SIGNAL")

    # ===================================================
    # TAB 3 - TECHNICAL ANALYSIS
    # ===================================================

    with tab3:

        st.subheader("Technical Indicators")

        st.write("### Moving Averages")

        st.line_chart(df[['MA5', 'MA10', 'MA20']])

        st.write("### Price Volatility")

        volatility = close_prices.pct_change().std() * 100

        st.metric(
            "Volatility",
            f"{volatility:.2f}%"
        )

except Exception as e:
    st.error(f"Error: {e}")