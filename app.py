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
    page_icon="QV",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0a0d14;
    --panel: #111827;
    --line: rgba(148, 163, 184, 0.2);
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #22d3ee;
    --accent-2: #f59e0b;
    --good: #10b981;
    --bad: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(34, 211, 238, 0.14), transparent 34rem),
        radial-gradient(circle at 90% 10%, rgba(245, 158, 11, 0.1), transparent 30rem),
        linear-gradient(180deg, #0a0d14 0%, #0f172a 48%, #0a0d14 100%);
    color: var(--text);
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    letter-spacing: 0;
}

div[data-testid="stMarkdownContainer"] p {
    color: #cbd5e1;
}

.hero {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 22px;
    background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(17, 24, 39, 0.72)),
        linear-gradient(90deg, rgba(34, 211, 238, 0.14), rgba(245, 158, 11, 0.1));
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
}

.eyebrow {
    color: var(--accent);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero h1 {
    font-size: clamp(2rem, 5vw, 4.4rem);
    line-height: 1;
    margin: 0 0 12px;
    font-weight: 800;
}

.hero p {
    max-width: 780px;
    margin: 0;
    color: #cbd5e1;
    font-size: 1.02rem;
    line-height: 1.65;
}

.section-label {
    color: #e2e8f0;
    font-size: 1.05rem;
    font-weight: 800;
    margin: 1.35rem 0 0.55rem;
}

.signal-card {
    border: 1px solid var(--line);
    border-left: 5px solid var(--accent);
    border-radius: 12px;
    padding: 18px 20px;
    background: rgba(15, 23, 42, 0.82);
}

.signal-card.buy {
    border-left-color: var(--good);
}

.signal-card.sell {
    border-left-color: var(--bad);
}

.signal-card .label {
    color: var(--muted);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.signal-card .value {
    color: var(--text);
    font-size: 1.8rem;
    font-weight: 800;
    margin-top: 4px;
}

.signal-card .sub {
    color: #cbd5e1;
    margin-top: 6px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.96));
    border-right: 1px solid rgba(148, 163, 184, 0.18);
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f8fafc;
}

section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
    font-weight: 700;
}

div[data-baseweb="input"],
div[data-baseweb="select"] > div,
div[data-testid="stDateInput"] input {
    background: rgba(15, 23, 42, 0.92) !important;
    border-color: rgba(148, 163, 184, 0.25) !important;
    color: #f8fafc !important;
    border-radius: 10px !important;
}

div[data-testid="metric-container"] {
    background: linear-gradient(145deg, rgba(23, 32, 51, 0.96), rgba(13, 18, 30, 0.96));
    border: 1px solid var(--line);
    padding: 18px 18px 16px;
    border-radius: 12px;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
}

div[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-weight: 700;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f8fafc;
    font-size: 1.65rem;
}

div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-weight: 800;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(15, 23, 42, 0.72);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 10px 10px 0 0;
    padding: 12px 18px;
    font-weight: 750;
}

.stTabs [aria-selected="true"] {
    color: #f8fafc !important;
    background: linear-gradient(135deg, rgba(34, 211, 238, 0.18), rgba(245, 158, 11, 0.1)) !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}

button[kind="secondary"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


def apply_chart_theme(fig, title, height=600):
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="#f8fafc")),
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(10, 13, 20, 0)",
        plot_bgcolor="rgba(15, 23, 42, 0.45)",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=28, r=28, t=78, b=36),
        hovermode="x unified"
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.12)",
        zeroline=False,
        rangeslider_visible=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.12)",
        zeroline=False
    )
    return fig


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">Live market intelligence</div>
    <h1>QuantVision AI</h1>
    <p>Track price action, technical momentum, and machine-learning forecasts from a focused financial analytics workspace.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Dashboard Settings")
st.sidebar.caption("Tune the market, history window, and forecast horizon.")

stock = st.sidebar.text_input(
    "Stock Ticker",
    "TSLA"
).upper().strip()

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

    st.markdown('<div class="section-label">Snapshot</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Current Price",
        f"${current_price:.2f}",
        f"{percent_change:.2f}%"
    )

    col2.metric(
        "Volume",
        f"{int(data['Volume'].squeeze().iloc[-1]):,}"
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

        st.markdown(f'<div class="section-label">{stock} Market Overview</div>', unsafe_allow_html=True)

        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=close_prices,
            name='Market Data',
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444"
        )])

        # Moving averages
        fig.add_trace(go.Scatter(
            x=data.index,
            y=close_prices.rolling(20).mean(),
            mode='lines',
            name='MA20',
            line=dict(color="#22d3ee", width=2)
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=close_prices.rolling(50).mean(),
            mode='lines',
            name='MA50',
            line=dict(color="#f59e0b", width=2)
        ))

        apply_chart_theme(fig, f"{stock} Price Action", 680)
        fig.update_yaxes(title_text="Price")

        st.plotly_chart(
            fig,
            width="stretch"
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
    y = df['Close'].squeeze()

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

        st.markdown('<div class="section-label">ML Prediction Dashboard</div>', unsafe_allow_html=True)

        forecast_col, accuracy_col = st.columns([2, 1])
        with accuracy_col:
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
            name='Actual Price',
            line=dict(color="#22d3ee", width=3)
        ))

        fig2.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df['Predicted'],
            mode='lines',
            name='Predicted Price',
            line=dict(color="#f59e0b", width=3)
        ))

        apply_chart_theme(fig2, "Actual vs Predicted Prices", 560)
        fig2.update_yaxes(title_text="Price")

        with forecast_col:
            st.plotly_chart(
                fig2,
                width="stretch"
            )

        # Future predictions
        st.markdown('<div class="section-label">Future Price Prediction</div>', unsafe_allow_html=True)

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

        future_df['Predicted Price'] = future_df['Predicted Price'].astype(float)

        fig3 = go.Figure()

        fig3.add_trace(go.Scatter(
            x=future_df['Date'],
            y=future_df['Predicted Price'],
            mode='lines+markers',
            name='Future Prediction',
            line=dict(color="#10b981", width=3),
            marker=dict(size=9, color="#22d3ee", line=dict(width=2, color="#0f172a"))
        ))

        apply_chart_theme(fig3, f"{prediction_days}-Day Future Forecast", 520)
        fig3.update_yaxes(title_text="Predicted Price")

        table_col, chart_col = st.columns([1, 2])
        with table_col:
            st.dataframe(
                future_df.style.format({'Predicted Price': '${:,.2f}'}),
                width="stretch",
                hide_index=True
            )
        with chart_col:
            st.plotly_chart(
                fig3,
                width="stretch"
            )

        # Buy/Sell Signal
        forecast_change = float(future_predictions[-1] - current_price)
        forecast_pct = (forecast_change / current_price) * 100
        if future_predictions[-1] > current_price:
            st.markdown(
                f"""
                <div class="signal-card buy">
                    <div class="label">Model signal</div>
                    <div class="value">BUY SIGNAL</div>
                    <div class="sub">Forecast is ${forecast_change:,.2f} ({forecast_pct:.2f}%) above the latest close.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="signal-card sell">
                    <div class="label">Model signal</div>
                    <div class="value">SELL SIGNAL</div>
                    <div class="sub">Forecast is ${abs(forecast_change):,.2f} ({abs(forecast_pct):.2f}%) below the latest close.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ===================================================
    # TAB 3 - TECHNICAL ANALYSIS
    # ===================================================

    with tab3:

        st.markdown('<div class="section-label">Technical Indicators</div>', unsafe_allow_html=True)

        technical_df = pd.DataFrame({
            'MA5': df['MA5'].values,
            'MA10': df['MA10'].values,
            'MA20': df['MA20'].values
        }, index=df.index)

        ma_fig = go.Figure()
        ma_colors = {
            'MA5': '#22d3ee',
            'MA10': '#f59e0b',
            'MA20': '#10b981'
        }
        for column, color in ma_colors.items():
            ma_fig.add_trace(go.Scatter(
                x=technical_df.index,
                y=technical_df[column],
                mode='lines',
                name=column,
                line=dict(color=color, width=2.6)
            ))
        apply_chart_theme(ma_fig, "Moving Averages", 520)
        ma_fig.update_yaxes(title_text="Price")
        st.plotly_chart(ma_fig, width="stretch")

        st.markdown('<div class="section-label">Risk Pulse</div>', unsafe_allow_html=True)

        volatility = close_prices.pct_change().std() * 100

        st.metric(
            "Volatility",
            f"{volatility:.2f}%"
        )

except Exception as e:
    st.error(f"Error: {e}")
