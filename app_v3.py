import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime

# Configure layout
st.set_page_config(layout="wide")
st.title("📈 Stable Buy/Sell Signal Finder")

# --- Sidebar Inputs ---
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Stock Symbol (e.g., TCS.NS, AAPL)", "RELIANCE.NS")
start = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
end = st.sidebar.date_input("End Date", datetime.now())

if start >= end:
    st.error("❌ End date must be after start date")
    st.stop()

# --- Fetch Stock Data ---
@st.cache_data(show_spinner="📥 Fetching market data...")
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, threads=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip().lower() for col in data.columns.values]
        else:
            data.columns = data.columns.str.lower()
        return data
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        st.stop()

df = get_stock_data(ticker, start, end)

# --- Debug Info ---
with st.sidebar.expander("🛠 Debug Info"):
    st.write("Columns received:", df.columns.tolist())
    st.write("Data shape:", df.shape)
    st.write("NA counts:", df.isna().sum())

# --- Determine Close Price Column ---
def get_price_column(df):
    candidates = ['close', 'adj close', 'price', 'last']
    for col in candidates:
        if col in df.columns:
            return col
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return numeric_cols[0] if len(numeric_cols) > 0 else None

price_col = get_price_column(df)
if price_col is None:
    st.error("❌ No valid price column found.")
    st.stop()

if price_col != 'close':
    df['close'] = df[price_col]
    st.sidebar.info(f"Using column '{price_col}' as closing prices")

# --- Clean Data ---
def clean_data(df):
    df_clean = df.dropna(subset=['close']).copy()
    if len(df_clean) < 5:
        return None
    for col in df_clean.columns:
        if col != 'close' and df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(method='ffill')
    return df_clean

df_clean = clean_data(df)
if df_clean is None:
    st.error("❌ Insufficient data after cleaning")
    st.stop()

# --- Calculate Indicators ---
def calculate_indicators(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['stability'] = df['close'].rolling(window=5, min_periods=1).std()
    df['recent_min'] = df['close'].rolling(window=5, min_periods=1).min()
    df['recent_max'] = df['close'].rolling(window=5, min_periods=1).max()
    df['volatility'] = df['close'].rolling(window=20).std()
    return df.fillna(method='ffill').dropna()

df_with_indicators = calculate_indicators(df_clean)

# --- Generate Signals ---
def generate_signals(df):
    df = df.copy()
    buy_condition = (
        (df['rsi'] < 35) &
        (df['stability'] < df['stability'].quantile(0.3))
    )
    sell_condition = (
        (df['rsi'] > 70) |
        (df['close'] >= df['recent_max'])
    )
    df['signal'] = np.select(
        [buy_condition, sell_condition],
        ['BUY', 'SELL'],
        default='HOLD'
    )
    df['signal_strength'] = np.where(
        buy_condition,
        (35 - df['rsi']) / 35,
        np.where(
            sell_condition,
            (df['rsi'] - 70) / 30,
            0
        )
    )
    return df

final_df = generate_signals(df_with_indicators)

# --- Table Output ---
st.subheader("📊 Signal Table (Last 60 Days)")
st.dataframe(
    final_df[['close', 'rsi', 'stability', 'signal', 'signal_strength']]
    .sort_index(ascending=False)
    .head(60)
    .style.format({
        'close': '{:.2f}',
        'rsi': '{:.1f}',
        'stability': '{:.4f}',
        'signal_strength': '{:.0%}'
    })
)

# --- Latest Signal Summary ---
latest = final_df.iloc[-1]
signal_color = "🟢" if latest['signal'] == 'BUY' else "🔴" if latest['signal'] == 'SELL' else "⚪"

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    ### Latest Signal
    - **Ticker:** {ticker}
    - **Date:** `{latest.name.strftime('%Y-%m-%d')}`
    - **Signal:** {signal_color} **{latest['signal']}**
    - **Strength:** {latest['signal_strength']:.0%}
    - **Price:** ₹{latest['close']:.2f}
    """)

with col2:
    st.markdown(f"""
    ### Technical Indicators
    - **RSI:** {latest['rsi']:.1f}
    - **Stability:** {latest['stability']:.4f}
    - **Recent Min:** {latest['recent_min']:.2f}
    - **Recent Max:** {latest['recent_max']:.2f}
    """)

# --- 🧠 AI-Driven Target Sell Price Estimation ---
st.subheader("📍 🧠 Smart Target Sell Price Estimation")

purchase_price = st.number_input("🛒 Your Purchase Price (₹)", min_value=0.0, format="%.2f", step=0.5)

if purchase_price > 0:
    try:
        current_price = latest['close']
        volatility = latest['volatility']
        resistance_price = latest['recent_max']
        rsi = latest['rsi']

        predicted_target = min(current_price + 1.5 * volatility, resistance_price)

        distance = predicted_target - current_price
        confidence = max(0, min(1, 1 - (distance / (2.5 * volatility)))) if volatility > 0 else 0

        expected_profit = ((predicted_target - purchase_price) / purchase_price) * 100

        if predicted_target <= purchase_price:
            st.warning("📉 Based on technicals, your buy price is too high for a profitable sell in the short term.")
        else:
            st.success(f"""
            ### 🎯 Recommended Sell Price: ₹{predicted_target:.2f}

            - 📈 Current Price: ₹{current_price:.2f}
            - 🧮 Expected Profit: **{expected_profit:.2f}%**
            - 🔁 Volatility (20-day Std Dev): ₹{volatility:.2f}
            - 📐 Resistance (Recent 5-day High): ₹{resistance_price:.2f}
            - ⚖️ RSI (14-day): {rsi:.1f}

            ### 🎯 Estimated Chance to Hit Target: **{confidence * 100:.0f}%**
            """)
    except Exception as e:
        st.error(f"⚠️ Could not compute smart target price: {e}")
else:
    st.info("ℹ️ Enter your purchase price to get an AI-driven target sell suggestion.")

# --- Plotly Chart ---
try:
    import plotly.graph_objects as go
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=final_df.index,
        y=final_df['close'],
        name='Price',
        line=dict(color='blue', width=2),
        hovertemplate='%{y:.2f}<extra></extra>'
    ))

    buys = final_df[final_df['signal'] == 'BUY']
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index,
            y=buys['close'],
            mode='markers',
            name='Buy',
            marker=dict(
                color='green',
                size=10 + (15 * buys['signal_strength'].abs()),
                symbol='triangle-up',
                line=dict(width=1, color='darkgreen')
            ),
            hovertemplate='BUY: %{y:.2f}<extra></extra>'
        ))

    sells = final_df[final_df['signal'] == 'SELL']
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index,
            y=sells['close'],
            mode='markers',
            name='Sell',
            marker=dict(
                color='red',
                size=10 + (15 * sells['signal_strength'].abs()),
                symbol='triangle-down',
                line=dict(width=1, color='darkred')
            ),
            hovertemplate='SELL: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=f"{ticker} Price and Buy/Sell Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Could not render chart: {e}")
