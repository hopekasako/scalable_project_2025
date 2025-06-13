import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
from real_time import EnhancedStockForecastingPipeline


st.set_page_config(page_title="Netflix Stock Forecast", layout="wide")


@st.cache_resource
def get_pipeline():
    return EnhancedStockForecastingPipeline()


pipeline = get_pipeline()
st.title("Netflix Stock Price Forecast")


# Create placeholders
price_cards = st.empty()
price_chart = st.empty()
indicators = st.empty()


# Auto-refresh mechanism
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()


def fetch_and_process_data():
    df = pipeline.fetch_real_time_data()
    if df is not None:
        df = pipeline.calculate_technical_indicators(df)
        return pipeline.make_predictions(df)
    return None


def update_dashboard():
    predictions_df = fetch_and_process_data()
   
    if predictions_df is not None:
        latest = predictions_df.iloc[-1]
       
        # Price cards
        col1, col2, col3 = price_cards.columns(3)
       
        # Current Price
        price_change = ((latest['Close'] - predictions_df.iloc[-2]['Close']) / predictions_df.iloc[-2]['Close'] * 100) if len(predictions_df) > 1 else 0
        col1.metric(
            "Current Price",
            f"${latest['Close']:.2f}",
            f"{price_change:.2f}%"
        )
       
        # Predicted Price
        prediction_diff = ((latest['Predicted_Price'] - latest['Close']) / latest['Close'] * 100)
        col2.metric(
            "Predicted Price",
            f"${latest['Predicted_Price']:.2f}",
            f"{prediction_diff:.2f}%"
        )
       
        # RSI
        col3.metric("RSI", f"{latest['RSI']:.2f}")
       
        # Generate unique keys based on timestamp
        timestamp = int(time.time() * 1000)
       
        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['Close'],
            name='Actual Price',
            line=dict(color='#2563eb')
        ))
        fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['Predicted_Price'],
            name='Predicted Price',
            line=dict(color='#dc2626')
        ))
        fig.update_layout(
            title='Price vs Prediction',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            height=500
        )
        price_chart.plotly_chart(fig, use_container_width=True, key=f"price_chart_{timestamp}")
       
        # Technical indicators
        col1, col2 = indicators.columns(2)
       
        # EMA Chart
        ema_fig = go.Figure()
        ema_fig.add_trace(go.Scatter(x=predictions_df['Datetime'], y=predictions_df['EMA_5'], name='EMA 5'))
        ema_fig.add_trace(go.Scatter(x=predictions_df['Datetime'], y=predictions_df['EMA_20'], name='EMA 20'))
        ema_fig.update_layout(title='Moving Averages', height=400)
        col1.plotly_chart(ema_fig, use_container_width=True, key=f"ema_chart_{timestamp}")
       
        # Volume Chart
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(x=predictions_df['Datetime'], y=predictions_df['Volume'], name='Volume'))
        volume_fig.update_layout(title='Trading Volume', height=400)
        col2.plotly_chart(volume_fig, use_container_width=True, key=f"volume_chart_{timestamp}")


# Auto-refresh logic
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    update_dashboard()


# Manual refresh button
if st.button('Refresh Data'):
    update_dashboard()


# Initial load
update_dashboard()


# Add auto-refresh using JavaScript
st.markdown(
    """
    <script>
        function reload() {
            window.location.reload();
        }
        setTimeout(reload, 60000);
    </script>
    """,
    unsafe_allow_html=True
)