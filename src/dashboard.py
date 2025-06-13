import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from real_time import EnhancedStockForecastingPipeline
import numpy as np
import pytz
import uuid


# Set page config with dark theme and collapsed sidebar
st.set_page_config(
    page_title="Netflix Stock Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Enhanced Custom CSS with Netflix theme and bigger text
st.markdown("""
    <style>
    /* Main container styling with Netflix theme */
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #000000 0%, #141414 100%);
        color: #FFFFFF;
        min-height: 100vh;
    }
    
    /* Netflix-inspired metric box styling */
    .stMetric {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
        width: 100% !important;
        min-width: 0 !important;
        min-height: 180px !important;
        height: 180px !important;
        padding: 2rem 1.5rem !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 32px rgba(229, 9, 20, 0.15) !important;
        border: 2px solid rgba(229, 9, 20, 0.3) !important;
        color: #FFFFFF !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
        margin: 0 !important;
        transition: all 0.4s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stMetric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #E50914 0%, #B81D24 100%);
    }
    
    .stMetric:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(229, 9, 20, 0.3);
        border-color: #E50914;
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%) !important;
    }
    
    /* Metric label styling - bigger text, top aligned */
    .stMetric > div > div:first-child {
        font-size: 1.6rem !important;
        color: #B3B3B3 !important;
        font-weight: 700 !important;
        margin-bottom: 1.2rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        line-height: 1.2 !important;
    }
    
    /* Metric value styling - much bigger text */
    .stMetric > div > div:nth-child(2) {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.5) !important;
        line-height: 1.1 !important;
    }
    
    /* Metric delta styling - bigger text */
    .stMetric > div > div:nth-child(3) {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 20px !important;
        background: rgba(229, 9, 20, 0.1) !important;
        border: 1px solid rgba(229, 9, 20, 0.3) !important;
    }
    
    
    /* Chart container styling with Netflix theme */
    .stPlotlyChart {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
        border-radius: 8px !important;
        padding: 0 !important;
        box-shadow: 0 8px 32px rgba(229, 9, 20, 0.15) !important;
        border: 2px solid rgba(229, 9, 20, 0.3) !important;
        margin-bottom: 2rem !important;
        position: relative !important;
        box-sizing: border-box !important;
    }
    
    .stPlotlyChart::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #E50914 0%, #B81D24 100%);
        border-radius: 8px 8px 0 0;
    }
    
    /* Netflix-style button */
    .stButton > button {
        width: 280px !important;
        background: linear-gradient(135deg, #E50914 0%, #B81D24 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(229, 9, 20, 0.4) !important;
        margin: 0 auto !important;
        display: block !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #B81D24 0%, #8B0000 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(229, 9, 20, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Netflix-inspired title styling */
    h1 {
        color: #FFFFFF !important;
        font-size: 4rem !important;
        font-weight: 900 !important;
        margin-bottom: 2rem !important;
        text-align: center !important;
        background: linear-gradient(135deg, #E50914 0%, #B81D24 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-shadow: 0 4px 20px rgba(229, 9, 20, 0.3) !important;
        letter-spacing: -2px !important;
    }
    
    h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        margin-top: 3rem !important;
        margin-bottom: 1.5rem !important;
        font-size: 1.8rem !important;
    }
    
    /* Netflix-style section headers */
    .section-header {
        background: linear-gradient(90deg, #1a1a1a 0%, #2a2a2a 100%);
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin: 3rem 0 2rem 0;
        border-left: 4px solid #E50914;
        box-shadow: 0 4px 20px rgba(229, 9, 20, 0.1);
    }
    
    .section-header h3 {
        margin: 0 !important;
        color: #FFFFFF !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    /* Selectbox styling with Netflix theme */
    .stSelectbox {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        color: #FFFFFF;
        border-radius: 4px;
        border: 1px solid rgba(229, 9, 20, 0.3);
    }
    
    /* Custom grid spacing */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin-bottom: 3rem;
    }
    
    /* Netflix-style status indicators */
    .status-indicator {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 10px;
        box-shadow: 0 0 10px rgba(229, 9, 20, 0.3);
    }
    
    .status-positive { 
        background: linear-gradient(135deg, #46d369 0%, #2fb344 100%);
        box-shadow: 0 0 15px rgba(70, 211, 105, 0.4);
    }
    .status-negative { 
        background: linear-gradient(135deg, #E50914 0%, #B81D24 100%);
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.4);
    }
    .status-neutral { 
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        box-shadow: 0 0 15px rgba(251, 191, 36, 0.4);
    }
    
    /* Enhanced loading animation */
    .loading-spinner {
        border: 3px solid #1a1a1a;
        border-top: 3px solid #E50914;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 10px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Center button container */
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 3rem 0;
        padding: 2rem 0;
    }
    
    /* Last update styling */
    .update-info {
        text-align: center;
        color: #B3B3B3;
        font-size: 1rem;
        margin-top: 2rem;
        padding: 1rem;
        background: rgba(26, 26, 26, 0.5);
        border-radius: 8px;
        border: 1px solid rgba(229, 9, 20, 0.2);
    }
    
    /* Responsive adjustments for Netflix theme */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        
        h1 {
            font-size: 2.5rem !important;
            letter-spacing: -1px !important;
        }
        
        .stMetric {
            min-height: 160px !important;
            height: 160px !important;
            padding: 1.5rem 1rem !important;
        }
        
        .stMetric > div > div:nth-child(2) {
            font-size: 2.2rem !important;
        }
        
        .stButton > button {
            width: 240px !important;
            font-size: 1rem !important;
        }
        
        .metric-grid {
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return EnhancedStockForecastingPipeline()


pipeline = get_pipeline()


# Enhanced main title with icon and styling
st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1>Netflix Stock Analysis Dashboard</h1>
        <p style="color: #9CA3AF; font-size: 1.1rem; margin-top: -1rem;">
            Real-time stock analysis with AI-powered predictions
        </p>
    </div>
""", unsafe_allow_html=True)


# Create placeholders with enhanced section headers
st.markdown("""
    <div class="section-header">
        <h3>📊 Key Performance Metrics</h3>
    </div>
""", unsafe_allow_html=True)
price_cards = st.empty()


st.markdown("""
    <div class="section-header">
        <h3>📈 Price Analysis & Predictions</h3>
    </div>
""", unsafe_allow_html=True)
price_chart = st.empty()


st.markdown("""
    <div class="section-header">
        <h3>🔍 Technical Analysis Indicators</h3>
    </div>
""", unsafe_allow_html=True)
indicators = st.empty()


st.markdown("""
    <div class="section-header">
        <h3>🎯 Model Performance Metrics</h3>
    </div>
""", unsafe_allow_html=True)
model_metrics = st.empty()


# Auto-refresh mechanism
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()


def calculate_model_metrics(predictions_df):
    mae = np.mean(np.abs(predictions_df['Close'] - predictions_df['Predicted_Price']))
    rmse = np.sqrt(np.mean((predictions_df['Close'] - predictions_df['Predicted_Price'])**2))
    mape = np.mean(np.abs((predictions_df['Close'] - predictions_df['Predicted_Price']) / predictions_df['Close'])) * 100
    return mae, rmse, mape


def get_status_indicator(value, thresholds):
    """Generate status indicator based on value and thresholds"""
    if value > thresholds['high']:
        return '<span class="status-indicator status-negative"></span>'
    elif value < thresholds['low']:
        return '<span class="status-indicator status-positive"></span>'
    else:
        return '<span class="status-indicator status-neutral"></span>'


def fetch_and_process_data():
    df = pipeline.fetch_real_time_data()
    if df is not None:
        # Ensure Datetime column is timezone-aware
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        if df['Datetime'].dt.tz is None:
            df['Datetime'] = df['Datetime'].dt.tz_localize('America/New_York')
        df = pipeline.calculate_technical_indicators(df)
        return pipeline.make_predictions(df)
    return None


def update_dashboard():
    # Show loading indicator
    with st.spinner('📡 Fetching real-time data...'):
        predictions_df = fetch_and_process_data()
   
    if predictions_df is not None and not predictions_df.empty:
        latest = predictions_df.iloc[-1]
       
        # Enhanced price cards with status indicators
        col1, col2, col3, col4 = price_cards.columns(4)
       
        # Current Price with trend indicator
        price_change = ((latest['Close'] - predictions_df.iloc[0]['Close']) / predictions_df.iloc[0]['Close'] * 100)
        trend_indicator = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
        
        col1.metric(
            f"{trend_indicator} Current Price",
            f"${latest['Close']:.2f}",
            f"{price_change:+.2f}%",
            delta_color="normal"
        )
       
        # Predicted Price with accuracy indicator
        prediction_diff = ((latest['Predicted_Price'] - latest['Close']) / latest['Close'] * 100)
        accuracy_indicator = "🎯" if abs(prediction_diff) < 2 else "⚠️"
        
        col2.metric(
            f"{accuracy_indicator} Predicted Price",
            f"${latest['Predicted_Price']:.2f}",
            f"{prediction_diff:+.2f}%",
            delta_color="normal"
        )
       
        # RSI with enhanced status indicators
        rsi_value = latest['RSI']
        if rsi_value > 70:
            rsi_indicator = "🔴"
            rsi_status = "Overbought"
        elif rsi_value < 30:
            rsi_indicator = "🟢"
            rsi_status = "Oversold"
        else:
            rsi_indicator = "🟡"
            rsi_status = "Neutral"
            
        col3.metric(
            f"{rsi_indicator} RSI ({rsi_status})",
            f"{rsi_value:.1f}",
            help="Relative Strength Index - measures momentum"
        )
       
        # Volume with trend analysis
        volume_change = ((latest['Volume'] - predictions_df.iloc[0]['Volume']) / predictions_df.iloc[0]['Volume'] * 100)
        volume_indicator = "📊" if volume_change > 10 else "📉" if volume_change < -10 else "📈"
        
        col4.metric(
            f"{volume_indicator} Trading Volume",
            f"{latest['Volume']:,.0f}",
            f"{volume_change:+.2f}%"
        )
       
        # Enhanced price chart with better styling
        fig = go.Figure()
       
        # Add confidence intervals with gradient
        std_dev = np.std(predictions_df['Close'] - predictions_df['Predicted_Price'])
        fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['Predicted_Price'] + 2*std_dev,
            fill=None,
            mode='lines',
            line=dict(color='rgba(220, 38, 38, 0)', width=0),
            name='Confidence Interval',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['Predicted_Price'] - 2*std_dev,
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(220, 38, 38, 0)', width=0),
            fillcolor='rgba(220, 38, 38, 0.1)',
            name='95% Confidence Interval'
        ))
       
        # Add actual prices with enhanced styling
        fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['Close'],
            name='Actual Price',
            line=dict(color='#2563eb', width=3),
            marker=dict(size=6, color='#2563eb')
        ))
        
        # Add predicted prices
        fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['Predicted_Price'],
            name='AI Prediction',
            line=dict(color='#dc2626', width=3, dash='dot'),
            marker=dict(size=6, color='#dc2626')
        ))
       
        # Enhanced layout with Netflix styling
        unique_id = str(uuid.uuid4())
        fig.update_layout(
            title=dict(
                text='Stock Price vs AI Predictions',
                font=dict(size=24, color='#FFFFFF', family='Arial Black'),
                x=0.5
            ),
            xaxis=dict(
                title='Time',
                gridcolor='#333333',
                color='#B3B3B3',
                tickfont=dict(size=14, color='#FFFFFF')
            ),
            yaxis=dict(
                title='Price ($)',
                gridcolor='#333333',
                color='#B3B3B3',
                tickfont=dict(size=14, color='#FFFFFF')
            ),
            height=550,
            template='plotly_dark',
            paper_bgcolor='rgba(26, 26, 26, 0.9)',
            plot_bgcolor='rgba(26, 26, 26, 0.9)',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(26, 26, 26, 0.9)',
                bordercolor='#E50914',
                borderwidth=1,
                font=dict(size=12, color='#FFFFFF')
            ),
            hovermode='x unified',
            font=dict(color='#FFFFFF')
        )
        price_chart.plotly_chart(fig, use_container_width=True, key=f"price_chart_{unique_id}")
       
        # Enhanced technical indicators
        col1, col2 = indicators.columns(2)
       
        # EMA Chart with enhanced styling
        ema_fig = go.Figure()
        ema_fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['Close'],
            name='Current Price',
            line=dict(color='#FFFFFF', width=2),
            opacity=0.7
        ))
        ema_fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['EMA_5'],
            name='EMA 5 (Fast)',
            line=dict(color='#2563eb', width=2)
        ))
        ema_fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['EMA_20'],
            name='EMA 20 (Slow)',
            line=dict(color='#dc2626', width=2)
        ))
        ema_fig.update_layout(
            title=dict(
                text='Exponential Moving Averages',
                font=dict(size=20, color='#FFFFFF', family='Arial Black')
            ),
            height=450,
            template='plotly_dark',
            paper_bgcolor='rgba(26, 26, 26, 0.9)',
            plot_bgcolor='rgba(26, 26, 26, 0.9)',
            showlegend=True,
            xaxis=dict(gridcolor='#333333', color='#B3B3B3', tickfont=dict(size=12, color='#FFFFFF')),
            yaxis=dict(gridcolor='#333333', color='#B3B3B3', tickfont=dict(size=12, color='#FFFFFF')),
            hovermode='x unified',
            font=dict(color='#FFFFFF')
        )
        col1.plotly_chart(ema_fig, use_container_width=True, key=f"ema_chart_{unique_id}")
       
        # Volume Chart with color coding
        volume_fig = go.Figure()
        colors = ['#22c55e' if vol > predictions_df['Volume'].mean() else '#ef4444' 
                 for vol in predictions_df['Volume']]
        
        volume_fig.add_trace(go.Bar(
            x=predictions_df['Datetime'],
            y=predictions_df['Volume'],
            name='Trading Volume',
            marker=dict(
                color=colors,
                opacity=0.8,
                line=dict(width=0)
            )
        ))
        volume_fig.update_layout(
            title=dict(
                text='Trading Volume Analysis',
                font=dict(size=20, color='#FFFFFF', family='Arial Black')
            ),
            height=450,
            template='plotly_dark',
            paper_bgcolor='rgba(26, 26, 26, 0.9)',
            plot_bgcolor='rgba(26, 26, 26, 0.9)',
            showlegend=False,
            xaxis=dict(gridcolor='#333333', color='#B3B3B3', tickfont=dict(size=12, color='#FFFFFF')),
            yaxis=dict(gridcolor='#333333', color='#B3B3B3', tickfont=dict(size=12, color='#FFFFFF')),
            hovermode='x',
            font=dict(color='#FFFFFF')
        )
        col2.plotly_chart(volume_fig, use_container_width=True, key=f"volume_chart_{unique_id}")
       
        # Enhanced model performance metrics
        mae, rmse, mape = calculate_model_metrics(predictions_df)
        col1, col2, col3 = model_metrics.columns(3)
       
        # Determine performance status
        mae_status = "🟢" if mae < 5 else "🟡" if mae < 10 else "🔴"
        rmse_status = "🟢" if rmse < 7 else "🟡" if rmse < 15 else "🔴"
        mape_status = "🟢" if mape < 3 else "🟡" if mape < 5 else "🔴"
        
        col1.metric(
            f"{mae_status} Mean Absolute Error",
            f"${mae:.2f}",
            help="Average prediction error in dollars"
        )
        col2.metric(
            f"{rmse_status} Root Mean Square Error",
            f"${rmse:.2f}",
            help="Standard deviation of prediction errors"
        )
        col3.metric(
            f"{mape_status} Mean Absolute % Error",
            f"{mape:.2f}%",
            help="Average percentage prediction error"
        )

    else:
        st.error("❌ Unable to fetch data. Please check your connection and try again.")


# Auto-refresh logic with enhanced feedback
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.info("🔄 Auto-refreshing dashboard...")
    update_dashboard()


# Enhanced control panel with centered button
st.markdown("---")
st.markdown('<div class="button-container">', unsafe_allow_html=True)

if st.button('🔄 Refresh Data Now', help="Manually refresh all data and charts"):
    st.session_state.last_refresh = time.time()
    update_dashboard()
    st.success("✅ Dashboard updated successfully!")

st.markdown('</div>', unsafe_allow_html=True)

# Display last refresh time with Netflix styling
last_update = datetime.fromtimestamp(st.session_state.last_refresh)
st.markdown(f"""
    <div class="update-info">
        <strong>Last Updated:</strong> {last_update.strftime('%Y-%m-%d %H:%M:%S')} | 
        <strong>Next Auto-Refresh:</strong> {(last_update + timedelta(minutes=1)).strftime('%H:%M:%S')}
    </div>
""", unsafe_allow_html=True)

# Initial load
update_dashboard()


# Enhanced auto-refresh with better user feedback
st.markdown(
    """
    <script>
        let refreshTimer;
        
        function startRefreshTimer() {
            refreshTimer = setTimeout(() => {
                console.log('Auto-refreshing dashboard...');
                window.location.reload();
            }, 60000);
        }
        
        function resetRefreshTimer() {
            clearTimeout(refreshTimer);
            startRefreshTimer();
        }
        
        // Start the timer
        startRefreshTimer();
        
        // Reset timer on user interaction
        document.addEventListener('click', resetRefreshTimer);
        document.addEventListener('keypress', resetRefreshTimer);
    </script>
    """,
    unsafe_allow_html=True
)