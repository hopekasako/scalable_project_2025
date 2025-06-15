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


# Enhanced Custom CSS with Netflix theme and optimized styling
st.markdown("""
    <style>
    /* Import Netflix-style fonts */
    @import url('https://fonts.googleapis.com/css2?family=Netflix+Sans:wght@400;500;700;900&display=swap');
    
    /* Global Netflix color variables */
    :root {
        --netflix-red: #E50914;
        --netflix-red-dark: #B81D24;
        --netflix-black: #000000;
        --netflix-dark-gray: #141414;
        --netflix-medium-gray: #2F2F2F;
        --netflix-light-gray: #B3B3B3;
        --netflix-white: #FFFFFF;
        --netflix-success: #46d369;
        --netflix-warning: #fbbf24;
        --netflix-error: #ef4444;
    }
    
    /* Main container styling with Netflix theme */
    .main {
        padding: 1.5rem;  
        background: linear-gradient(135deg, var(--netflix-black) 0%, var(--netflix-dark-gray) 100%);
        color: var(--netflix-white);
        min-height: 100vh;
        font-family: 'Netflix Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Netflix-inspired metric box styling with improved layout */
    .stMetric {
        background: linear-gradient(145deg, rgba(20, 20, 20, 0.9), rgba(47, 47, 47, 0.7)) !important;
        backdrop-filter: blur(12px) !important;
        width: 100% !important;
        min-width: 0 !important;
        min-height: 200px !important;
        height: 200px !important;
        padding: 2rem 1.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px rgba(229, 9, 20, 0.15), 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(229, 9, 20, 0.2) !important;
        color: var(--netflix-white) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        align-items: flex-start !important;
        margin: 0 !important;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    /* Netflix signature red accent bar */
    .stMetric::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--netflix-red) 0%, var(--netflix-red-dark) 100%);
        border-radius: 12px 12px 0 0;
    }
    
    /* Enhanced hover effects */
    .stMetric:hover {
        background: linear-gradient(145deg, rgba(47, 47, 47, 0.9), rgba(60, 60, 60, 0.8)) !important;
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 48px rgba(229, 9, 20, 0.25), 0 8px 24px rgba(0, 0, 0, 0.4);
        border-color: var(--netflix-red);
    }
    
    /* Metric label styling - Netflix typography */
    .stMetric > div > div:first-child {
        font-size: 1.4rem !important;
        color: var(--netflix-light-gray) !important;
        font-weight: 500 !important;
        margin-bottom: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        line-height: 1.2 !important;
        opacity: 0.9 !important;
    }
    
    /* Metric value styling - Netflix bold numbers */
    .stMetric > div > div:nth-child(2) {
        font-size: 1.7rem! important;
        font-weight: 700 !important;
        color: var(--netflix-white) !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 4px 12px rgba(0,0,0,0.6) !important;
        line-height: 1 !important;
        font-family: 'Netflix Sans', monospace !important;
    }
    
    /* Metric delta styling - Netflix pill design */
    .stMetric > div > div:nth-child(3) {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.4rem 1rem !important;
        border-radius: 25px !important;
        background: rgba(229, 9, 20, 0.15) !important;
        border: 1px solid rgba(229, 9, 20, 0.4) !important;
        backdrop-filter: blur(8px) !important;
        transition: all 0.3s ease !important;
    }
    
    /* Chart container styling with Netflix theme */
    .stPlotlyChart {
        background: linear-gradient(145deg, rgba(20, 20, 20, 0.95), rgba(47, 47, 47, 0.8)) !important;
        backdrop-filter: blur(16px) !important;
        border-radius: 12px !important;
        padding: 0 !important;
        box-shadow: 0 12px 40px rgba(229, 9, 20, 0.12), 0 4px 16px rgba(0, 0, 0, 0.3) !important;
        border: 1px solid rgba(229, 9, 20, 0.2) !important;
        margin-bottom: 2.5rem !important;
        position: relative !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }
    
    .stPlotlyChart::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--netflix-red) 0%, var(--netflix-red-dark) 100%);
        border-radius: 12px 12px 0 0;
        z-index: 1;
    }
    
    /* Netflix-style CTA button */
    .stButton > button {
        width: 320px !important;
        background: linear-gradient(135deg, var(--netflix-red) 0%, var(--netflix-red-dark) 100%) !important;
        color: var(--netflix-white) !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 1.2rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        box-shadow: 0 6px 24px rgba(229, 9, 20, 0.4) !important;
        margin: 0 auto !important;
        display: block !important;
        font-family: 'Netflix Sans', sans-serif !important;
        cursor: pointer !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--netflix-red-dark) 0%, #8B0000 100%) !important;
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 36px rgba(229, 9, 20, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.01) !important;
    }
    
    /* Netflix-inspired hero title */
    .netflix-hero {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2.5rem 0;
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, transparent 100%);
        border-radius: 16px;
        border: 1px solid rgba(229, 9, 20, 0.1);
    }
    
    .netflix-logo {
        height: 150px;
        width: 250px;
        margin-bottom: 0.5rem;
        filter: brightness(1.1);
        transition: transform 0.3s ease;
    }
    
    .netflix-logo:hover {
        transform: scale(1.05);
    }
    
    .hero-title {
        color: var(--netflix-white) !important;
        font-size: 3rem !important;
        font-weight: 900 !important;
        margin-bottom: 1rem !important;
        background: linear-gradient(135deg, var(--netflix-red) 0%, var(--netflix-red-dark) 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-shadow: 0 4px 20px rgba(229, 9, 20, 0.3) !important;
        letter-spacing: -3px !important;
        font-family: 'Netflix Sans', sans-serif !important;
    }
    
    .hero-subtitle {
        color: var(--netflix-light-gray) !important;
        font-size: 1.2rem !important;
        font-weight: 200 !important;
        margin-top: -0.5rem !important;
        opacity: 0.9 !important;
    }
    
    /* Netflix-style section headers with improved design */
    .section-header {
        background: linear-gradient(135deg, rgba(20, 20, 20, 0.8) 0%, rgba(47, 47, 47, 0.6) 100%);
        backdrop-filter: blur(12px);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin: 3.5rem 0 2.5rem 0;
        border-left: 6px solid var(--netflix-red);
        box-shadow: 0 8px 32px rgba(229, 9, 20, 0.1), 0 2px 8px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100%;
        background: linear-gradient(90deg, transparent 0%, rgba(229, 9, 20, 0.05) 100%);
    }
    
    .section-header h3 {
        margin: 0 !important;
        color: var(--netflix-white) !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        display: flex !important;
        align-items: center !important;
        gap: 1rem !important;
    }
    
    /* Enhanced grid layout */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 2.5rem;
        margin-bottom: 3.5rem;
    }
    
    /* Netflix-style status indicators with animations */
    .status-indicator {
        display: inline-block;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        margin-right: 12px;
        position: relative;
        animation: pulse 2s infinite;
    }
    
    .status-positive { 
        background: linear-gradient(135deg, var(--netflix-success) 0%, #2fb344 100%);
        box-shadow: 0 0 20px rgba(70, 211, 105, 0.5);
    }
    .status-negative { 
        background: linear-gradient(135deg, var(--netflix-red) 0%, var(--netflix-red-dark) 100%);
        box-shadow: 0 0 20px rgba(229, 9, 20, 0.5);
    }
    .status-neutral { 
        background: linear-gradient(135deg, var(--netflix-warning) 0%, #f59e0b 100%);
        box-shadow: 0 0 20px rgba(251, 191, 36, 0.5);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    /* Enhanced loading animation */
    .loading-spinner {
        border: 4px solid rgba(229, 9, 20, 0.1);
        border-top: 4px solid var(--netflix-red);
        border-radius: 50%;
        width: 32px;
        height: 32px;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 12px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Button container with Netflix styling */
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 4rem 0;
        padding: 3rem 0;
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.05) 0%, transparent 100%);
        border-radius: 16px;
        border: 1px solid rgba(229, 9, 20, 0.1);
    }
    
    /* Last update info with Netflix design */
    .update-info {
        text-align: center;
        color: var(--netflix-light-gray);
        font-size: 1.1rem;
        margin-top: 3rem;
        padding: 1.5rem 2rem;
        background: linear-gradient(145deg, rgba(20, 20, 20, 0.8), rgba(47, 47, 47, 0.6));
        backdrop-filter: blur(12px);
        border-radius: 12px;
        border: 1px solid rgba(229, 9, 20, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Custom scrollbar with Netflix theme */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--netflix-dark-gray);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--netflix-red) 0%, var(--netflix-red-dark) 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--netflix-red-dark) 0%, #8B0000 100%);
    }
    
    /* Responsive design improvements */
    @media (max-width: 1200px) {
        .hero-title {
            font-size: 3.5rem !important;
            letter-spacing: -2px !important;
        }
        
        .metric-grid {
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
        }
    }
    
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        
        .hero-title {
            font-size: 2.8rem !important;
            letter-spacing: -1px !important;
        }
        
        .hero-subtitle {
            font-size: 1.1rem !important;
        }
        
        .stMetric {
            min-height: 180px !important;
            height: 180px !important;
            padding: 1.5rem 1.2rem !important;
        }
        
        .stMetric > div > div:nth-child(2) {
            font-size: 2.6rem !important;
        }
        
        .stButton > button {
            width: 280px !important;
            font-size: 1rem !important;
            padding: 1rem 2rem !important;
        }
        
        .section-header {
            padding: 1.5rem 2rem;
            margin: 2.5rem 0 2rem 0;
        }
        
        .section-header h3 {
            font-size: 1.5rem !important;
        }
        
        .metric-grid {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
    }
    
    @media (max-width: 480px) {
        .hero-title {
            font-size: 2.2rem !important;
        }
        
        .stMetric > div > div:nth-child(2) {
            font-size: 2.2rem !important;
        }
        
        .stButton > button {
            width: 240px !important;
        }
    }
    
    /* Accessibility improvements */
    .stMetric:focus-within {
        outline: 2px solid var(--netflix-red);
        outline-offset: 2px;
    }
    
    /* Custom animations for smooth transitions */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stMetric, .stPlotlyChart, .section-header {
        animation: fadeInUp 0.6s ease-out;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return EnhancedStockForecastingPipeline()


pipeline = get_pipeline()


# Enhanced Netflix-style hero section
st.markdown("""
    <div class="netflix-hero">
        <img src="https://logos-world.net/wp-content/uploads/2020/04/Netflix-Logo.png" alt="Netflix Logo" class="netflix-logo">
        <h1 class="hero-title">Stock Analysis Dashboard</h1>
        <p class="hero-subtitle">
            Real-time stock analysis powered by AI • Advanced predictions • Live market data
        </p>
    </div>
""", unsafe_allow_html=True)


# Create placeholders with enhanced section headers
st.markdown("""
    <div class="section-header">
        <h3><span>📊</span> Key Performance Metrics</h3>
    </div>
""", unsafe_allow_html=True)
price_cards = st.empty()


st.markdown("""
    <div class="section-header">
        <h3><span>📈</span> Price Analysis & AI Predictions</h3>
    </div>
""", unsafe_allow_html=True)
price_chart = st.empty()


st.markdown("""
    <div class="section-header">
        <h3><span>🔍</span> Technical Analysis Indicators</h3>
    </div>
""", unsafe_allow_html=True)
indicators = st.empty()


st.markdown("""
    <div class="section-header">
        <h3><span>🎯</span> AI Model Performance Metrics</h3>
    </div>
""", unsafe_allow_html=True)
model_metrics = st.empty()


# Auto-refresh mechanism
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()


def calculate_model_metrics(predictions_df):
    """Calculate AI model performance metrics"""
    mae = np.mean(np.abs(predictions_df['Close'] - predictions_df['Predicted_Price']))
    rmse = np.sqrt(np.mean((predictions_df['Close'] - predictions_df['Predicted_Price'])**2))
    mape = np.mean(np.abs((predictions_df['Close'] - predictions_df['Predicted_Price']) / predictions_df['Close'])) * 100
    return mae, rmse, mape


def get_trend_emoji(value, positive_threshold=0, negative_threshold=0):
    """Get appropriate emoji based on trend"""
    if value > positive_threshold:
        return "📈"
    elif value < negative_threshold:
        return "📉"
    else:
        return "➡️"


def get_rsi_status(rsi_value):
    """Get RSI status with appropriate indicators"""
    if rsi_value > 70:
        return "🔴", "Overbought", "Strong sell signal"
    elif rsi_value < 30:
        return "🟢", "Oversold", "Strong buy signal"
    elif rsi_value > 60:
        return "🟡", "Neutral-High", "Moderate sell signal"
    elif rsi_value < 40:
        return "🟡", "Neutral-Low", "Moderate buy signal"
    else:
        return "⚪", "Neutral", "Hold signal"


def fetch_and_process_data():
    """Fetch and process real-time stock data"""
    df = pipeline.fetch_real_time_data()
    if df is not None:
        # Ensure Datetime column is timezone-aware
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        if df['Datetime'].dt.tz is None:
            df['Datetime'] = df['Datetime'].dt.tz_localize('America/New_York')
        df = pipeline.calculate_technical_indicators(df)
        return pipeline.make_predictions(df)
    return None


def create_enhanced_plotly_theme():
    """Create enhanced Plotly theme matching Netflix design"""
    return {
        'layout': {
            'paper_bgcolor': 'rgba(20, 20, 20, 0.95)',
            'plot_bgcolor': 'rgba(20, 20, 20, 0.95)',
            'font': {'color': '#FFFFFF', 'family': 'Netflix Sans, Arial, sans-serif'},
            'colorway': ['#E50914', '#2563eb', '#22c55e', '#fbbf24', '#8b5cf6'],
            'xaxis': {
                'gridcolor': '#333333',
                'color': '#B3B3B3',
                'tickfont': {'size': 12, 'color': '#FFFFFF'},
                'linecolor': '#333333'
            },
            'yaxis': {
                'gridcolor': '#333333',
                'color': '#B3B3B3',
                'tickfont': {'size': 12, 'color': '#FFFFFF'},
                'linecolor': '#333333'
            }
        }
    }


def update_dashboard():
    """Update dashboard with latest data and enhanced visualizations"""
    # Show enhanced loading indicator
    with st.spinner('🔄 Fetching real-time market data...'):
        predictions_df = fetch_and_process_data()
   
    if predictions_df is not None and not predictions_df.empty:
        latest = predictions_df.iloc[-1]
        previous = predictions_df.iloc[0] if len(predictions_df) > 1 else latest
       
        # Enhanced price cards with better status indicators
        col1, col2, col3, col4 = price_cards.columns(4)
       
        # Current Price with enhanced trend analysis
        price_change = ((latest['Close'] - previous['Close']) / previous['Close'] * 100)
        trend_emoji = get_trend_emoji(price_change, 0.5, -0.5)
        
        col1.metric(
            f"{trend_emoji} Current Price",
            f"${latest['Close']:.2f}",
            f"{price_change:+.2f}%",
            delta_color="normal",
            help="Real-time stock price with trend analysis"
        )
       
        # AI Prediction with confidence indicator
        prediction_diff = ((latest['Predicted_Price'] - latest['Close']) / latest['Close'] * 100)
        accuracy_emoji = "🎯" if abs(prediction_diff) < 2 else "⚠️" if abs(prediction_diff) < 5 else "❌"
        
        col2.metric(
            f"{accuracy_emoji} AI Prediction",
            f"${latest['Predicted_Price']:.2f}",
            f"{prediction_diff:+.2f}%",
            delta_color="normal",
            help="AI-powered price prediction with confidence level"
        )
       
        # RSI with enhanced status analysis
        rsi_value = latest['RSI']
        rsi_emoji, rsi_status, rsi_help = get_rsi_status(rsi_value)
            
        col3.metric(
            f"{rsi_emoji} RSI ({rsi_status})",
            f"{rsi_value:.1f}",
            help=f"Relative Strength Index - {rsi_help}"
        )
       
        # Volume with advanced trend analysis
        volume_change = ((latest['Volume'] - previous['Volume']) / previous['Volume'] * 100) if previous['Volume'] > 0 else 0
        volume_emoji = "🔥" if volume_change > 20 else "📊" if volume_change > 0 else "📉"
        
        col4.metric(
            f"{volume_emoji} Trading Volume",
            f"{latest['Volume']:,.0f}",
            f"{volume_change:+.2f}%",
            help="Trading volume with trend analysis"
        )
       
        # Enhanced price chart with Netflix styling
        fig = go.Figure()
       
        # Add confidence intervals with better styling
        std_dev = np.std(predictions_df['Close'] - predictions_df['Predicted_Price'])
        fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['Predicted_Price'] + 2*std_dev,
            fill=None,
            mode='lines',
            line=dict(color='rgba(229, 9, 20, 0)', width=0),
            name='Confidence Band',
            showlegend=False,
            hoverinfo='skip'
        ))
       
        # Add actual prices with enhanced styling
        fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['Close'],
            name='Actual Price',
            line=dict(color='#DC2626', width=1.5, shape='linear'),
            marker=dict(size=4, color='#DC2626', line=dict(width=1, color='#DC2626')),
            hovertemplate='<b>Actual Price</b><br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Add predicted prices with Netflix red
        fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['Predicted_Price'],
            name='AI Prediction',
            line=dict(color='#2563EB', width=1.5, shape='linear'),
            marker=dict(size=4, color='#2563EB'),
            hovertemplate='<b>AI Prediction</b><br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
       
        # Enhanced layout with Netflix theme
        unique_id = str(uuid.uuid4())
        fig.update_layout(
            title=dict(
                text='🤖 Stock Price vs AI Predictions',
                font=dict(size=24, color='#FFFFFF', family='Netflix Sans, Arial, sans-serif'),
                x=0.02,
                y=0.95
            ),
            xaxis=dict(
                title='Time Period',
                gridcolor='#333333',
                color='#B3B3B3',
                tickfont=dict(size=12, color='#FFFFFF'),
                linecolor='#333333'
            ),
            yaxis=dict(
                title='Price (USD)',
                gridcolor='#333333',
                color='#B3B3B3',
                tickfont=dict(size=12, color='#FFFFFF'),
                side='right',
                linecolor='#333333'
            ),
            height=600,
            paper_bgcolor='rgba(20, 20, 20, 0.95)',
            plot_bgcolor='rgba(20, 20, 20, 0.95)',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(20, 20, 20, 0.9)',
                bordercolor='#E50914',
                borderwidth=1,
                font=dict(size=12, color='#FFFFFF')
            ),
            hovermode='x unified',
            font=dict(color='#FFFFFF', family='Netflix Sans, Arial, sans-serif')
        )
        price_chart.plotly_chart(fig, use_container_width=True, key=f"price_chart_{unique_id}")
       
        # Enhanced technical indicators with Netflix styling
        col1, col2 = indicators.columns(2)
       
        # EMA Chart with enhanced Netflix styling
        ema_fig = go.Figure()
        ema_fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['Close'],
            name='Current Price',
            line=dict(color='#FFFFFF', width=1.5, shape='linear'),
            opacity=0.8,
            hovertemplate='<b>Current Price</b><br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        ema_fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['EMA_5'],
            name='EMA 5 (Fast Signal)',
            line=dict(color='#22c55e', width=1.5, shape='linear'),
            hovertemplate='<b>EMA 5</b><br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        ema_fig.add_trace(go.Scatter(
            x=predictions_df['Datetime'],
            y=predictions_df['EMA_20'],
            name='EMA 20 (Trend)',
            line=dict(color='#E50914', width=1.5, shape='linear'),
            hovertemplate='<b>EMA 20</b><br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        ema_fig.update_layout(
            title=dict(
                text='📊 Exponential Moving Averages',
                font=dict(size=20, color='#FFFFFF', family='Netflix Sans, Arial, sans-serif'),
                x=0.02
            ),
            height=500,
            paper_bgcolor='rgba(20, 20, 20, 0.95)',
            plot_bgcolor='rgba(20, 20, 20, 0.95)',
            showlegend=True,
            xaxis=dict(
                gridcolor='#333333', 
                color='#B3B3B3', 
                tickfont=dict(size=11, color='#FFFFFF'),
                linecolor='#333333'
            ),
            yaxis=dict(
                gridcolor='#333333', 
                color='#B3B3B3', 
                tickfont=dict(size=11, color='#FFFFFF'),
                linecolor='#333333'
            ),
            hovermode='x unified',
            font=dict(color='#FFFFFF', family='Netflix Sans, Arial, sans-serif'),
            legend=dict(
                bgcolor='rgba(20, 20, 20, 0.9)',
                bordercolor='#E50914',
                borderwidth=1,
                font=dict(size=10, color='#FFFFFF')
            )
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
            height=500,
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
       
        # Enhanced AI model performance metrics
        mae, rmse, mape = calculate_model_metrics(predictions_df)
        col1, col2, col3 = model_metrics.columns(3)
       
        # Enhanced performance status with detailed analysis
        def get_performance_status(metric, thresholds):
            if metric < thresholds[0]:
                return "🟢", "Excellent", "High accuracy"
            elif metric < thresholds[1]:
                return "🟡", "Good", "Moderate accuracy"
            elif metric < thresholds[2]:
                return "🟠", "Fair", "Acceptable accuracy"
            else:
                return "🔴", "Poor", "Low accuracy"
        
        mae_emoji, mae_status, mae_desc = get_performance_status(mae, [2, 5, 10])
        rmse_emoji, rmse_status, rmse_desc = get_performance_status(rmse, [3, 7, 15])
        mape_emoji, mape_status, mape_desc = get_performance_status(mape, [1.5, 3, 5])
        
        col1.metric(
            f"{mae_emoji} Mean Absolute Error",
            f"${mae:.2f}",
            f"{mae_status} - {mae_desc}",
            help="Average prediction error in dollars - lower is better"
        )
        col2.metric(
            f"{rmse_emoji} Root Mean Square Error",
            f"${rmse:.2f}",
            f"{rmse_status} - {rmse_desc}",
            help="Standard deviation of prediction errors - measures consistency"
        )
        col3.metric(
            f"{mape_emoji} Mean Absolute % Error",
            f"{mape:.2f}%",
            f"{mape_status} - {mape_desc}",
            help="Average percentage prediction error - relative accuracy measure"
        )
        
        # Add additional insights section
        st.markdown("""
            <div class="section-header">
                <h3><span>💡</span> AI Insights & Market Analysis</h3>
            </div>
        """, unsafe_allow_html=True)
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            # Market sentiment analysis
            price_momentum = (latest['Close'] - predictions_df['Close'].iloc[-5]) / predictions_df['Close'].iloc[-5] * 100 if len(predictions_df) > 5 else 0
            volume_trend = (latest['Volume'] - predictions_df['Volume'].mean()) / predictions_df['Volume'].mean() * 100
            
            sentiment_score = (price_momentum * 0.4 + (100 - rsi_value) * 0.3 + volume_trend * 0.3) / 100
            
            if sentiment_score > 0.2:
                sentiment_emoji, sentiment_text, sentiment_color = "🟢", "Bullish", "#22c55e"
            elif sentiment_score < -0.2:
                sentiment_emoji, sentiment_text, sentiment_color = "🔴", "Bearish", "#E50914"
            else:
                sentiment_emoji, sentiment_text, sentiment_color = "🟡", "Neutral", "#fbbf24"
            
            st.markdown(f"""
                <div style="
                    background: linear-gradient(145deg, rgba(20, 20, 20, 0.8), rgba(47, 47, 47, 0.6));
                    backdrop-filter: blur(12px);
                    padding: 2rem;
                    border-radius: 12px;
                    border-left: 4px solid {sentiment_color};
                    margin-bottom: 1.5rem;
                    height: 250px;
                ">
                    <h4 style="color: #FFFFFF; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                        {sentiment_emoji} Market Sentiment: <span style="color: {sentiment_color};">{sentiment_text}</span>
                    </h4>
                    <p style="color: #B3B3B3; margin-bottom: 0.5rem;">
                        <strong>Price Momentum:</strong> {price_momentum:+.2f}% (5-period)
                    </p>
                    <p style="color: #B3B3B3; margin-bottom: 0.5rem;">
                        <strong>Volume vs Average:</strong> {volume_trend:+.2f}%
                    </p>
                    <p style="color: #B3B3B3; margin: 0;">
                        <strong>AI Confidence:</strong> {100 - abs(prediction_diff):.1f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with insights_col2:
            # Trading signals summary
            ema_signal = "Buy" if latest['EMA_5'] > latest['EMA_20'] else "Sell"
            ema_signal_color = "#22c55e" if ema_signal == "Buy" else "#E50914"
            
            rsi_signal = "Buy" if rsi_value < 30 else "Sell" if rsi_value > 70 else "Hold"
            rsi_signal_color = "#22c55e" if rsi_signal == "Buy" else "#E50914" if rsi_signal == "Sell" else "#fbbf24"
            
            st.markdown(f"""
                <div style="
                    background: linear-gradient(145deg, rgba(20, 20, 20, 0.8), rgba(47, 47, 47, 0.6));
                    backdrop-filter: blur(12px);
                    padding: 2rem;
                    border-radius: 12px;
                    border-left: 4px solid #E50914;
                    margin-bottom: 1.5rem;
                    height: 250px;
                ">
                    <h4 style="color: #FFFFFF; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                        🎯 Trading Signals Summary
                    </h4>
                    <p style="color: #B3B3B3; margin-bottom: 0.5rem;">
                        <strong>EMA Crossover:</strong> <span style="color: {ema_signal_color};">{ema_signal}</span>
                    </p>
                    <p style="color: #B3B3B3; margin-bottom: 0.5rem;">
                        <strong>RSI Signal:</strong> <span style="color: {rsi_signal_color};">{rsi_signal}</span>
                    </p>
                    <p style="color: #B3B3B3; margin-bottom: 0.5rem;">
                        <strong>AI Prediction:</strong> <span style="color: {'#22c55e' if prediction_diff > 0 else '#E50914'};">
                            {'Upward' if prediction_diff > 0 else 'Downward'} ({abs(prediction_diff):.2f}%)
                        </span>
                    </p>
                    <p style="color: #B3B3B3; margin: 0;">
                        <strong>Overall Signal:</strong> <span style="color: {sentiment_color};">{sentiment_text}</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)

    else:
        st.error("❌ Unable to fetch market data. Please check your connection and try refreshing.")
        st.markdown("""
            <div style="
                text-align: center;
                padding: 3rem;
                background: linear-gradient(145deg, rgba(239, 68, 68, 0.1), rgba(185, 28, 28, 0.05));
                border-radius: 12px;
                border: 1px solid rgba(239, 68, 68, 0.3);
                margin: 2rem 0;
            ">
                <h3 style="color: #ef4444; margin-bottom: 1rem;">🚫 Data Unavailable</h3>
                <p style="color: #B3B3B3;">
                    The market data service is currently unavailable. This could be due to:
                </p>
                <ul style="color: #B3B3B3; text-align: left; max-width: 500px; margin: 1rem auto;">
                    <li>Network connectivity issues</li>
                    <li>API rate limits</li>
                    <li>Market closure periods</li>
                    <li>Service maintenance</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


# Auto-refresh logic with enhanced feedback
if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.info("🔄 Auto-refreshing dashboard with latest market data...")
    update_dashboard()


# Enhanced control panel with Netflix-style button
st.markdown("---")

refresh_button = st.button(
    '🔄 Refresh Market Data', 
    help="Manually refresh all data, charts, and AI predictions",
    use_container_width=False
)

if refresh_button:
    st.session_state.last_refresh = time.time()
    with st.spinner("🤖 Updating AI predictions and market analysis..."):
        update_dashboard()
    st.success("✅ Dashboard updated with latest market data and AI insights!")

st.markdown('</div>', unsafe_allow_html=True)

# Enhanced status display with Netflix styling
last_update = datetime.fromtimestamp(st.session_state.last_refresh)
next_update = last_update + timedelta(minutes=1)
time_until_refresh = int((next_update - datetime.now()).total_seconds())

st.markdown(f"""
    <div class="update-info">
        <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; flex-wrap: wrap;">
            <div>
                <strong>🕒 Last Updated:</strong> {last_update.strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            <div>
                <strong>⏰ Next Auto-Refresh:</strong> {next_update.strftime('%H:%M:%S')}
            </div>
            <div>
                <strong>⏳ Time Remaining:</strong> {max(0, time_until_refresh)}s
            </div>
        </div>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(229, 9, 20, 0.2);">
            <small style="color: #888;">
                🤖 Powered by AI • 📊 Real-time data • 🎯 Advanced analytics
            </small>
        </div>
    </div>
""", unsafe_allow_html=True)

# Initial dashboard load
update_dashboard()

# Enhanced auto-refresh with better user experience
st.markdown(
    """
    <script>
        let refreshTimer;
        let countdownTimer;
        
        function updateCountdown() {
            const nextRefresh = new Date(Date.now() + 60000);
            const now = new Date();
            const timeLeft = Math.max(0, Math.floor((nextRefresh - now) / 1000));
            
            // Update countdown display if element exists
            const countdownElements = document.querySelectorAll('[data-countdown]');
            countdownElements.forEach(el => {
                el.textContent = timeLeft + 's';
            });
        }
        
        function startRefreshTimer() {
            refreshTimer = setTimeout(() => {
                console.log('🔄 Auto-refreshing Netflix dashboard...');
                window.location.reload();
            }, 60000);
            
            // Start countdown timer
            countdownTimer = setInterval(updateCountdown, 1000);
        }
        
        function resetRefreshTimer() {
            clearTimeout(refreshTimer);
            clearInterval(countdownTimer);
            startRefreshTimer();
        }
        
        // Initialize timers
        startRefreshTimer();
        
        // Reset timer on user interaction
        document.addEventListener('click', resetRefreshTimer);
        document.addEventListener('keypress', resetRefreshTimer);
        document.addEventListener('scroll', resetRefreshTimer);
        
        // Add loading states for better UX
        document.addEventListener('DOMContentLoaded', function() {
            const buttons = document.querySelectorAll('button');
            buttons.forEach(button => {
                button.addEventListener('click', function() {
                    if (this.textContent.includes('Refresh')) {
                        this.innerHTML = '<div class="loading-spinner"></div>Refreshing...';
                        this.disabled = true;
                    }
                });
            });
        });
    </script>
    """,
    unsafe_allow_html=True
)