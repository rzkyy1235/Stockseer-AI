"""
╔══════════════════════════════════════════════════════════════════╗
║   Stock Price Prediction & Trading Signal — Hybrid LSTM+LGB     ║
║   Streamlit App  |  Technical Indicators  |  Signal Dashboard    ║
╚══════════════════════════════════════════════════════════════════╝

Requirements:
    pip install streamlit yfinance pandas numpy scikit-learn \
                tensorflow lightgbm ta plotly
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import ta
import lightgbm as lgb
import joblib, io, os

# TensorFlow / Keras ─ lazy import to speed startup
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input,
                                     BatchNormalization, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSeer AI — Hybrid LSTM+LightGBM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== LOGIN SYSTEM =====
USER_CREDENTIALS = {
    "admin": "123",
    "rizkysyahp05@gmail.com": "123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    user = st.session_state.username
    pw = st.session_state.password

    if user in USER_CREDENTIALS and USER_CREDENTIALS[user] == pw:
        st.session_state.logged_in = True
    else:
        st.error("Username / Password salah!")

def landing_page():
    # Custom CSS for modern login page
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');
    
    .login-container {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background: radial-gradient(circle at 10% 20%, #0C1120, #05070C);
        padding: 24px;
    }
    
    .glass-card {
        background: rgba(17, 21, 32, 0.85);
        backdrop-filter: blur(14px);
        border-radius: 28px;
        border: 1px solid rgba(30, 37, 53, 0.7);
        box-shadow: 0 25px 45px rgba(0, 0, 0, 0.5), 0 0 0 0.5px rgba(0, 245, 160, 0.1) inset;
        width: 100%;
        max-width: 1100px;
        display: flex;
        flex-direction: row;
        overflow: hidden;
    }
    
    .brand-panel {
        flex: 1.2;
        background: linear-gradient(135deg, rgba(0, 20, 30, 0.6), rgba(0, 0, 0, 0.4));
        padding: 48px 40px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border-right: 1px solid #1E2535;
    }
    
    .company-badge {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 48px;
    }
    
    .logo-icon {
        font-size: 2.2rem;
        background: linear-gradient(135deg, #00F5A0, #00B4FF);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    
    .company-name {
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 1.25rem;
        background: linear-gradient(90deg, #E2E8F0, #94A3B8);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    
    .nice-to-see {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #00F5A0;
        background: rgba(0, 245, 160, 0.12);
        display: inline-block;
        padding: 6px 16px;
        border-radius: 30px;
        margin-bottom: 24px;
        font-weight: 600;
    }
    
    .welcome-back {
        font-size: 3.2rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        background: linear-gradient(135deg, #FFFFFF, #00B4FF);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        line-height: 1.2;
        margin-bottom: 20px;
    }
    
    .lorem-text {
        color: #64748B;
        font-size: 0.9rem;
        line-height: 1.5;
        max-width: 90%;
        margin-top: 12px;
        border-left: 2px solid #00F5A0;
        padding-left: 16px;
    }
    
    .form-panel {
        flex: 1;
        padding: 48px 44px;
        background: rgba(17, 21, 32, 0.5);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .form-header h2 {
        font-size: 1.9rem;
        font-weight: 600;
        background: linear-gradient(90deg, #fff, #94A3B8);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 10px;
    }
    
    .form-header p {
        color: #64748B;
        font-size: 0.85rem;
    }
    
    .login-input-group {
        margin-bottom: 24px;
    }
    
    .login-input-label {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748B;
        margin-bottom: 8px;
    }
    
    .login-input-label i {
        font-size: 0.8rem;
        color: #00F5A0;
    }
    
    .stTextInput > div > div {
        background: rgba(10, 13, 20, 0.7) !important;
        border: 1.5px solid #1E2535 !important;
        border-radius: 14px !important;
        color: #E2E8F0 !important;
    }
    
    .login-checkbox {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #64748B;
        font-size: 0.8rem;
        margin: 20px 0;
    }
    
    .login-subscribe-btn {
        width: 100%;
        background: linear-gradient(95deg, #00F5A0, #00B4FF);
        border: none;
        border-radius: 40px;
        padding: 14px 0;
        font-weight: 700;
        font-size: 1rem;
        font-family: 'Space Mono', monospace;
        color: #05070C;
        cursor: pointer;
        transition: all 0.25s;
        margin-top: 20px;
    }
    
    .demo-hint {
        margin-top: 30px;
        font-size: 0.7rem;
        text-align: center;
        color: #2d3a4f;
        border-top: 1px dashed #1E2535;
        padding-top: 20px;
    }
    
    .demo-hint span {
        color: #00F5A0;
        font-family: 'Space Mono', monospace;
    }
    
    @media (max-width: 780px) {
        .glass-card {
            flex-direction: column;
            max-width: 480px;
        }
        .brand-panel {
            padding: 32px 28px;
            border-right: none;
            border-bottom: 1px solid #1E2535;
        }
        .welcome-back {
            font-size: 2.4rem;
        }
        .form-panel {
            padding: 40px 28px;
        }
        .lorem-text {
            max-width: 100%;
        }
    }
    </style>
    
    <div class="login-container">
        <div class="glass-card">
            <div class="brand-panel">
                <div>
                    <div class="company-badge">
                        <i class="fas fa-chart-line logo-icon"></i>
                        <span class="company-name">STOCKSEER AI</span>
                    </div>
                    <div class="welcome-section">
                        <div class="nice-to-see">
                            <i class="fas fa-smile-wink" style="margin-right: 6px;"></i> Nice to see you again
                        </div>
                        <div class="welcome-back">
                            WELCOME <br> BACK
                        </div>
                        </div>
                    </div>
                </div>
            </div>
            
           
                """, unsafe_allow_html=True)
    
    # Input fields
    st.text_input("Email ID / Username", key="username", placeholder="admin / rizkysyahp05@gmail.com")
    st.text_input("Password", type="password", key="password", placeholder="••••••")
    
    # Keep me signed in and member link
    col1, col2 = st.columns([2, 1])
    with col1:
        st.checkbox("Keep me signed in", key="keep_signed")
    with col2:
        st.markdown('<p style="text-align: right; margin-top: 8px;"><a href="#" style="color: #00B4FF; text-decoration: none; font-size: 0.8rem;">Already a member? <i class="fas fa-arrow-right"></i></a></p>', unsafe_allow_html=True)
    
    # Login button
    st.button("SUBSCRIBE", on_click=login, use_container_width=True)
    

# ⛔ STOP kalau belum login
if not st.session_state.logged_in:
    landing_page()
    st.stop()

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS  (dark-terminal aesthetic)
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

:root {
    --bg:        #0A0D14;
    --surface:   #111520;
    --border:    #1E2535;
    --accent:    #00F5A0;
    --accent2:   #00B4FF;
    --danger:    #FF4C6A;
    --warn:      #FFB830;
    --text:      #E2E8F0;
    --muted:     #64748B;
    --radius:    12px;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace !important; }
[data-testid="stMetricDelta"] svg { display:none; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* Selectbox / inputs */
.stSelectbox > div > div,
.stTextInput > div > div,
.stNumberInput > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Signal badges */
.badge {
    display:inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    letter-spacing: .05em;
}
.badge-buy  { background: rgba(0,245,160,.15); color: #00F5A0; border: 1px solid #00F5A0; }
.badge-sell { background: rgba(255,76,106,.15); color: #FF4C6A; border: 1px solid #FF4C6A; }
.badge-hold { background: rgba(255,184,48,.15);  color: #FFB830; border: 1px solid #FFB830; }

/* Header */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00F5A0, #00B4FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Progress bar */
.stProgress > div > div { background: var(--accent) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

/* Info / warning boxes */
.stAlert { border-radius: var(--radius) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Plotly transparent bg */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ░░  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download OHLCV data via yfinance."""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Ticker '{ticker}' tidak ditemukan atau tidak ada data.")
    df.dropna(inplace=True)
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ~30+ technical indicators using the `ta` library."""
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Trend ──────────────────────────────────────────────
    df["SMA_10"]  = ta.trend.sma_indicator(close, 10)
    df["SMA_20"]  = ta.trend.sma_indicator(close, 20)
    df["SMA_50"]  = ta.trend.sma_indicator(close, 50)
    df["EMA_10"]  = ta.trend.ema_indicator(close, 10)
    df["EMA_20"]  = ta.trend.ema_indicator(close, 20)
    df["EMA_50"]  = ta.trend.ema_indicator(close, 50)
    macd = ta.trend.MACD(close)
    df["MACD"]       = macd.macd()
    df["MACD_Signal"]= macd.macd_signal()
    df["MACD_Diff"]  = macd.macd_diff()
    adx = ta.trend.ADXIndicator(high, low, close)
    df["ADX"]    = adx.adx()
    df["ADX_pos"]= adx.adx_pos()
    df["ADX_neg"]= adx.adx_neg()
    df["CCI"]    = ta.trend.cci(high, low, close, 20)
    df["TRIX"]   = ta.trend.trix(close)

    # ── Momentum ────────────────────────────────────────────
    df["RSI"]    = ta.momentum.rsi(close, 14)
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["Stoch_K"]= stoch.stoch()
    df["Stoch_D"]= stoch.stoch_signal()
    df["Williams"]= ta.momentum.williams_r(high, low, close)
    df["ROC"]    = ta.momentum.roc(close)
    df["MOM"]    = ta.momentum.awesome_oscillator(high, low)

    # ── Volatility ──────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, 20, 2)
    df["BB_Upper"]= bb.bollinger_hband()
    df["BB_Mid"]  = bb.bollinger_mavg()
    df["BB_Lower"]= bb.bollinger_lband()
    df["BB_Width"]= bb.bollinger_wband()
    df["BB_Pct"]  = bb.bollinger_pband()
    df["ATR"]     = ta.volatility.average_true_range(high, low, close)
    df["Keltner_H"]= ta.volatility.keltner_channel_hband(high, low, close)
    df["Keltner_L"]= ta.volatility.keltner_channel_lband(high, low, close)

    # ── Volume ──────────────────────────────────────────────
    df["OBV"]    = ta.volume.on_balance_volume(close, vol)
    df["VWAP"]   = ta.volume.volume_weighted_average_price(high, low, close, vol)
    df["MFI"]    = ta.volume.money_flow_index(high, low, close, vol)
    df["CMF"]    = ta.volume.chaikin_money_flow(high, low, close, vol)
    df["FI"]     = ta.volume.force_index(close, vol)

    # ── Price-derived features ──────────────────────────────
    df["Return_1"] = close.pct_change(1)
    df["Return_3"] = close.pct_change(3)
    df["Return_5"] = close.pct_change(5)
    df["HL_Ratio"] = (high - low) / close
    df["OC_Ratio"] = (df["Open"] - close) / close

    df.dropna(inplace=True)
    return df


def create_labels(df: pd.DataFrame, horizon: int = 1,
                  threshold: float = 0.005) -> pd.Series:
    """
    Generate 3-class label for LightGBM:
      2 = BUY  (future return > +threshold)
      0 = SELL (future return < -threshold)
      1 = HOLD (otherwise)
    """
    future_ret = df["Close"].shift(-horizon) / df["Close"] - 1
    labels = np.where(future_ret > threshold, 2,
             np.where(future_ret < -threshold, 0, 1))
    return pd.Series(labels, index=df.index, name="Label")


def build_lstm_model(seq_len: int, n_features: int,
                     units: int = 64, dropout: float = 0.2) -> tf.keras.Model:
    """Bidirectional LSTM for next-price regression."""
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        Bidirectional(LSTM(units, return_sequences=True)),
        BatchNormalization(),
        Dropout(dropout),
        Bidirectional(LSTM(units // 2, return_sequences=False)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(1e-3), loss="huber",
                  metrics=["mae"])
    return model


def prepare_lstm_data(df: pd.DataFrame, feature_cols: list,
                      target_col: str, seq_len: int,
                      train_ratio: float = 0.8):
    """Scale → sequences → train/test split."""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_raw = scaler_X.fit_transform(df[feature_cols].values)
    y_raw = scaler_y.fit_transform(df[[target_col]].values)

    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_raw)):
        X_seq.append(X_raw[i - seq_len : i])
        y_seq.append(y_raw[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    split = int(len(X_seq) * train_ratio)
    return (X_seq[:split], X_seq[split:],
            y_seq[:split], y_seq[split:],
            scaler_X, scaler_y)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return dict(MAE=mae, RMSE=rmse, R2=r2, MAPE=mape)


SIGNAL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
SIGNAL_COLOR = {"BUY": "#00F5A0", "SELL": "#FF4C6A", "HOLD": "#FFB830"}


# ═══════════════════════════════════════════════════════════════════
# ░░  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════

def plot_candlestick(df: pd.DataFrame, ticker: str,
                     signals: pd.Series = None) -> go.Figure:
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20],
                        vertical_spacing=0.03)

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing_fillcolor="#00F5A0", decreasing_fillcolor="#FF4C6A",
        increasing_line_color="#00F5A0", decreasing_line_color="#FF4C6A",
    ), row=1, col=1)

    # MAs
    for col, color in [("SMA_20","#00B4FF"),("EMA_50","#FFB830")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col],
                name=col, line=dict(color=color, width=1.2, dash="dot"),
                opacity=.8), row=1, col=1)

    # BB
    if "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"],
            name="BB Upper", line=dict(color="#8B5CF6", width=.8), opacity=.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"],
            name="BB Lower", line=dict(color="#8B5CF6", width=.8),
            fill="tonexty", fillcolor="rgba(139,92,246,.06)", opacity=.5), row=1, col=1)

    # Buy/Sell markers
    if signals is not None:
        buy  = df[signals == "BUY"]
        sell = df[signals == "SELL"]
        fig.add_trace(go.Scatter(x=buy.index,  y=buy["Low"]  * .995,
            mode="markers", marker=dict(symbol="triangle-up",   size=9, color="#00F5A0"),
            name="Buy"), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell.index, y=sell["High"] * 1.005,
            mode="markers", marker=dict(symbol="triangle-down", size=9, color="#FF4C6A"),
            name="Sell"), row=1, col=1)

    # MACD
    if "MACD" in df.columns:
        colors_macd = ["#00F5A0" if v >= 0 else "#FF4C6A"
                       for v in df["MACD_Diff"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Diff"],
            marker_color=colors_macd, name="MACD Hist", opacity=.7), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
            line=dict(color="#00B4FF", width=1.2), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"],
            line=dict(color="#FF4C6A", width=1.2), name="Signal"), row=2, col=1)

    # Volume
    vol_colors = ["#00F5A0" if df["Close"].iloc[i] >= df["Open"].iloc[i]
                  else "#FF4C6A" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
        marker_color=vol_colors, name="Volume", opacity=.7), row=3, col=1)

    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b> — Price Chart",
                   font=dict(family="Space Mono", size=16, color="#E2E8F0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,21,32,1)",
        font=dict(color="#E2E8F0"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
        height=640,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor="#1E2535", row=i, col=1)
        fig.update_yaxes(gridcolor="#1E2535", row=i, col=1)
    return fig


def plot_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
        line=dict(color="#00B4FF", width=1.5), name="RSI"))
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,76,106,.08)",
                  line_width=0, annotation_text="Overbought",
                  annotation_font_color="#FF4C6A")
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,245,160,.08)",
                  line_width=0, annotation_text="Oversold",
                  annotation_font_color="#00F5A0")
    fig.add_hline(y=70, line_dash="dash", line_color="#FF4C6A", opacity=.5)
    fig.add_hline(y=30, line_dash="dash", line_color="#00F5A0", opacity=.5)
    fig.update_layout(
        title="RSI (14)", height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,21,32,1)",
        font=dict(color="#E2E8F0"),
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0,100], gridcolor="#1E2535"),
        xaxis=dict(gridcolor="#1E2535"),
    )
    return fig


def plot_prediction(dates_train, y_train_actual, y_train_pred,
                    dates_test, y_test_actual, y_test_pred,
                    dates_future=None, y_future=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_train, y=y_train_actual,
        name="Train Actual", line=dict(color="#64748B", width=1.2)))
    fig.add_trace(go.Scatter(x=dates_train, y=y_train_pred,
        name="Train Pred", line=dict(color="#00B4FF", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=dates_test, y=y_test_actual,
        name="Test Actual", line=dict(color="#E2E8F0", width=1.5)))
    fig.add_trace(go.Scatter(x=dates_test, y=y_test_pred,
        name="Test Pred", line=dict(color="#00F5A0", width=2)))

    if dates_future is not None:
        fig.add_trace(go.Scatter(
            x=list(dates_test[-3:]) + list(dates_future),
            y=list(y_test_pred[-3:]) + list(y_future),
            name="Forecast", mode="lines",
            line=dict(color="#FFB830", width=2, dash="dash"),
        ))

    fig.update_layout(
        title="<b>Price Prediction — Hybrid LSTM+LightGBM</b>",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,21,32,1)",
        font=dict(color="#E2E8F0", family="DM Sans"),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
        height=420,
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(gridcolor="#1E2535"),
        yaxis=dict(gridcolor="#1E2535"),
    )
    return fig


def plot_feature_importance(model_lgb, feature_names: list) -> go.Figure:
    importance = model_lgb.feature_importances_
    idx = np.argsort(importance)[-20:]
    fig = go.Figure(go.Bar(
        x=importance[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        marker=dict(
            color=importance[idx],
            colorscale=[[0,"#1E2535"],[0.5,"#00B4FF"],[1,"#00F5A0"]],
        ),
    ))
    fig.update_layout(
        title="Top 20 Feature Importance (LightGBM)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,21,32,1)",
        font=dict(color="#E2E8F0"),
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(gridcolor="#1E2535"),
        yaxis=dict(gridcolor="#1E2535"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# ░░  SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<p class="hero-title">📈 StockSeer AI</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748B;font-size:.8rem">Hybrid LSTM + LightGBM</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Konfigurasi Data")
    ticker = st.text_input("Ticker Symbol", value="BBCA.JK",
                           help="Contoh: BBCA.JK, TLKM.JK, AAPL, TSLA, BTC-USD")

    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Periode", ["1y","2y","3y","5y"], index=1)
    with col2:
        interval = st.selectbox("Interval", ["1d","1wk"], index=0)

    st.divider()
    st.markdown("### 🧠 Parameter Model")
    seq_len = st.slider("Sequence Length (LSTM)", 10, 60, 30)
    lstm_units = st.slider("LSTM Units", 32, 256, 64, step=32)
    lstm_epochs = st.slider("Max Epochs (LSTM)", 20, 200, 50, step=10)
    dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.2, step=0.05)
    forecast_days = st.slider("Forecast Days", 1, 30, 7)

    st.divider()
    st.markdown("### 🎯 Sinyal Trading")
    signal_threshold = st.slider("Threshold Return (%)", 0.1, 2.0, 0.5, step=0.1) / 100
    signal_horizon    = st.selectbox("Horizon Label (hari)", [1, 3, 5, 10], index=1)

    st.divider()
    run_button = st.button("🚀  Jalankan Analisis", use_container_width=True)

    st.markdown("""
    <div style="margin-top:2rem;padding:12px;background:rgba(0,245,160,.05);
                border:1px solid rgba(0,245,160,.2);border-radius:8px;font-size:.75rem;
                color:#64748B;line-height:1.6">
    ⚠️ <strong style="color:#FFB830">Disclaimer</strong><br>
    Ini adalah alat riset. Bukan saran investasi. Selalu lakukan due diligence.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ░░  MAIN APP
# ═══════════════════════════════════════════════════════════════════

st.markdown('<p class="hero-title">📈 Stock Prediction & Trading Signal</p>',
            unsafe_allow_html=True)
st.markdown('<p style="color:#64748B;margin-top:-12px">Model Hybrid LSTM + LightGBM  |  Indikator Teknikal Lengkap</p>',
            unsafe_allow_html=True)
st.divider()

if not run_button:
    # Landing info
    cols = st.columns(3)
    for c, icon, title, desc in zip(
        cols,
        ["🤖","📊","⚡"],
        ["Model Hybrid","30+ Indikator","Sinyal Real-time"],
        [
            "Menggabungkan kekuatan Bidirectional LSTM untuk time-series dan LightGBM untuk klasifikasi sinyal.",
            "RSI, MACD, Bollinger Bands, ADX, Stochastic, OBV, MFI, ATR, VWAP, CCI, dan banyak lagi.",
            "BUY / SELL / HOLD berdasarkan ensemble prediksi harga + klasifikasi return masa depan."
        ]
    ):
        c.markdown(f"""
        <div style="background:#111520;border:1px solid #1E2535;border-radius:12px;
                    padding:20px;height:140px;">
            <div style="font-size:1.8rem">{icon}</div>
            <div style="font-weight:700;margin:8px 0 4px">{title}</div>
            <div style="color:#64748B;font-size:.85rem">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.info("👈  Atur parameter di sidebar, lalu klik **Jalankan Analisis**.")
    st.stop()


# ─────────────────────────────────────────────────────────────────
# STEP 1 — Load data
# ─────────────────────────────────────────────────────────────────
with st.status(f"📥 Mengunduh data **{ticker}** …", expanded=True) as status:
    try:
        df_raw = fetch_data(ticker, period, interval)
        st.write(f"✅ {len(df_raw)} baris  |  {df_raw.index[0].date()} → {df_raw.index[-1].date()}")

        df = add_technical_indicators(df_raw)
        st.write(f"✅ {df.shape[1]} fitur teknikal ditambahkan")

        labels = create_labels(df, horizon=signal_horizon, threshold=signal_threshold)
        df["Label"] = labels
        df.dropna(inplace=True)
        st.write(f"✅ Label sinyal dibuat (horizon={signal_horizon}d, threshold={signal_threshold*100:.1f}%)")
        status.update(label="Data siap!", state="complete")
    except Exception as e:
        st.error(f"❌ Gagal mengunduh data: {e}")
        st.stop()

# Quick stats row
info = yf.Ticker(ticker).fast_info
c1, c2, c3, c4, c5 = st.columns(5)
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2])
delta_pct   = (last_close - prev_close) / prev_close * 100

c1.metric("Harga Terakhir",  f"{last_close:,.2f}", f"{delta_pct:+.2f}%")
c2.metric("RSI (14)",        f"{df['RSI'].iloc[-1]:.1f}")
c3.metric("MACD",            f"{df['MACD'].iloc[-1]:.4f}")
c4.metric("ATR",             f"{df['ATR'].iloc[-1]:.4f}")
c5.metric("Volume",          f"{int(df['Volume'].iloc[-1]):,}")

st.divider()

# ─────────────────────────────────────────────────────────────────
# STEP 2 — TABS Layout
# ─────────────────────────────────────────────────────────────────
tab_chart, tab_model, tab_signal, tab_indicator, tab_backtest = st.tabs([
    "📊 Chart",
    "🤖 Model & Prediksi",
    "⚡ Sinyal Trading",
    "📐 Indikator",
    "🧪 Backtest",
])

# ══════════════════════════
# TAB 1 — CHART
# ══════════════════════════
with tab_chart:
    st.plotly_chart(plot_candlestick(df, ticker), use_container_width=True)
    st.plotly_chart(plot_rsi(df), use_container_width=True)


# ══════════════════════════
# TAB 2 — MODEL
# ══════════════════════════
with tab_model:
    st.markdown("### 🧠 Pelatihan Model Hybrid")

    # Feature columns
    exclude = ["Label","Open","High","Low","Volume","Close"]
    feat_cols = [c for c in df.columns if c not in exclude]
    target_col = "Close"

    # ── LSTM Training ────────────────────────────────────────
    with st.expander("🔧 Bidirectional LSTM — Regresi Harga", expanded=True):
        prog_bar = st.progress(0, text="Mempersiapkan data LSTM …")

        (X_tr, X_te, y_tr, y_te,
         scaler_X, scaler_y) = prepare_lstm_data(
            df, feat_cols, target_col, seq_len)

        prog_bar.progress(20, "Membangun arsitektur LSTM …")
        lstm_model = build_lstm_model(seq_len, len(feat_cols),
                                      units=lstm_units, dropout=dropout_rate)

        prog_bar.progress(30, "Melatih LSTM …")
        es = EarlyStopping(patience=10, restore_best_weights=True)
        rlr= ReduceLROnPlateau(factor=.5, patience=5, verbose=0)

        hist = lstm_model.fit(
            X_tr, y_tr,
            validation_split=0.1,
            epochs=lstm_epochs,
            batch_size=32,
            callbacks=[es, rlr],
            verbose=0,
        )
        prog_bar.progress(80, "Evaluasi LSTM …")

        # Predictions
        y_tr_pred = scaler_y.inverse_transform(lstm_model.predict(X_tr, verbose=0))
        y_te_pred = scaler_y.inverse_transform(lstm_model.predict(X_te, verbose=0))
        y_tr_act  = scaler_y.inverse_transform(y_tr)
        y_te_act  = scaler_y.inverse_transform(y_te)

        metrics_lstm = evaluate_model(y_te_act.flatten(), y_te_pred.flatten())
        prog_bar.progress(100, "LSTM selesai ✅")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE",  f"{metrics_lstm['MAE']:,.4f}")
        c2.metric("RMSE", f"{metrics_lstm['RMSE']:,.4f}")
        c3.metric("R²",   f"{metrics_lstm['R2']:.4f}")
        c4.metric("MAPE", f"{metrics_lstm['MAPE']:.2f}%")

        # Date indices for plot
        df_no_first = df.iloc[seq_len:]
        split_idx   = int(len(df_no_first) * 0.8)
        dates_train = df_no_first.index[:split_idx]
        dates_test  = df_no_first.index[split_idx:]

        # Forecast future
        last_seq = scaler_X.transform(df[feat_cols].values)[-seq_len:]
        future_preds = []
        cur_seq = last_seq.copy()
        for _ in range(forecast_days):
            p = lstm_model.predict(cur_seq[np.newaxis], verbose=0)
            future_preds.append(float(scaler_y.inverse_transform(p)[0,0]))
            # roll window (naive: repeat last row with predicted close substituted)
            new_row = cur_seq[-1].copy()
            cur_seq = np.vstack([cur_seq[1:], new_row])

        future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1),
                                      periods=forecast_days)

        st.plotly_chart(
            plot_prediction(dates_train, y_tr_act.flatten(), y_tr_pred.flatten(),
                            dates_test,  y_te_act.flatten(), y_te_pred.flatten(),
                            future_dates, future_preds),
            use_container_width=True,
        )

        # Loss curve
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=hist.history["loss"],    name="Train Loss", line=dict(color="#00B4FF")))
        fig_loss.add_trace(go.Scatter(y=hist.history["val_loss"],name="Val Loss",   line=dict(color="#FF4C6A")))
        fig_loss.update_layout(title="Training Loss", height=250,
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(17,21,32,1)",
                               font=dict(color="#E2E8F0"),
                               margin=dict(l=0,r=0,t=40,b=0),
                               xaxis=dict(gridcolor="#1E2535"),
                               yaxis=dict(gridcolor="#1E2535"))
        st.plotly_chart(fig_loss, use_container_width=True)

    # ── LightGBM Training ────────────────────────────────────
    with st.expander("🌿 LightGBM — Klasifikasi Sinyal", expanded=True):
        prog_lgb = st.progress(0, text="Mempersiapkan data LightGBM …")

        df_lgb = df[feat_cols + ["Label"]].dropna()
        X_lgb  = df_lgb[feat_cols].values
        y_lgb  = df_lgb["Label"].values.astype(int)

        split_lgb  = int(len(X_lgb) * 0.8)
        X_tr_l, X_te_l = X_lgb[:split_lgb], X_lgb[split_lgb:]
        y_tr_l, y_te_l = y_lgb[:split_lgb], y_lgb[split_lgb:]

        prog_lgb.progress(20, "Melatih LightGBM …")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr_l, y_tr_l,
            eval_set=[(X_te_l, y_te_l)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        prog_lgb.progress(90, "Evaluasi …")

        y_pred_lgb = lgb_model.predict(X_te_l)
        acc = np.mean(y_pred_lgb == y_te_l)

        from sklearn.metrics import classification_report
        report = classification_report(
            y_te_l, y_pred_lgb,
            target_names=["SELL","HOLD","BUY"], output_dict=True)

        prog_lgb.progress(100, "LightGBM selesai ✅")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",  f"{acc*100:.2f}%")
        c2.metric("F1 BUY",   f"{report['BUY']['f1-score']*100:.2f}%")
        c3.metric("F1 SELL",  f"{report['SELL']['f1-score']*100:.2f}%")

        st.plotly_chart(plot_feature_importance(lgb_model, feat_cols),
                        use_container_width=True)


# ══════════════════════════
# TAB 3 — SIGNAL
# ══════════════════════════
with tab_signal:
    st.markdown("### ⚡ Sinyal Trading Terkini")

    try:
        # Generate signals for full dataset
        df_sig = df[feat_cols].dropna()
        all_preds = lgb_model.predict(df_sig.values)
        all_proba = lgb_model.predict_proba(df_sig.values)

        df_signal = df.loc[df_sig.index].copy()
        df_signal["Signal"]   = [SIGNAL_MAP[p] for p in all_preds]
        df_signal["Prob_BUY"]  = all_proba[:, 2]
        df_signal["Prob_HOLD"] = all_proba[:, 1]
        df_signal["Prob_SELL"] = all_proba[:, 0]

        # Latest signal
        latest = df_signal.iloc[-1]
        sig_now = latest["Signal"]
        badge_cls = f"badge-{sig_now.lower()}"

        st.markdown(
            f"""
            <div style="background:#111520;border:1px solid #1E2535;border-radius:12px;
                        padding:24px 32px;display:flex;align-items:center;gap:24px;margin-bottom:20px">
                <div>
                    <div style="color:#64748B;font-size:.8rem;margin-bottom:4px">SINYAL TERBARU</div>
                    <span class="badge {badge_cls}">{sig_now}</span>
                </div>
                <div>
                    <div style="color:#64748B;font-size:.8rem;margin-bottom:4px">TANGGAL</div>
                    <div style="font-family:'Space Mono',monospace">{latest.name.strftime('%Y-%m-%d')}</div>
                </div>
                <div>
                    <div style="color:#64748B;font-size:.8rem;margin-bottom:4px">HARGA</div>
                    <div style="font-family:'Space Mono',monospace">{latest['Close']:,.2f}</div>
                </div>
                <div>
                    <div style="color:#64748B;font-size:.8rem;margin-bottom:4px">PROB. BUY</div>
                    <div style="font-family:'Space Mono',monospace;color:#00F5A0">{latest['Prob_BUY']*100:.1f}%</div>
                </div>
                <div>
                    <div style="color:#64748B;font-size:.8rem;margin-bottom:4px">PROB. SELL</div>
                    <div style="font-family:'Space Mono',monospace;color:#FF4C6A">{latest['Prob_SELL']*100:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability bar chart
        fig_prob = go.Figure(go.Bar(
            x=["SELL","HOLD","BUY"],
            y=[latest["Prob_SELL"], latest["Prob_HOLD"], latest["Prob_BUY"]],
            marker_color=["#FF4C6A","#FFB830","#00F5A0"],
            text=[f"{v*100:.1f}%" for v in
                  [latest["Prob_SELL"], latest["Prob_HOLD"], latest["Prob_BUY"]]],
            textposition="outside",
        ))
        fig_prob.update_layout(
            title="Probabilitas Sinyal Terakhir",
            height=300, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,21,32,1)",
            font=dict(color="#E2E8F0"),
            yaxis=dict(range=[0,1], gridcolor="#1E2535"),
            xaxis=dict(gridcolor="#1E2535"),
            margin=dict(l=0,r=0,t=40,b=0),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # Chart with signals
        st.plotly_chart(
            plot_candlestick(df_signal.tail(120), ticker,
                             signals=df_signal["Signal"].tail(120)),
            use_container_width=True
        )

        # Signal history table
        st.markdown("#### 📋 Riwayat 30 Sinyal Terakhir")
        recent_sig = df_signal[["Close","Signal","Prob_BUY","Prob_HOLD","Prob_SELL"]].tail(30)[::-1]
        recent_sig.index = recent_sig.index.strftime("%Y-%m-%d")
        recent_sig = recent_sig.rename(columns={
            "Close":"Harga","Signal":"Sinyal",
            "Prob_BUY":"P(BUY)","Prob_HOLD":"P(HOLD)","Prob_SELL":"P(SELL)"
        })
        recent_sig["P(BUY)"]  = (recent_sig["P(BUY)"]  * 100).round(1).astype(str) + "%"
        recent_sig["P(HOLD)"] = (recent_sig["P(HOLD)"] * 100).round(1).astype(str) + "%"
        recent_sig["P(SELL)"] = (recent_sig["P(SELL)"] * 100).round(1).astype(str) + "%"
        st.dataframe(recent_sig, use_container_width=True)

    except Exception as e:
        st.warning(f"Jalankan tab Model terlebih dahulu. ({e})")


# ══════════════════════════
# TAB 4 — INDICATORS
# ══════════════════════════
with tab_indicator:
    st.markdown("### 📐 Ringkasan Indikator Teknikal")

    last = df.iloc[-1]

    def ind_card(name, value, status=""):
        color = "#00F5A0" if status=="bullish" else ("#FF4C6A" if status=="bearish" else "#FFB830")
        return f"""
        <div style="background:#111520;border:1px solid #1E2535;border-radius:8px;
                    padding:12px 16px;margin-bottom:8px">
            <div style="color:#64748B;font-size:.75rem">{name}</div>
            <div style="font-family:'Space Mono',monospace;font-size:1rem;color:{color}">{value}</div>
        </div>"""

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Trend**")
        rsi_val = last["RSI"]
        rsi_status = "bearish" if rsi_val > 70 else ("bullish" if rsi_val < 30 else "neutral")
        st.markdown(ind_card("RSI (14)", f"{rsi_val:.2f}", rsi_status), unsafe_allow_html=True)
        macd_status = "bullish" if last["MACD"] > last["MACD_Signal"] else "bearish"
        st.markdown(ind_card("MACD", f"{last['MACD']:.4f}", macd_status), unsafe_allow_html=True)
        adx_status = "bullish" if last["ADX_pos"] > last["ADX_neg"] else "bearish"
        st.markdown(ind_card("ADX", f"{last['ADX']:.2f}", adx_status), unsafe_allow_html=True)
        st.markdown(ind_card("SMA 20", f"{last['SMA_20']:,.2f}",
                             "bullish" if last["Close"]>last["SMA_20"] else "bearish"),
                   unsafe_allow_html=True)
        st.markdown(ind_card("EMA 50", f"{last['EMA_50']:,.2f}",
                             "bullish" if last["Close"]>last["EMA_50"] else "bearish"),
                   unsafe_allow_html=True)

    with c2:
        st.markdown("**Momentum & Volatilitas**")
        stoch_status = "bearish" if last["Stoch_K"] > 80 else ("bullish" if last["Stoch_K"] < 20 else "neutral")
        st.markdown(ind_card("Stoch %K", f"{last['Stoch_K']:.2f}", stoch_status), unsafe_allow_html=True)
        st.markdown(ind_card("ATR", f"{last['ATR']:.4f}", "neutral"), unsafe_allow_html=True)
        bb_status = "bearish" if last["BB_Pct"] > 1 else ("bullish" if last["BB_Pct"] < 0 else "neutral")
        st.markdown(ind_card("BB %B", f"{last['BB_Pct']:.3f}", bb_status), unsafe_allow_html=True)
        st.markdown(ind_card("BB Width", f"{last['BB_Width']:.4f}", "neutral"), unsafe_allow_html=True)
        st.markdown(ind_card("CCI (20)", f"{last['CCI']:.2f}",
                             "bearish" if last["CCI"]>100 else ("bullish" if last["CCI"]<-100 else "neutral")),
                   unsafe_allow_html=True)

    with c3:
        st.markdown("**Volume**")
        st.markdown(ind_card("OBV",   f"{last['OBV']:,.0f}",  "neutral"), unsafe_allow_html=True)
        st.markdown(ind_card("VWAP",  f"{last['VWAP']:,.2f}", "bullish" if last["Close"]>last["VWAP"] else "bearish"),
                   unsafe_allow_html=True)
        st.markdown(ind_card("MFI",   f"{last['MFI']:.2f}",
                             "bearish" if last["MFI"]>80 else ("bullish" if last["MFI"]<20 else "neutral")),
                   unsafe_allow_html=True)
        st.markdown(ind_card("CMF",   f"{last['CMF']:.4f}", "bullish" if last["CMF"]>0 else "bearish"),
                   unsafe_allow_html=True)
        st.markdown(ind_card("Williams %R", f"{last['Williams']:.2f}",
                             "bearish" if last["Williams"]>-20 else ("bullish" if last["Williams"]<-80 else "neutral")),
                   unsafe_allow_html=True)

    # Multi-indicator chart
    sel_ind = st.multiselect(
        "Pilih indikator untuk di-plot",
        options=[c for c in df.columns if c not in ["Open","High","Low","Close","Volume","Label"]],
        default=["RSI","MACD","ATR","OBV"],
    )
    if sel_ind:
        fig_multi = make_subplots(rows=len(sel_ind), cols=1,
                                  shared_xaxes=True, vertical_spacing=0.04)
        colors = ["#00F5A0","#00B4FF","#FFB830","#FF4C6A","#8B5CF6","#EC4899"]
        for i, ind in enumerate(sel_ind):
            fig_multi.add_trace(
                go.Scatter(x=df.index, y=df[ind], name=ind,
                           line=dict(color=colors[i % len(colors)], width=1.4)),
                row=i+1, col=1,
            )
            fig_multi.update_yaxes(title_text=ind, row=i+1, col=1,
                                   gridcolor="#1E2535", title_font_size=10)
            fig_multi.update_xaxes(gridcolor="#1E2535", row=i+1, col=1)
        fig_multi.update_layout(
            height=220 * len(sel_ind),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,21,32,1)",
            font=dict(color="#E2E8F0"),
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_multi, use_container_width=True)


# ══════════════════════════
# TAB 5 — BACKTEST
# ══════════════════════════
with tab_backtest:
    st.markdown("### 🧪 Backtest Strategi Sederhana")
    st.info("Strategi: Masuk (Long) saat sinyal BUY, keluar saat SELL atau HOLD. Tidak ada short-selling.")

    try:
        initial_capital = st.number_input("Modal Awal (IDR/USD)", value=10_000_000,
                                          step=1_000_000, min_value=100_000)
        df_bt = df_signal[["Close","Signal"]].copy()

        capital   = float(initial_capital)
        position  = 0.0
        portfolio = []
        trades    = []

        for i, (date, row) in enumerate(df_bt.iterrows()):
            price = float(row["Close"])
            sig   = row["Signal"]

            if sig == "BUY" and position == 0:
                shares   = capital / price
                position = shares
                capital  = 0.0
                trades.append(dict(Date=date, Type="BUY", Price=price, Shares=shares))

            elif sig in ("SELL", "HOLD") and position > 0 and sig == "SELL":
                capital  = position * price
                trades.append(dict(Date=date, Type="SELL", Price=price, Shares=position,
                                   PnL=capital - initial_capital))
                position = 0.0

            total_value = capital + position * price
            portfolio.append(dict(Date=date, Value=total_value))

        df_port = pd.DataFrame(portfolio).set_index("Date")

        # Buy & Hold comparison
        df_bah = (df_bt["Close"] / df_bt["Close"].iloc[0]) * initial_capital

        # Metrics
        final_val    = df_port["Value"].iloc[-1]
        total_ret    = (final_val / initial_capital - 1) * 100
        bah_ret      = (df_bah.iloc[-1] / initial_capital - 1) * 100
        max_dd_val   = ((df_port["Value"] - df_port["Value"].cummax()) / df_port["Value"].cummax()).min() * 100
        n_trades     = len([t for t in trades if t["Type"] == "BUY"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return",       f"{total_ret:+.2f}%")
        c2.metric("Buy & Hold Return",  f"{bah_ret:+.2f}%")
        c3.metric("Max Drawdown",       f"{max_dd_val:.2f}%")
        c4.metric("Jumlah Trade",       n_trades)

        # Portfolio curve
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=df_port.index, y=df_port["Value"],
            name="Strategi Sinyal", line=dict(color="#00F5A0", width=2)))
        fig_bt.add_trace(go.Scatter(x=df_bah.index, y=df_bah,
            name="Buy & Hold", line=dict(color="#64748B", width=1.5, dash="dot")))
        fig_bt.update_layout(
            title="Kurva Portofolio",
            height=360, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,21,32,1)",
            font=dict(color="#E2E8F0"),
            legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=50,b=0),
            xaxis=dict(gridcolor="#1E2535"),
            yaxis=dict(gridcolor="#1E2535"),
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        if trades:
            df_trades = pd.DataFrame(trades)
            df_trades["Date"] = df_trades["Date"].dt.strftime("%Y-%m-%d")
            st.markdown("#### 📋 Log Trade")
            st.dataframe(df_trades, use_container_width=True)

    except Exception as e:
        st.warning(f"Jalankan tab Model & Sinyal terlebih dahulu. ({e})")

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#1E2535;font-size:.75rem;padding:8px 0">
    StockSeer AI  •  Hybrid LSTM + LightGBM  •  Built with Streamlit  •  ⚠️ Bukan saran investasi
</div>
""", unsafe_allow_html=True)