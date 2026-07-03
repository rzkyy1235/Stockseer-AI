# ============================================================
# APLIKASI WEB PREDIKSI HARGA SAHAM & SINYAL TRADING
# Model Hybrid LSTM-LightGBM + Indikator Teknikal
# Framework: Streamlit
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 1. KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="Prediksi Saham Hybrid LSTM-LightGBM",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Aplikasi Prediksi Harga Saham dan Sinyal Trading")
st.markdown("**Model Hybrid LSTM-LightGBM + Indikator Teknikal**")
st.markdown("Aplikasi ini memprediksi harga penutupan saham **besok** dan memberikan sinyal **BELI**, **JUAL**, atau **TAHAN**.")

# ==============================
# 2. FUNGSI INDIKATOR TEKNIKAL
# ==============================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# ==============================
# 3. FUNGSI PREPROCESSING & MODEL (CACHED)
# ==============================
@st.cache_resource(ttl=3600)
def load_or_train_model(ticker, start_date, end_date):
    """
    Load data, hitung indikator, latih model hybrid LSTM-LightGBM.
    Return: model LSTM (feature extractor), model LightGBM, scaler_X, scaler_y, df_model, features list, sequence length
    """
    # Ambil data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return None, None, None, None, None, None, None
    
    df = df[['Close']].copy()
    df.columns = ['Close']
    
    # Hitung indikator teknikal
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_%B'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['RSI'] = compute_rsi(df['Close'])
    macd_line, macd_signal = compute_macd(df['Close'])
    df['MACD_line'] = macd_line
    df['MACD_signal'] = macd_signal
    
    features = ['Close', 'EMA_20', 'BB_%B', 'RSI', 'MACD_line', 'MACD_signal']
    df_model = df[features].dropna()
    if len(df_model) < 100:
        return None, None, None, None, None, None, None
    
    # Siapkan data untuk supervised learning (X = fitur hari t, y = Close hari t+1)
    X = df_model[features].values[:-1]
    y = df_model['Close'].values[1:]
    
    # Normalisasi
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Split data (80% training, 20% testing) - urutan waktu
    split = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]
    
    # Untuk LSTM, perlu sequence 3D (samples, timesteps, features)
    SEQ_LEN = 60
    # Fungsi membuat sequence dari data skala
    def create_sequences(data, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(data)):
            X_seq.append(data[i-seq_len:i])
            y_seq.append(data[i, 0])  # asumsi kolom pertama adalah target (Close)
        return np.array(X_seq), np.array(y_seq)
    
    # Untuk training LSTM, kita gunakan data training yang sudah discale
    # Kita perlu menyusun ulang X_train dan y_train dalam bentuk sequence
    # Karena X_train adalah (samples, features), kita perlu mengubahnya kembali ke bentuk time series?
    # Pendekatan hybrid: ekstrak fitur temporal dengan LSTM dari sequence harga + indikator.
    # Alternatif: gunakan data urutan asli dengan window 60.
    # Di sini kita akan gunakan seluruh data scaled untuk membuat sequence.
    # Tapi karena split dilakukan setelah sequence, kita harus berhati-hati.
    # Solusi: buat sequence dari seluruh X_scaled, lalu split.
    X_seq_all, y_seq_all = create_sequences(X_scaled, SEQ_LEN)
    split_seq = int(0.8 * len(X_seq_all))
    X_train_seq, X_test_seq = X_seq_all[:split_seq], X_seq_all[split_seq:]
    y_train_seq, y_test_seq = y_seq_all[:split_seq], y_seq_all[split_seq:]
    
    # Bangun LSTM untuk ekstraksi fitur temporal
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, X_scaled.shape[1])))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50, return_sequences=False))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))  # output regresi (bisa diabaikan nanti)
    lstm_model.compile(optimizer='adam', loss='mse')
    
    # Latih LSTM
    lstm_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0)
    
    # Ekstraktor fitur (buang layer Dense)
    feature_extractor = Sequential(lstm_model.layers[:-1])  # tanpa output layer
    # Ekstrak fitur dari data training dan testing
    X_train_features = feature_extractor.predict(X_train_seq, verbose=0)
    X_test_features = feature_extractor.predict(X_test_seq, verbose=0)
    # Target tetap y_train_seq dan y_test_seq (harga besok)
    
    # Latih LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
    lgb_model.fit(X_train_features, y_train_seq)
    
    # Prediksi pada data uji
    y_pred_scaled = lgb_model.predict(X_test_features)
    # Kembalikan ke skala asli
    # Karena y disimpan dalam bentuk scaled berdasarkan scaler_y yang di-fit ke y asli (harga close)
    # Namun y_seq_all adalah target dari sequence (harga besok dalam skala). Kita perlu invers.
    # Cara mudah: gunakan scaler_y yang sudah di-fit pada y asli dari awal (y adalah harga asli setelah normalisasi)
    # Karena y_seq_all adalah bagian dari y_scaled (yang di-shift), kita bisa gunakan scaler_y.
    # Kita ambil y_test_seq (yang sudah discale) lalu inverse.
    y_test_actual_scaled = y_test_seq
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_actual = scaler_y.inverse_transform(y_test_actual_scaled.reshape(-1, 1)).ravel()
    
    # Evaluasi
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    
    # Simpan juga data hasil untuk keperluan sinyal (ambil tanggal yang sesuai)
    # Tanggal untuk data test sequence dimulai setelah SEQ_LEN + split_seq
    test_start_idx = SEQ_LEN + split_seq
    test_dates = df_model.index[test_start_idx : test_start_idx + len(y_test_actual)]
    
    return (feature_extractor, lgb_model, scaler_X, scaler_y, df_model, features, 
            SEQ_LEN, X_scaled, y_scaled, split, test_dates, y_test_actual, y_pred_actual, rmse, mae)

# ==============================
# 4. SIDEBAR: PILIH SAHAM & PARAMETER
# ==============================
st.sidebar.header("⚙️ Pengaturan")

# Daftar saham (default)
default_tickers = ["BBRI.JK", "BBCA.JK", "TLKM.JK", "ASII.JK", "UNVR.JK", "ADRO.JK"]
selected_ticker = st.sidebar.selectbox("Pilih Kode Saham", default_tickers)

start_date = st.sidebar.date_input("Tanggal Mulai", value=pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("Tanggal Akhir", value=pd.to_datetime("2024-01-01"))

if st.sidebar.button("🚀 Prediksi Sekarang", type="primary"):
    with st.spinner(f"Memproses {selected_ticker}... (mengunduh data, melatih model hybrid LSTM-LightGBM)"):
        result = load_or_train_model(selected_ticker, start_date, end_date)
        if result[0] is None:
            st.error("Data tidak cukup untuk diprediksi. Coba perpanjang periode atau pilih saham lain.")
        else:
            (feature_extractor, lgb_model, scaler_X, scaler_y, df_model, features,
             SEQ_LEN, X_scaled, y_scaled, split, test_dates, y_actual, y_pred, rmse, mae) = result
            
            # Tampilkan metrik
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rmse:.2f}")
            col2.metric("MAE", f"{mae:.2f}")
            col3.metric("Total Data Uji", len(y_actual))
            
            # Buat DataFrame sinyal
            results_df = pd.DataFrame({
                'Tanggal': test_dates,
                'Harga_Sekarang': y_actual,
                'Prediksi_Besok': y_pred
            })
            
            # Tentukan sinyal (threshold 0.5%)
            threshold = 0.005
            signals = []
            for i in range(len(results_df)):
                if i == 0:
                    signals.append('TAHAN')
                else:
                    harga_sekarang = results_df.iloc[i-1]['Harga_Sekarang']
                    pred_esok = results_df.iloc[i]['Prediksi_Besok']
                    if pred_esok > harga_sekarang * (1 + threshold):
                        signals.append('BELI')
                    elif pred_esok < harga_sekarang * (1 - threshold):
                        signals.append('JUAL')
                    else:
                        signals.append('TAHAN')
            results_df['Sinyal'] = signals
            
            st.subheader(f"📊 Rekomendasi Trading Terbaru untuk {selected_ticker}")
            st.dataframe(results_df.tail(10), use_container_width=True)
            
            # Grafik
            st.subheader("📈 Grafik Harga Aktual vs Prediksi")
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(results_df['Harga_Sekarang'].values, label='Harga Aktual', color='blue', linewidth=1.5)
            ax.plot(results_df['Prediksi_Besok'].values, label='Prediksi Hybrid LSTM-LightGBM', color='red', linestyle='--', linewidth=1.5)
            
            # Marker sinyal (hanya 100 titik terakhir)
            start_plot = max(0, len(results_df) - 100)
            for i in range(start_plot, len(results_df)-1):
                if i > 0:
                    harga_sekarang = results_df.iloc[i-1]['Harga_Sekarang']
                    pred_esok = results_df.iloc[i]['Prediksi_Besok']
                    if pred_esok > harga_sekarang * 1.005:
                        ax.scatter(i, results_df.iloc[i]['Harga_Sekarang'], marker='^', color='green', s=100, zorder=5, label='BELI' if i==start_plot else "")
                    elif pred_esok < harga_sekarang * 0.995:
                        ax.scatter(i, results_df.iloc[i]['Harga_Sekarang'], marker='v', color='red', s=100, zorder=5, label='JUAL' if i==start_plot else "")
            ax.set_xlabel('Hari ke- (Data Uji)')
            ax.set_ylabel('Harga Penutupan')
            ax.set_title(f'Prediksi Harga Saham {selected_ticker}')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            # Tombol download CSV
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Detail Sinyal (CSV)",
                data=csv,
                file_name=f"detail_sinyal_{selected_ticker.replace('.', '_')}.csv",
                mime="text/csv",
            )
else:
    st.info("👈 Silakan pilih saham di menu samping, lalu klik 'Prediksi Sekarang'.")

st.sidebar.markdown("---")
st.sidebar.caption("Dibangun dengan Streamlit | Model Hybrid LSTM-LightGBM")