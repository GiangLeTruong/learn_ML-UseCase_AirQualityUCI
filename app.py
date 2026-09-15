import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import các hàm từ thư mục src của dự án
from src.data_loader import load_raw_data
from src.preprocessing import replace_nulls, impute_missing_values
from src.features import add_datetime_features, create_lag_features, create_rolling_features
from src.evaluate import calculate_metrics
from src.models.baselines import train_linear_regression, train_random_forest, train_predict_arima
from src.models.advanced import build_and_train_lstm

# Cấu hình trang Streamlit
st.set_page_config(page_title="Air Quality Prediction", layout="wide")
st.title("☁️ Ứng dụng Dự đoán Mức độ Ô nhiễm Không khí")
st.markdown("Dự án phân tích và dự báo nồng độ các chất ô nhiễm dựa trên tập dữ liệu **AirQualityUCI**.")

# ==========================================
# SIDEBAR: CẤU HÌNH THAM SỐ
# ==========================================
st.sidebar.header("⚙️ 1. Cấu hình Dữ liệu")

pollutant_map = {
    'CO (Carbon Monoxide)': ('CO(GT)', 'PT08.S1(CO)'),
    'NMHC (Non-methane Hydrocarbons)': ('NMHC(GT)', 'PT08.S2(NMHC)'),
    'NOx (Nitrogen Oxides)': ('NOx(GT)', 'PT08.S3(NOx)'),
    'NO2 (Nitrogen Dioxide)': ('NO2(GT)', 'PT08.S4(NO2)')
}

selected_pollutant = st.sidebar.selectbox("Chọn chất ô nhiễm mục tiêu (Target):", list(pollutant_map.keys()))
target_col, sensor_col = pollutant_map[selected_pollutant]

st.sidebar.markdown(f"**Biến mục tiêu:** `{target_col}`\n\n**Cảm biến chính:** `{sensor_col}`")

# Cho phép chọn thêm các biến thời tiết
weather_cols = st.sidebar.multiselect(
    "Chọn thêm biến thời tiết:", 
    ['T', 'RH', 'AH'], 
    default=['T', 'RH']
)

st.sidebar.header("🛠 2. Feature Engineering")
lags = st.sidebar.slider("Số giờ trễ (Lag Features):", min_value=1, max_value=12, value=3)
window = st.sidebar.slider("Khung giờ (Rolling Mean):", min_value=2, max_value=24, value=6)

st.sidebar.header("🤖 3. Chọn Mô hình")
use_lr = st.sidebar.checkbox("Linear Regression", value=True)
use_rf = st.sidebar.checkbox("Random Forest", value=True)
use_lstm = st.sidebar.checkbox("LSTM (Deep Learning)", value=False)
use_arima = st.sidebar.checkbox("ARIMA", value=False)

# ==========================================
# MAIN APP: XỬ LÝ & HUẤN LUYỆN
# ==========================================
if st.button("🚀 Chạy Huấn luyện & Đánh giá", type="primary"):
    
    # 1. LOAD VÀ XỬ LÝ DỮ LIỆU
    with st.spinner('Đang tải và làm sạch dữ liệu...'):
        df = load_raw_data('data/raw/AirQualityUCI.csv')
        df = replace_nulls(df)
        df = impute_missing_values(df)
        
        # Feature Engineering
        df = add_datetime_features(df)
        
        # Xác định các cột cần tạo feature
        feature_cols = [target_col, sensor_col] + weather_cols
        
        df = create_lag_features(df, columns=feature_cols, lags=lags)
        df = create_rolling_features(df, columns=feature_cols, window=window)
        df.dropna(inplace=True)
        
    st.success("Tải và chuẩn bị dữ liệu thành công!")
    with st.expander("Xem trước dữ liệu sau khi trích xuất đặc trưng"):
        st.dataframe(df.head())

    # 2. CHUẨN BỊ TẬP TRAIN/TEST
    with st.spinner('Đang phân chia và chuẩn hóa tập Train/Test...'):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # 3. HUẤN LUYỆN & DỰ ĐOÁN
    results = []
    predictions_dict = {}
    
    progress_bar = st.progress(0)
    
    if use_lr:
        st.write("⏳ Đang chạy Linear Regression...")
        lr_model = train_linear_regression(X_train_scaled, y_train)
        preds = lr_model.predict(X_test_scaled)
        metrics = calculate_metrics(y_test, preds, "Linear Regression")
        results.append({"Model": "Linear Regression", **metrics})
        predictions_dict["Linear Regression"] = preds
        
    if use_rf:
        st.write("⏳ Đang chạy Random Forest...")
        rf_model = train_random_forest(X_train_scaled, y_train)
        preds = rf_model.predict(X_test_scaled)
        metrics = calculate_metrics(y_test, preds, "Random Forest")
        results.append({"Model": "Random Forest", **metrics})
        predictions_dict["Random Forest"] = preds
        
    if use_lstm:
        st.write("⏳ Đang chạy LSTM (Có thể mất vài phút)...")
        # Reshape cho LSTM
        X_train_lstm = X_train_scaled.values.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.values.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        _, preds = build_and_train_lstm(X_train_lstm, y_train.values, X_test_lstm, epochs=10)
        metrics = calculate_metrics(y_test, preds, "LSTM")
        results.append({"Model": "LSTM", **metrics})
        predictions_dict["LSTM"] = preds
        
    if use_arima:
        st.write("⏳ Đang chạy ARIMA (Chỉ dùng Target history)...")
        preds = train_predict_arima(y_train, y_test, order=(3,1,1))
        metrics = calculate_metrics(y_test, preds, "ARIMA")
        results.append({"Model": "ARIMA", **metrics})
        predictions_dict["ARIMA"] = preds

    progress_bar.progress(100)
    
    # ==========================================
    # KẾT QUẢ & TRỰC QUAN HÓA
    # ==========================================
    if results:
        st.markdown("---")
        st.subheader("📊 Bảng So sánh Hiệu năng Các Mô hình")
        results_df = pd.DataFrame(results).set_index("Model")
        
        # Highlight giá trị tốt nhất (MAE, RMSE thấp nhất, R2 cao nhất)
        st.dataframe(results_df.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                                     .highlight_max(subset=['R2'], color='lightgreen'), 
                     use_container_width=True)
        
        st.subheader("📈 Đồ thị so sánh: Thực tế vs Dự đoán (200 giờ đầu của tập Test)")
        
        # Vẽ biểu đồ sử dụng matplotlib bọc trong Streamlit
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(y_test.index[:200], y_test.values[:200], label='Thực tế (Actual)', color='black', linewidth=2)
        
        colors = ['red', 'blue', 'green', 'orange']
        for idx, (model_name, preds) in enumerate(predictions_dict.items()):
            ax.plot(y_test.index[:200], preds[:200], label=f'Dự đoán ({model_name})', color=colors[idx % len(colors)], alpha=0.7)
            
        ax.set_title(f"Dự báo nồng độ {target_col}")
        ax.set_ylabel("Nồng độ")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    else:
        st.warning("Vui lòng chọn ít nhất một mô hình ở thanh bên trái!")