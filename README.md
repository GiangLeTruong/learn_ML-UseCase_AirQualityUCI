# learn_ML-UseCase_AirQualityUCI
air-quality-uci-prediction/
│
├── data/
│   ├── raw/                      # Dữ liệu gốc AirQualityUCI.csv (chứa giá trị -200)
│   └── processed/                # Dữ liệu sạch đã qua xử lý, interpolate và tạo lag features
│
├── notebooks/                    # Nơi chứa các Jupyter Notebooks nghiên cứu
│   ├── 01_eda_and_cleaning.ipynb # Làm sạch -200, vẽ đồ thị tương quan
│   ├── 02_feature_engineering.ipynb # Trích xuất lag features, rolling statistics
│   └── 03_model_experiments.ipynb   # Huấn luyện và so sánh Baseline vs Advanced
│
├── src/                          # Mã nguồn chính (Modular Python Code)
│   ├── data_loader.py            # Script load dữ liệu gốc và định dạng datetime
│   ├── preprocessing.py         # Hàm xử lý null, -200, interpolation
│   ├── features.py               # Hàm tạo các tính năng lag, rolling, datetime
│   ├── models/                   # Lớp chứa định nghĩa và huấn luyện mô hình
│   │   ├── baselines.py          # Code chạy Linear Regression, ARIMA, Random Forest
│   │   └── advanced.py           # Code chạy XGBoost / LSTM
│   ├── evaluate.py               # Các hàm tính MAE, RMSE, R2 và vẽ đồ thị dự đoán
│   └── utils.py                  # Các hàm phụ trợ (save/load model, config)
│
├── models/                       # Thư mục lưu các mô hình đã huấn luyện (.pkl, .joblib)
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost_model.pkl
│
├── reports/                      # Kết quả xuất ra báo cáo
│   └── figures/                  # Biểu đồ EDA, đồ thị so sánh thực tế vs dự đoán
│       ├── correlation_matrix.png
│       └── prediction_comparison.png
│
├── config.yaml                   # File lưu đường dẫn, thông số huấn luyện (learning rate, window size...)
├── requirements.txt              # Danh sách thư viện (pandas, numpy, scikit-learn, xgboost, matplotlib...)
├── .gitignore                    # Bỏ qua các file rác, bytecode Python, file data nặng
└── README.md                     # File mô tả tổng quan và hướng dẫn chạy đồ án

# Giải thích tên các column trong bộ dữ liệu:
- Các chất ô nhiễm không khí
CO(GT): Nồng độ khí Cacbon Monoxit (Carbon Monoxide).
NMHC(GT): Tổng nồng độ các Hydrocacbon phi metan (Non-Methane Hydrocarbons).
C6H6(GT): Nồng độ khí Benzen (Benzene).
NOx(GT): Nồng độ Các oxit nitơ (Nitrogen Oxides).
NO2(GT): Nồng độ khí Nitơ Dioxit (Nitrogen Dioxide).
* GT: True Ground Truth = Dữ liệu thực tế chuẩn xác

- Các biến thời tiết
T (Temperature): Nhiệt độ môi trường (đơn vị: độ C).
RH (Relative Humidity): Độ ẩm tương đối (đơn vị: %).
AH (Absolute Humidity): Độ ẩm tuyệt đối (khối lượng hơi nước trong 1 mét khối không khí).

- Nhóm cảm biến PT08.Sx
PT08.S1(CO): Cảm biến hướng tới đo khí CO.
PT08.S2(NMHC): Cảm biến hướng tới đo Hydrocacbon.
PT08.S3(NOx): Cảm biến hướng tới đo Oxit Nitơ.
PT08.S4(NO2): Cảm biến hướng tới đo Nitơ Dioxit.
PT08.S5(O3): Cảm biến hướng tới đo khí Ozon (O_3).

------------------------------------------------------------------------------------------------
# Mục tiêu khoa học của tập dữ liệu AirQualityUCI
Ý tưởng: Liệu chúng ta có thể dùng thuật toán Machine Learning để biến các cảm biến PT08 giá rẻ (kết hợp với Nhiệt độ T và Độ ẩm RH) thành một chiếc máy đo chính xác tương đương với thiết bị GT đắt tiền được không?
=> Nếu mô hình dự đoán thành công, người ta có thể chế tạo các máy đo chất lượng không khí mini, giá thành rẻ để lắp đặt đại trà ở khắp mọi nơi trong thành phố, thay vì chỉ có một vài trạm quan trắc lớn đắt tiền.