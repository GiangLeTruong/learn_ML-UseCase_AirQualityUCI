from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_predict_arima(train_series, test_series, order=(5, 1, 0)):
    """
    ARIMA là mô hình thống kê chỉ nhận dữ liệu 1 chiều (Univariate) mà không cần X.
    Hàm này train và predict trả về array dự đoán.
    """
    warnings.filterwarnings("ignore")
    print(f"Đang huấn luyện ARIMA với cấu hình order={order}...")
    
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    
    predictions = model_fit.forecast(steps=len(test_series))
    return predictions.values