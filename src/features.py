import pandas as pd

def add_datetime_features(df):
    """
    Trích xuất các đặc trưng thời gian từ index Datetime.
    Giúp mô hình nhận biết được chu kỳ ngày/đêm và các mùa.
    """
    print("Create Datetime Features ...")
    df_feat = df.copy()
    
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek  # Monday = 0, Sunday = 6
    df_feat['month'] = df_feat.index.month
    
    df_feat['is_weekend'] = (df_feat.index.dayofweek >= 5).astype(int)
    return df_feat

def create_lag_features(df, columns, lags=3):
    """
    Tạo Lag Features cho các cột được chỉ định.
    Ví dụ: lag=1 nghĩa là lấy giá trị của 1 giờ trước đó.
    """
    print(f"Đang tạo {lags} Lag Features cho các cột: {columns}...")
    df_feat = df.copy()
    
    for col in columns:
        for i in range(1, lags + 1):
            df_feat[f'{col}_lag_{i}'] = df_feat[col].shift(i)
            
    return df_feat

def create_rolling_features(df, columns, window=6):
    """
    Tạo đặc trưng trung bình trượt (Rolling Statistics) để làm mượt dữ liệu
    và bắt được xu hướng chung trong 'window' giờ vừa qua.
    """
    print(f"Đang tạo Rolling Features (mean, window={window}) cho các cột: {columns}...")
    df_feat = df.copy()
    
    for col in columns:
        df_feat[f'{col}_rolling_mean_{window}'] = df_feat[col].shift(1).rolling(window=window).mean()
        
    return df_feat