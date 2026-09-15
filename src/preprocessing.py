import numpy as np
import pandas as pd

def replace_nulls(df, null_value=-200):
    """
    Thay thế các giá trị báo lỗi (mặc định -200) thành NaN.
    """
    print(f"Đang thay thế giá trị {null_value} thành NaN...")
    
    df_cleaned = df.copy()
    df_cleaned.replace(null_value, np.nan, inplace=True)
    
    missing_count = df_cleaned.isna().sum().sum()
    print(f"Tổng số giá trị khuyết (NaN) sau khi chuyển đổi: {missing_count}")
    
    return df_cleaned

def impute_missing_values(df):
    """
    Điền các giá trị NaN bằng phương pháp nội suy theo thời gian (time interpolation).
    """
    print("Đang tiến hành nội suy dữ liệu khuyết...")
    df_imputed = df.interpolate(method='time')
    
    df_imputed = df_imputed.bfill()
    
    print("Đã hoàn tất điền dữ liệu khuyết.")
    return df_imputed