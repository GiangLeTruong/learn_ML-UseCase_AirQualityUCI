import pandas as pd

def load_raw_data(filepath):
    """
    Load dữ liệu AirQualityUCI từ file CSV và chuẩn hóa cột Datetime.
    """
    print(f"Đang đọc dữ liệu từ: {filepath} ...")
    
    df = pd.read_csv(filepath, parse_dates=False)
    
    df.dropna(how='all', inplace=True)      
    df.dropna(axis=1, how='all', inplace=True) 
    
    datetime_series = df['Date'] + ' ' + df['Time']
    df['Datetime'] = pd.to_datetime(datetime_series, format='mixed', dayfirst=True)
    
    df.set_index('Datetime', inplace=True)
    
    df.drop(columns=['Date', 'Time'], inplace=True)
    
    print(f"Dữ liệu đã load thành công! Kích thước: {df.shape}")
    return df