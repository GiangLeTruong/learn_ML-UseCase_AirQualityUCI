import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Tính toán và in ra các chỉ số đánh giá mô hình hồi quy.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- Kết quả cho {model_name} ---")
    print(f"MAE  (Sai số tuyệt đối trung bình) : {mae:.4f}")
    print(f"RMSE (Sai số toàn phương trung bình): {rmse:.4f}")
    print(f"R2   (Hệ số xác định)             : {r2:.4f}\n")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def plot_predictions(y_true, y_pred, index, title="Thực tế vs Dự đoán", save_name=None):
    """
    Vẽ đồ thị đường so sánh giữa giá trị thực tế và giá trị dự đoán.
    Chỉ nên đưa vào một khoảng thời gian ngắn (vd: 200 giờ) để dễ nhìn.
    """
    plt.figure(figsize=(15, 6))
    
    plt.plot(index, y_true, label='Thực tế (True)', color='blue', linewidth=1.5)
    plt.plot(index, y_pred, label='Dự đoán (Pred)', color='red', linestyle='--', linewidth=1.5)
    
    plt.title(title, fontsize=16)
    plt.xlabel('Thời gian', fontsize=12)
    plt.ylabel('Nồng độ', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_name:
        os.makedirs('../reports/figures', exist_ok=True)
        save_path = f"../reports/figures/{save_name}"
        plt.savefig(save_path, dpi=300)
        print(f"Đã lưu đồ thị tại: {save_path}")
        
    plt.show()