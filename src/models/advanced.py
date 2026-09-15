import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def build_and_train_lstm(X_train, y_train, X_test, epochs=20, batch_size=32):
    """
    Xây dựng và huấn luyện mô hình LSTM.
    Lưu ý: X_train, X_test phải được reshape về dạng 3D trước khi đưa vào hàm này.
    """

    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    print("Đang huấn luyện LSTM (Deep Learning)...")
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    
    y_pred = model.predict(X_test)
    return model, y_pred.flatten()