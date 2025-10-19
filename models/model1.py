import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def train_regression_model(
    file,
    target_column,
    test_size=0.2,
    epochs=50,
    layers=None,
    dropout=0.2,
    normalize=True,
    progress_callback=None,
    stop_callback=None
):
    if layers is None:
        layers = [(128, "relu"), (64, "relu")]

    # --- 1️⃣ Завантаження даних ---
    if file.endswith(".csv"):
        data = pd.read_csv(file)
    else:
        data = pd.read_excel(file)

    if target_column not in data.columns:
        raise ValueError(f"Колонка '{target_column}' не знайдена у файлі.")

    # --- 2️⃣ Обмеження кількості рядків для стабільності ---
    if len(data) > 10000:
        data = data.sample(10000, random_state=42)
        print("⚠️ Використовується 10 000 рядків із великого датасету для стабільності.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # --- 3️⃣ Обробка колонок ---
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessors = []
    if categorical_cols:
        preprocessors.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))
    if numeric_cols:
        if normalize:
            preprocessors.append(("num", StandardScaler(), numeric_cols))
        else:
            preprocessors.append(("num", "passthrough", numeric_cols))

    ct = ColumnTransformer(preprocessors)
    X_processed = ct.fit_transform(X)

    # --- 4️⃣ Приведення типів ---
    X_processed = X_processed.astype(np.float32)
    y = y.astype(np.float32).values

    # --- 5️⃣ Поділ ---
    x_train, x_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, random_state=42)

    # --- 6️⃣ tf.data.Dataset ---
    batch_size = 32
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # --- 7️⃣ Побудова моделі ---
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(X_processed.shape[1],))])
    for neurons, activation in layers:
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # --- 8️⃣ Прогрес бар ---
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback.emit(int((epoch + 1) / epochs * 100))
            if stop_callback and stop_callback():
                self.model.stop_training = True

    # --- 9️⃣ Навчання ---
    history = model.fit(
        train_ds,
        epochs=epochs,
        verbose=0,
        callbacks=[ProgressCallback()]
    )

    # --- 🔟 Оцінка ---
    loss, mae = model.evaluate(test_ds, verbose=0)

    print(f"✅ Модель навчена. MAE = {mae:.4f}")
    return model, mae, history, ct, X.columns
