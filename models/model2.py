import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def train_classification_model(
        file,
        target_column,
        epochs=30,
        layers=None,
        dropout=0.2,
        test_size=0.2,
        progress_callback=None,
        stop_callback=None
):

    # === 1. Load the dataset ===
    if file.endswith(".xls") or file.endswith(".xlsx"):
        data = pd.read_excel(file)
    elif file.endswith(".csv"):
        data = pd.read_csv(file)
    else:
        raise ValueError("Only .xls, .xlsx, and .csv formats are supported")

    feature_columns = [col for col in data.columns if col != target_column]
    X = data[feature_columns].values

    # === 2. Encoding labels ===
    y = data[target_column].values

    # Якщо це числова колонка, але кількість унікальних значень невелика — трактуємо як класи
    if np.issubdtype(data[target_column].dtype, np.number):
        unique_values = np.unique(y)
        if len(unique_values) > 20:
            raise ValueError(
                f"Target column '{target_column}' looks numeric with {len(unique_values)} unique values. "
                f"This is likely a regression task, not classification."
            )

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # === 3. Scale features ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === 4. Split into train/test sets ===
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=42
    )

    # === 5. Define model architecture ===
    if layers is None:
        layers = [(128, "relu"), (64, "relu")]

    num_classes = len(set(y_encoded))

    model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(X.shape[1],))])
    for neurons, activation in layers:
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    # === 6. Compile model ===
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # === 7. Progress callback ===
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback.emit(int((epoch + 1) / epochs * 100))
            if stop_callback and stop_callback():
                print("⏹ Training stopped by user.")
                self.model.stop_training = True

    # === 8. Train model ===
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        verbose=0,
        callbacks=[ProgressCallback()]
    )

    # === 9. Evaluate model ===
    if stop_callback and stop_callback():
        return model, 0.0, history, encoder, scaler, X

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return model, acc, history, encoder, scaler, X
