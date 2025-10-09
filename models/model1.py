import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_iris_model(file="iris.xls", epochs=30, layers=None, dropout=0.2,
                     progress_callback=None, stop_callback=None):
    if layers is None:
        layers = [(128, "relu"), (64, "relu")]

    data = pd.read_excel(file, engine="xlrd")
    X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
    y = data["Species"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    X = X / X.max(axis=0)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(4,))])
    for neurons, activation in layers:
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(3))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback.emit(int((epoch + 1) / epochs * 100))
            if stop_callback and stop_callback():
                print("⏹ Training interrupted by user.")
                self.model.stop_training = True

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        verbose=0,
        callbacks=[ProgressCallback()]
    )

    # Якщо навчання зупинено — не рахуємо accuracy
    if stop_callback and stop_callback():
        return model, 0.0, history, encoder, X

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return model, acc, history, encoder, X
