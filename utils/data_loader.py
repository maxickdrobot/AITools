import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data_and_encoder( file = "Iris.xls"):
    # завантажуємо дані
    data = pd.read_excel(file, engine="xlrd")  # xlrd не підтримує xlsx
    X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
    y = data["Species"].values
    encoder = LabelEncoder()
    encoder.fit(y)
    X = X / X.max(axis=0)  # нормалізація
    return X, encoder
