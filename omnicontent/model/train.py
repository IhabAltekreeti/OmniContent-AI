import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
import os, pickle

from omnicontent.config import MODEL_PATH, get_logger
log = get_logger("model.train")

DATA_PATH   = "/content/Social Media Engagement Dataset.csv"
SCALER_PATH = "omnicontent/model/scaler.pkl"


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)

    df["hashtag_count"] = df["hashtags"].fillna("").apply(lambda x: len(str(x).split(",")))
    df["keyword_count"] = df["keywords"].fillna("").apply(lambda x: len(str(x).split(",")))
    df["text_length"]   = df["text_content"].fillna("").apply(len)
    df["has_emotion"]   = df["emotion_type"].notna().astype(int)

    features = [
        "sentiment_score",
        "toxicity_score",
        "hashtag_count",
        "keyword_count",
        "text_length",
        "has_emotion",
    ]

    df = df[features + ["engagement_rate"]].dropna()

    X = df[features].values
    y = df["engagement_rate"].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({"X": scaler_X, "y": scaler_y, "features": features}, f)

    log.info(f"Data prepared: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="viral_mlp")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_and_save():
    X, y = load_and_prepare()
    log.info("Model training starting...")
    model = build_model(X.shape[1])
    history = model.fit(
        X, y,
        epochs=40,
        batch_size=64,
        validation_split=0.2,
        verbose=1,
    )
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    log.info(f"Model saved -> {MODEL_PATH}")
    return model, history


if __name__ == "__main__":
    train_and_save()
