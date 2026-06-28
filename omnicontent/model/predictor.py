import numpy as np
import pickle
import keras

from omnicontent.config import MODEL_PATH, get_logger
log = get_logger("model.predictor")

SCALER_PATH = "omnicontent/model/scaler.pkl"

_model = None
_scaler_data = None


def _load():
    global _model, _scaler_data
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            _scaler_data = pickle.load(f)
        log.info("Model and scaler loaded.")
    return _model, _scaler_data


def predict_viral_score(keyword: str) -> float:
    model, scaler_data = _load()

    word_count = len(keyword.split())
    hashtag_count = max(1, word_count)
    keyword_count = word_count

    raw_features = np.array([[
        0.3,
        0.05,
        hashtag_count,
        keyword_count,
        len(keyword) * 8,
        1,
    ]])

    X_scaled = scaler_data["X"].transform(raw_features)
    y_scaled = model.predict(X_scaled, verbose=0)
    viral_score = scaler_data["y"].inverse_transform(y_scaled)[0][0]

    viral_score = float(np.clip(viral_score, 0, 1))
    log.info(f"Keyword='{keyword}' -> viral_score={viral_score:.3f}")
    return viral_score


def score_script_quality(script: dict) -> dict:
    issues = []
    score = 1.0

    voiceover = script.get("voiceover", "")
    scenes = script.get("scenes", [])

    if len(voiceover) > 500:
        issues.append("Voiceover exceeds 500 characters.")
        score -= 0.3

    if len(voiceover) < 50:
        issues.append("Voiceover is too short, insufficient content.")
        score -= 0.2

    if len(scenes) != 3:
        issues.append(f"Scene count should be 3, found {len(scenes)}.")
        score -= 0.3

    total_duration = sum(s.get("duration", 0) for s in scenes)
    if abs(total_duration - 30) > 5:
        issues.append(f"Total scene duration differs significantly from 30s: {total_duration}s")
        score -= 0.2

    score = max(0.0, score)
    return {"quality_score": round(score, 2), "issues": issues}
