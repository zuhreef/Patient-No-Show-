import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_PATH = Path("models/model.pkl")
FEATURES_PATH = Path("artifacts/feature_columns.json")

app = FastAPI(title="Patient No-Show Prediction API")


class PredictionRequest(BaseModel):
    Gender: int
    Age: int
    Scholarship: int
    Hipertension: int
    Diabetes: int
    Alcoholism: int
    Handcap: int
    SMS_received: int
    lead_days: float
    appointment_weekday: int


def load_model_and_features():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run training first.")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Feature columns file not found. Run training first.")

    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r") as f:
        feature_cols = json.load(f)

    return model, feature_cols


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        model, feature_cols = load_model_and_features()

        input_data = request.model_dump()
        df = pd.DataFrame([input_data])

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_cols]

        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        return {
            "prediction": prediction,
            "probability_no_show": probability,
            "model_version": "v1"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))