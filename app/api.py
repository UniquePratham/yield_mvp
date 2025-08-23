from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os
import pandas as pd
import numpy as np
import json

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
FEATURE_META_PATH = os.getenv("FEATURE_META_PATH", "models/feature_list.json")

model = joblib.load(MODEL_PATH)
with open(FEATURE_META_PATH) as f:
    meta = json.load(f)
expected_features = meta["features"]

class Payload(BaseModel):
    Crop: str
    Crop_Year: int
    Season: str
    State: str
    Area: float
    Production: float
    Annual_Rainfall: float
    Fertilizer: float
    Pesticide: float

app = FastAPI(title="Yield Predictor API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(p: Payload):
    df = pd.DataFrame([p.dict()])

    # Align columns
    for col in expected_features:
        if col not in df.columns:
            df[col] = np.nan
    df = df[expected_features]

    y = model.predict(df)[0]
    return {"yield_pred": float(y)}
