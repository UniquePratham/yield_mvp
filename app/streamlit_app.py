import streamlit as st
import pandas as pd
import joblib, os
from datetime import datetime
import numpy as np
import json

st.set_page_config(page_title="Yield Predictor", page_icon="🌾", layout="centered")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
FEATURE_META_PATH = os.getenv("FEATURE_META_PATH", "models/feature_list.json")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_meta():
    with open(FEATURE_META_PATH) as f:
        return json.load(f)

model = load_model()
meta = load_meta()
expected_features = meta["features"]

st.title("🌾 Crop Yield Predictor")

# ---------------- SINGLE PREDICTION ---------------- #
with st.expander("Single Prediction"):
    col1, col2 = st.columns(2)

    crop = col1.text_input("Crop", "Wheat")
    crop_year = col2.number_input("Crop Year", 2000, 2100, 2024)
    season = col1.text_input("Season", "Rabi")
    state = col2.text_input("State", "Punjab")
    area = col1.number_input("Area (hectares)", 0.0, 1e6, 1000.0)
    production = col2.number_input("Production (tons)", 0.0, 1e7, 2500.0)
    rainfall = col1.number_input("Annual Rainfall (mm)", 0.0, 2000.0, 500.0)
    fertilizer = col2.number_input("Fertilizer (kg/ha)", 0.0, 1000.0, 100.0)
    pesticide = col1.number_input("Pesticide (kg/ha)", 0.0, 1000.0, 10.0)

    if st.button("Predict Yield"):
        df = pd.DataFrame([{
            "Crop": crop,
            "Crop_Year": crop_year,
            "Season": season,
            "State": state,
            "Area": area,
            "Production": production,
            "Annual_Rainfall": rainfall,
            "Fertilizer": fertilizer,
            "Pesticide": pesticide
        }])

        # Ensure all expected columns exist
        for col in expected_features:
            if col not in df.columns:
                df[col] = np.nan
        df = df[expected_features]

        y = model.predict(df)[0]
        st.success(f"Predicted Yield: **{y:.2f}**")

st.divider()
st.subheader("Batch Predictions")
st.text('Schema Should Match the Following Columns:\n'
        'Crop,Crop_Year,Season,State,Area,Production,Annual_Rainfall,Fertilizer,Pesticide,Yield')

uploaded = st.file_uploader("Upload CSV with schema above", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)

    # Drop target if provided
    preds = model.predict(df.drop(columns=["Yield"], errors="ignore"))

    df_out = df.copy()
    df_out["yield_pred"] = preds
    st.dataframe(df_out.head(50))
    st.download_button("Download Predictions", df_out.to_csv(index=False), "predictions.csv")
