# AI Yield Prediction — 10‑Hour MVP

This repo trains a yield prediction model using tabular features (weather + soil) and optional NDVI.
- **Model**: LightGBM (fast, strong on tabular), with fallback to XGBoost.
- **Inputs**: CSV with columns (case-insensitive): `region,crop,year,rainfall,tmin,tmax,ndvi,soil_ph,soil_n,soil_p,soil_k,yield`.
- **Outputs**: `models/model.pkl` and `models/feature_list.json`.
- **Apps**: Streamlit UI and FastAPI for programmatic use.

## Quickstart
1. Install deps: `pip install -r requirements.txt`
2. Put your dataset at `data/train.csv` (or keep `data/sample_data.csv` to test the flow).
3. Train: `python src/train.py --data data/train.csv --outdir models`
4. Predict on CSV: `python src/infer.py --model models/model.pkl --data data/sample_data.csv --out predictions.csv`
5. Run UI: `streamlit run app/streamlit_app.py`
6. Run API: `uvicorn app.api:app --reload --port 8000`

## Dataset Notes
- Replace `data/sample_data.csv` with a real dataset. You can map/rename columns to match the expected schema.
- Missing `ndvi` is okay; the pipeline handles missing columns and NaNs.
- For quick NDVI time series export from Sentinel‑2, see the Earth Engine script in the main chat.