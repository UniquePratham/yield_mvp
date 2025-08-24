import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS   # ✅ Enable CORS

# -----------------------------
# Load model and metadata
# -----------------------------
with open("models/crop_yield_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/categories.json", "r") as f:
    categories = json.load(f)

with open("models/feature_list.json", "r") as f:
    _fl = json.load(f)
    if isinstance(_fl, dict) and "features" in _fl:
        feature_list = _fl["features"]
    elif isinstance(_fl, list):
        feature_list = _fl
    else:
        feature_list = list(_fl)

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
CORS(app)  # ✅ allow all origins

# -----------------------------
# Helper functions
# -----------------------------
def _map_to_saved_category(val, saved_cats):
    """Match val with saved category (ignoring whitespace).
       If no match, return None."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    for cat in saved_cats:
        if str(cat).strip() == s:
            return cat
    if val in saved_cats:
        return val
    return None

def preprocess_input(data):
    """Convert JSON input to DataFrame compatible with the model."""
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Input must be a dict (single row) or list of dicts.")

    df = pd.DataFrame(data)

    # Handle categorical features
    for col, cats in categories.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _map_to_saved_category(v, cats))
            df[col] = pd.Categorical(df[col], categories=cats)
        else:
            df[col] = pd.Categorical([None] * len(df), categories=cats)

    # Ensure all features exist
    for feat in feature_list:
        if feat not in df.columns:
            df[feat] = 0

    df = df[feature_list]
    return df

# -----------------------------
# Routes
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json(force=True)
        df = preprocess_input(input_data)
        preds = model.predict(df)

        results = [{"Yield": float(p)} for p in preds]
        return jsonify(results if len(results) > 1 else results[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
