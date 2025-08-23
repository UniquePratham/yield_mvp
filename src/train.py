import argparse, os, json, joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from utils import load_and_align, split_features_target


def main(args):
    df = load_and_align(args.data)
    X, y, cat_cols, num_cols = split_features_target(df)

    if y is None:
        raise ValueError("Dataset must contain a 'Yield' column for training.")

    # --- Preprocess: numeric + categorical ---
    num_pipeline = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
        ("sc", StandardScaler())
    ])
    cat_pipeline = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop"
    )

    # --- Model ---
    model = LGBMRegressor(
        n_estimators=2000,        # huge, but early stopping will cut it
        learning_rate=0.01,
        max_depth=6,               # unlimited depth, use num_leaves to control
        num_leaves=256,             # bigger trees (default 31 is too small)
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        device="gpu",               # <<< THIS is key
        gpu_platform_id=0,
        gpu_device_id=0,
        random_state=42,
        n_jobs=-1
    )
        
    if args.model == "xgb" and HAS_XGB:
        model = XGBRegressor(
            n_estimators=20000,         # high, let early stopping decide
            learning_rate=0.01,
            max_depth=12,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="gpu_hist",      # <<< GPU
            predictor="gpu_predictor",   # <<< GPU
            gpu_id=0,
            random_state=42,
            verbosity=1
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    # --- Train/val split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)

    # --- Evaluation ---
    preds = pipe.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_val, preds))
    r2 = float(r2_score(y_val, preds))

    # --- Save artifacts ---
    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.outdir, "model.pkl"))
    meta = {
        "features": list(X.columns),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2}
    }
    with open(os.path.join(args.outdir, "feature_list.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to {args.outdir}. Metrics => RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to training CSV")
    ap.add_argument("--outdir", default="models", help="Output directory")
    ap.add_argument("--n_estimators", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--model", choices=["lgbm", "xgb"], default="lgbm")
    args = ap.parse_args()
    main(args)
