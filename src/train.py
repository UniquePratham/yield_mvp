import os
import argparse
import joblib
import pandas as pd
import numpy as np 
import lightgbm as lgb
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

def load_and_clean(csv_path: str):
    df = pd.read_csv(csv_path)

    # Ensure correct column names
    expected_target = "Yield"
    if expected_target not in df.columns:
        raise ValueError(f"Dataset must contain '{expected_target}' column. Found: {df.columns.tolist()}")

    # Features = everything except target
    X = df.drop(columns=[expected_target])
    y = df[expected_target]

    # Convert categoricals → category dtype
    for col in ["Crop", "Season", "State"]:
        if col in X.columns:
            X[col] = X[col].astype("category")

    return X, y

def main(args):
    print("📂 Loading dataset...")
    X, y = load_and_clean(args.data)

    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # LightGBM Dataset
    lgb_train = lgb.Dataset(
        X_train, y_train, categorical_feature=["Crop", "Season", "State"]
    )
    lgb_val = lgb.Dataset(
        X_val, y_val, reference=lgb_train, categorical_feature=["Crop", "Season", "State"]
    )

    # Params
    params = {
        "objective": "poisson",
        "metric": "rmse",
        "n_estimators": 10000,
        "learning_rate": 0.01,
        "max_depth": 2,
        "num_leaves": 256,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "device": "gpu" if args.gpu else "cpu"
    }

    # Custom callback to update tqdm progress bar
    pbar = tqdm(total=params["n_estimators"], desc="🚀 Training Progress")

    def tqdm_callback(env):
        pbar.update(1)
        if env.iteration == params["n_estimators"] - 1:
            pbar.close()

    print("⚡ Training LightGBM model...")
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100), tqdm_callback]
    )

    # Predictions
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = np.maximum(y_pred, 0)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    r2 = r2_score(y_val, y_pred)

    print(f"✅ Training complete. RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Save model + metadata
    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, "crop_yield_model.pkl")
    feature_list_path = os.path.join(args.outdir, "feature_list.json")
    categories_path = os.path.join(args.outdir, "categories.json")

    joblib.dump(model, model_path)

    # Save features
    with open(feature_list_path, "w") as f:
        json.dump({"features": list(X.columns)}, f, indent=4)

    # Save categorical levels
    categories = {}
    for col in ["Crop", "Season", "State"]:
        if col in X.columns and pd.api.types.is_categorical_dtype(X[col]):
            categories[col] = X[col].cat.categories.tolist()

    with open(categories_path, "w") as f:
        json.dump(categories, f, indent=4)

    print(f"📦 Model saved to: {model_path}")
    print(f"📝 Feature list saved to: {feature_list_path}")
    print(f"📝 Categories saved to: {categories_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save model")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()
    main(args)
