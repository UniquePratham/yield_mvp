import pandas as pd
import numpy as np

# Canonical columns based on NEW schema
CANON_COLS = [
    "Crop", "Crop_Year", "Season", "State",
    "Area", "Production", "Annual_Rainfall",
    "Fertilizer", "Pesticide", "Yield"
]

def load_and_align(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize column names (case-insensitive match)
    lower_cols = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=lower_cols, inplace=True)

    # Map schema aliases (if dataset uses different names)
    aliases = {
        "crop": "Crop",
        "crop_year": "Crop_Year",
        "year": "Crop_Year",
        "season": "Season",
        "state": "State",
        "area": "Area",
        "production": "Production",
        "annual_rainfall": "Annual_Rainfall",
        "rainfall": "Annual_Rainfall",
        "fertilizer": "Fertilizer",
        "fertilizer_used": "Fertilizer",
        "pesticide": "Pesticide",
        "pesticide_used": "Pesticide",
        "yield": "Yield",
        "yield_tons_per_hectare": "Yield"
    }

    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Keep only schema columns
    keep = [c for c in CANON_COLS if c in df.columns]
    df = df[keep]

    # Convert numeric columns
    num_cols = ["Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with missing target
    if "Yield" in df.columns:
        df = df.dropna(subset=["Yield"])

    return df

def split_features_target(df: pd.DataFrame):
    y = df["Yield"] if "Yield" in df.columns else None
    X = df.drop(columns=["Yield"], errors="ignore")

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    return X, y, cat_cols, num_cols
