import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# ============================
# Kaggle setup
# ============================
api = KaggleApi()
api.authenticate()

# Datasets to pull (you can add/remove more)
DATASETS = {
   "crop-yield-prediction": "samuelotiattakorah/agriculture-crop-yield",
}

RAW_DIR = "./raw_data"
os.makedirs(RAW_DIR, exist_ok=True)

# ============================
# Download datasets
# ============================
for name, dataset in DATASETS.items():
    print(f"Downloading {name} -> {dataset}")
    api.dataset_download_files(dataset, path=RAW_DIR, unzip=True)

# ============================
# Processing functions
# ============================
def process_crop_data():
    path = os.path.join(RAW_DIR, "crop_yield.csv")  # adjust based on dataset
    df = pd.read_csv(path)
    df = df.rename(columns={
        "State_Name": "region",
        "Crop": "crop",
        "Crop_Year": "year",
        "Production": "yield"
    })
    return df[["region", "crop", "year", "yield"]]

def process_weather_data():
    path = os.path.join(RAW_DIR, "crop_yield.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Region": "region",
        "Year": "year",
        "AvgTemperature": "tavg"
    })
    # Fake min/max
    df["tmin"] = df["tavg"] - 3
    df["tmax"] = df["tavg"] + 3
    df["rainfall"] = df["tavg"].apply(lambda x: max(0, 100 - x))  # placeholder
    return df[["region", "year", "tmin", "tmax", "rainfall"]]

def process_soil_data():
    path = os.path.join(RAW_DIR, "crop_yield.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={
        "N": "soil_n",
        "P": "soil_p",
        "K": "soil_k",
        "ph": "soil_ph",
        "label": "crop"
    })
    df["region"] = "Unknown"
    df["year"] = 2020
    return df[["region", "crop", "year", "soil_n", "soil_p", "soil_k", "soil_ph"]]

def process_ndvi_data():
    path = os.path.join(RAW_DIR, "crop_yield.csv")  # placeholder, replace with NDVI dataset
    df = pd.read_csv(path)
    df["ndvi"] = (df["N"] + df["P"] + df["K"]) / 300  # mock NDVI calc
    df["region"] = "Unknown"
    df["crop"] = df["label"]
    df["year"] = 2020
    return df[["region", "crop", "year", "ndvi"]]

# ============================
# Merge into target schema
# ============================
def merge_datasets():
    crop_df = process_crop_data()
    weather_df = process_weather_data()
    soil_df = process_soil_data()
    ndvi_df = process_ndvi_data()

    # Merge step by step
    df = pd.merge(crop_df, weather_df, on=["region", "year"], how="left")
    df = pd.merge(df, soil_df, on=["region", "crop", "year"], how="left")
    df = pd.merge(df, ndvi_df, on=["region", "crop", "year"], how="left")

    # Final target schema
    df_final = df[[
        "region", "crop", "year", "rainfall",
        "tmin", "tmax", "ndvi",
        "soil_ph", "soil_n", "soil_p", "soil_k", "yield"
    ]]

    df_final.to_csv("train.csv", index=False)
    print("✅ train.csv generated successfully!")

if __name__ == "__main__":
    merge_datasets()
