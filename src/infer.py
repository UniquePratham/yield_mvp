import argparse, joblib, pandas as pd
from utils import load_and_align

def main(args):
    model = joblib.load(args.model)
    df = load_and_align(args.data)
    X = df.drop(columns=['yield'], errors='ignore')
    preds = model.predict(X)
    out = df.copy()
    out['yield_pred'] = preds
    out.to_csv(args.out, index=False)
    print(f"Wrote predictions -> {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model.pkl")
    ap.add_argument("--data", required=True, help="CSV for inference (with or without yield column)")
    ap.add_argument("--out", default="predictions.csv", help="Output CSV path")
    args = ap.parse_args()
    main(args)