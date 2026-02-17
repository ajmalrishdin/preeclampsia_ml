#!/usr/bin/env python
# coding: utf-8
"""
inference.py – Load a trained model + scaler and predict on new data.

Usage examples
--------------
# Predict a single outcome with one model:
  python inference.py \
      --model  output/models/preeclampsia_onset_RF_model.joblib \
      --scaler output/models/preeclampsia_onset_RF_scaler.joblib \
      --input  new_patients.csv \
      --output predictions.csv

# If your CSV uses a different delimiter (default auto-detects ; , \\t):
  python inference.py --model ... --scaler ... --input data.tsv --output preds.csv
"""

import argparse, os, sys, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference with a saved model and scaler.")
    p.add_argument("--model",  type=str, required=True,
                   help="Path to the .joblib model file.")
    p.add_argument("--scaler", type=str, required=True,
                   help="Path to the .joblib scaler file.")
    p.add_argument("--input",  type=str, required=True,
                   help="Path to the input CSV/TSV with feature columns.")
    p.add_argument("--output", type=str, default=None,
                   help="Path to save predictions CSV. "
                        "If omitted, prints to stdout.")
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────
def smart_read(path):
    """Auto-detect delimiter (;  ,  \\t)."""
    for d in [';', ',', '\t']:
        df = pd.read_csv(path, delimiter=d)
        if df.shape[1] > 1:
            return df
    raise ValueError("Unable to detect delimiter.")


OUTCOME_COLS = [
    "gestational_age_delivery", "newborn_weight",
    "preeclampsia_onset", "delivery_type", "newborn_vital_status",
    "newborn_malformations", "eclampsia_hellp", "iugr",
]


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. Load model & scaler ------------------------------------------------
    if not os.path.isfile(args.model):
        print(f"[ERROR] Model file not found: {args.model}"); sys.exit(1)
    if not os.path.isfile(args.scaler):
        print(f"[ERROR] Scaler file not found: {args.scaler}"); sys.exit(1)

    model  = joblib.load(args.model)
    scaler = joblib.load(args.scaler)
    print(f"[INFO] Loaded model  : {args.model}")
    print(f"[INFO] Loaded scaler : {args.scaler}")

    # 2. Read input data -----------------------------------------------------
    df = smart_read(args.input)

    # Drop ID column if present
    if 'id' in df.columns:
        ids = df['id'].copy()
        df = df.drop(columns=['id'])
    else:
        ids = None

    # Drop unnamed index column if present
    if df.columns[0].startswith('Unnamed'):
        df = df.drop(columns=[df.columns[0]])

    # Drop any outcome columns that happen to be in the input
    drop_cols = [c for c in OUTCOME_COLS if c in df.columns]
    if drop_cols:
        print(f"[INFO] Dropping outcome columns found in input: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # 3. Encode object/string columns ----------------------------------------
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    feature_names = list(df.columns)
    print(f"[INFO] {len(df)} samples × {len(feature_names)} features")

    # 4. Scale ---------------------------------------------------------------
    X_scaled = scaler.transform(df.values)

    # 5. Predict -------------------------------------------------------------
    predictions = model.predict(X_scaled)

    # Try to get probabilities (only classifiers expose predict_proba)
    probas = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_scaled)
        except Exception:
            pass

    # 6. Build output dataframe ----------------------------------------------
    result = pd.DataFrame()
    if ids is not None:
        result["id"] = ids.values

    result["prediction"] = predictions

    if probas is not None:
        classes = model.classes_
        for i, cls in enumerate(classes):
            result[f"prob_class_{cls}"] = probas[:, i]

    # 7. Output --------------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"[INFO] Predictions saved to: {args.output}")
    else:
        print("\n── Predictions ──")
        print(result.to_string(index=False))

    return result


if __name__ == "__main__":
    main()
