"""Data loading and feature preparation for smart-meter theft analytics."""
import pandas as pd
import numpy as np

def load_and_prepare(uploaded_file, target_col="FLAG", id_cols=None):
    """Load a labelled CSV and create consumer-level summary features."""
    df = pd.read_csv(uploaded_file)
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")
    id_cols = id_cols or []
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.notna()
    df, y = df.loc[valid].copy(), y.loc[valid].astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain binary labels 0 and 1.")
    Xraw = df.drop(columns=[c for c in [target_col, *id_cols] if c in df.columns])
    Xraw = Xraw.apply(pd.to_numeric, errors="coerce")
    feats = pd.DataFrame(index=Xraw.index)
    feats["reading_mean"] = Xraw.mean(axis=1)
    feats["reading_std"] = Xraw.std(axis=1).fillna(0)
    feats["reading_min"] = Xraw.min(axis=1)
    feats["reading_max"] = Xraw.max(axis=1)
    feats["reading_median"] = Xraw.median(axis=1)
    feats["reading_cv"] = feats["reading_std"] / feats["reading_mean"].abs().replace(0, np.nan)
    feats["zero_fraction"] = (Xraw == 0).mean(axis=1)
    feats["missing_fraction"] = Xraw.isna().mean(axis=1)
    feats["negative_fraction"] = (Xraw < 0).mean(axis=1)
    split = max(1, Xraw.shape[1] // 2)
    feats["first_half_mean"] = Xraw.iloc[:, :split].mean(axis=1)
    feats["second_half_mean"] = Xraw.iloc[:, split:].mean(axis=1)
    feats["recent_to_earlier_ratio"] = feats["second_half_mean"] / feats["first_half_mean"].abs().replace(0, np.nan)
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.fillna(feats.median(numeric_only=True)).fillna(0)
    return df, feats, y, Xraw.shape[1]

def make_demo_data(n=1400, seed=42):
    """Synthetic demonstration data; not real utility/customer data."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.12, n)
    rows = []
    for label in y:
        level = rng.normal(8.5 if label == 0 else 5.2, 1.8 if label == 0 else 2.2)
        vals = np.maximum(0, rng.normal(level, 1.8, 48))
        if label:
            vals[rng.choice(48, 8, replace=False)] *= rng.uniform(0.05, 0.3)
        vals[rng.choice(48, 2, replace=False)] = np.nan
        rows.append(vals)
    readings = pd.DataFrame(rows, columns=[f"reading_{i+1}" for i in range(48)])
    readings.insert(0, "FLAG", y)
    return readings
