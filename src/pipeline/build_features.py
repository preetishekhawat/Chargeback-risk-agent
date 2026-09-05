"""
Feature engineering on the joined dataset.
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

CATEGORICAL_COLS = [
    "payment_type",
    "authorization_status",
    "transaction_status",
    "order_status",
    "order_channel",
    "derived_delivery_state",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "payment_value" in df.columns:
        df["amount_bucket"] = pd.qcut(
            df["payment_value"].fillna(df["payment_value"].median()),
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        )

    if "payment_total" in df.columns and "payment_value" in df.columns:
        df["total_to_value_ratio"] = df["payment_total"] / df["payment_value"].replace(0, pd.NA)

    if "delivery_delay_days" in df.columns:
        df["is_delayed"] = (df["delivery_delay_days"].fillna(0) > 0).astype(int)

    ts_col = "order_purchase_timestamp"
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df["purchase_hour"] = df[ts_col].dt.hour
        df["purchase_dow"] = df[ts_col].dt.dayofweek

    return df


def encode_categoricals(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    present_cols = [c for c in cols if c in df.columns]
    df = pd.get_dummies(df, columns=present_cols, dummy_na=True)
    return df


def main():
    in_path = PROCESSED_DIR / "joined.csv"
    df = pd.read_csv(in_path, low_memory=False)

    df = add_derived_features(df)

    all_cat_cols = CATEGORICAL_COLS + ["amount_bucket"]
    df = encode_categoricals(df, all_cat_cols)

    out_path = PROCESSED_DIR / "features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved feature-engineered dataset to {out_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()