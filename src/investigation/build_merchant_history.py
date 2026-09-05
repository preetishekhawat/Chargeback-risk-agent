"""
Build merchant-history features using ONLY orders strictly before the
current order's purchase timestamp, per merchant -- mirrors the
customer-history logic but keyed on merchant_id.

merchant_previous_dispute_rate is the key candidate: given the observed
merchant-level variance (std 0.0458 vs base rate 0.0963), this is the
most promising untested feature so far.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def main():
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    disputes = pd.read_csv(RAW_DIR / "disputes.csv")

    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["is_disputed"] = orders["order_id"].isin(disputes["canonical_order_id"]).astype(int)

    orders = orders.sort_values(["merchant_id", "order_purchase_timestamp", "order_id"]).reset_index(drop=True)
    grp = orders.groupby("merchant_id")

    orders["merchant_previous_order_count"] = grp.cumcount()
    orders["merchant_previous_dispute_count"] = grp["is_disputed"].cumsum() - orders["is_disputed"]

    n_prev = orders["merchant_previous_order_count"].replace(0, pd.NA)
    orders["merchant_previous_dispute_rate"] = orders["merchant_previous_dispute_count"] / n_prev

    history_cols = [
        "order_id",
        "merchant_previous_order_count",
        "merchant_previous_dispute_count",
        "merchant_previous_dispute_rate",
    ]
    history = orders[history_cols]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "merchant_history.csv"
    history.to_csv(out_path, index=False)
    print(f"Saved merchant history features to {out_path}")
    print(f"Shape: {history.shape}")
    print("\n--- Describe ---")
    print(history.describe())
    print(f"\nRows with merchant_previous_order_count == 0 (cold start, no prior data): "
          f"{(history['merchant_previous_order_count'] == 0).sum()} "
          f"({(history['merchant_previous_order_count'] == 0).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()