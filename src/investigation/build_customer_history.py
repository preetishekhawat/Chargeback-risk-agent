"""
Build customer-history features using ONLY orders strictly before the
current order's purchase timestamp. Every feature here is computed as
"sum/count of prior orders" via cumsum(including current) - (current
row's own value), which correctly excludes the current row.

Output: one row per order_id, with history features + order_purchase_timestamp
kept for merging into the main feature set.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def main():
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    transactions = pd.read_csv(RAW_DIR / "transactions.csv")

    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

    # Order value = sum of payment_value across all transactions for that order
    order_value = (
        transactions.groupby("canonical_order_id")["payment_value"]
        .sum()
        .rename("order_value")
        .reset_index()
    )
    orders = orders.merge(order_value, how="left", left_on="order_id", right_on="canonical_order_id")
    orders["order_value"] = orders["order_value"].fillna(0)

    orders["is_canceled"] = (orders["order_status"] == "canceled").astype(int)
    orders["is_delayed"] = (orders["delivery_delay_days"].fillna(0) > 0).astype(int)

    # Sort chronologically within each customer -- critical for "prior only"
    orders = orders.sort_values(["customer_id", "order_purchase_timestamp", "order_id"]).reset_index(drop=True)
    grp = orders.groupby("customer_id")

    # cumcount() gives count of PRIOR rows in the group (0-indexed, excludes current)
    orders["customer_previous_order_count"] = grp.cumcount()

    # cumsum(including current) - current_value = sum of PRIOR rows only
    orders["customer_previous_total_spend"] = grp["order_value"].cumsum() - orders["order_value"]
    orders["customer_previous_cancel_count"] = grp["is_canceled"].cumsum() - orders["is_canceled"]
    orders["customer_previous_delayed_count"] = grp["is_delayed"].cumsum() - orders["is_delayed"]

    prior_delay_sum = grp["delivery_delay_days"].cumsum() - orders["delivery_delay_days"].fillna(0)

    n_prev = orders["customer_previous_order_count"].replace(0, pd.NA)
    orders["customer_previous_avg_order_value"] = orders["customer_previous_total_spend"] / n_prev
    orders["customer_previous_cancellation_rate"] = orders["customer_previous_cancel_count"] / n_prev
    orders["customer_previous_delayed_order_rate"] = orders["customer_previous_delayed_count"] / n_prev
    orders["customer_previous_avg_delivery_delay"] = prior_delay_sum / n_prev

    # Days since previous order (shift(1) within group = the PRIOR row's timestamp)
    prev_ts = grp["order_purchase_timestamp"].shift(1)
    orders["customer_days_since_previous_order"] = (
        orders["order_purchase_timestamp"] - prev_ts
    ).dt.total_seconds() / 86400

    history_cols = [
        "order_id",
        "customer_previous_order_count",
        "customer_previous_total_spend",
        "customer_previous_avg_order_value",
        "customer_previous_cancel_count",
        "customer_previous_cancellation_rate",
        "customer_previous_avg_delivery_delay",
        "customer_previous_delayed_order_rate",
        "customer_days_since_previous_order",
    ]
    history = orders[history_cols]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "customer_history.csv"
    history.to_csv(out_path, index=False)
    print(f"Saved customer history features to {out_path}")
    print(f"Shape: {history.shape}")
    print("\n--- Sample ---")
    print(history.head(10))
    print("\n--- Describe ---")
    print(history.describe())


if __name__ == "__main__":
    main()