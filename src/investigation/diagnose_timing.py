"""
Diagnostic: for every disputed transaction, compare dispute_opened_at
against order_delivered_customer_date to determine whether disputes
happen before, after, or around delivery.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

def main():
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    disputes = pd.read_csv(RAW_DIR / "disputes.csv")

    # disputes.csv already has canonical_order_id -- join straight to orders
    df = disputes.merge(
        orders[["order_id", "order_delivered_customer_date", "order_purchase_timestamp"]],
        how="left", left_on="canonical_order_id", right_on="order_id"
    )

    df["dispute_opened_at"] = pd.to_datetime(df["dispute_opened_at"], errors="coerce")
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")

    total = len(df)
    missing_delivery = df["order_delivered_customer_date"].isna().sum()

    has_both = df.dropna(subset=["dispute_opened_at", "order_delivered_customer_date"]).copy()
    has_both["days_delivery_minus_dispute"] = (
        has_both["order_delivered_customer_date"] - has_both["dispute_opened_at"]
    ).dt.total_seconds() / 86400

    before_delivery = (has_both["days_delivery_minus_dispute"] > 0).sum()
    after_delivery = (has_both["days_delivery_minus_dispute"] < 0).sum()
    same_day = (has_both["days_delivery_minus_dispute"] == 0).sum()

    print(f"Total disputed transactions: {total}")
    print(f"Missing delivery date: {missing_delivery} ({missing_delivery/total*100:.1f}%)")
    print(f"With both dates available: {len(has_both)}")
    print()
    print(f"Dispute BEFORE delivery: {before_delivery} ({before_delivery/len(has_both)*100:.1f}%)")
    print(f"Dispute AFTER delivery:  {after_delivery} ({after_delivery/len(has_both)*100:.1f}%)")
    print(f"Dispute SAME DAY as delivery: {same_day} ({same_day/len(has_both)*100:.1f}%)")

    print()
    print("=== Distribution of (delivery_date - dispute_opened_at) in days ===")
    print(has_both["days_delivery_minus_dispute"].describe())

    df["days_dispute_minus_purchase"] = (
        df["dispute_opened_at"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    print()
    print("=== Distribution of (dispute_opened_at - order_purchase_timestamp) in days ===")
    print(df["days_dispute_minus_purchase"].describe())
    negative_purchase_gap = (df["days_dispute_minus_purchase"] < 0).sum()
    print(f"\nDisputes opened BEFORE purchase timestamp (should be 0, sanity check): {negative_purchase_gap}")


if __name__ == "__main__":
    main()