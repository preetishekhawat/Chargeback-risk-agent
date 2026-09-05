"""
Load and join transactions, orders, and disputes.

Uses the razorpay_dispute_ai_dataset_v1 folder. transactions.csv has
the real link column: canonical_order_id, which joins to order_id
in orders.csv.

Place v1's orders.csv, transactions.csv, and disputes.csv directly in
data/raw/ (flat, not nested) before running this.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def load_raw():
    transactions = pd.read_csv(RAW_DIR / "transactions.csv")
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    disputes = pd.read_csv(RAW_DIR / "disputes.csv")
    return transactions, orders, disputes


def build_joined_dataset(transactions, orders, disputes):
    # Real join key: transactions.canonical_order_id -> orders.order_id
    df = transactions.merge(
        orders, how="left", left_on="canonical_order_id", right_on="order_id"
    )

    disputes_slim = disputes[
        ["transaction_id", "dispute_type", "dispute_reason", "dispute_amount", "dispute_status"]
    ].copy()
    disputes_slim["is_disputed"] = 1

    df = df.merge(disputes_slim, how="left", on="transaction_id")
    df["is_disputed"] = df["is_disputed"].fillna(0).astype(int)

    return df


def main():
    print("Loading raw CSVs...")
    transactions, orders, disputes = load_raw()

    print(f"transactions: {transactions.shape}")
    print(f"orders: {orders.shape}")
    print(f"disputes: {disputes.shape}")

    df = build_joined_dataset(transactions, orders, disputes)
    print(f"\nJoined dataset shape: {df.shape}")
    print(f"Disputed rows: {df['is_disputed'].sum()} / {len(df)}")
    print(f"Rows with no matching order: {df['order_id'].isna().sum()}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "joined.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved joined dataset to {out_path}")


if __name__ == "__main__":
    main()