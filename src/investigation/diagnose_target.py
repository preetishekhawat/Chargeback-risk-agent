"""
Investigate the target variable construction itself:
- class balance
- whether disputes cluster in a specific time period
- duplicate transactions/orders
- whether disputes are associated with a narrow subset (e.g. one
  merchant, one payment_type, one order_channel) rather than being
  spread across the population
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


def main():
    transactions = pd.read_csv(RAW_DIR / "transactions.csv")
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    disputes = pd.read_csv(RAW_DIR / "disputes.csv")

    print("=== Class balance ===")
    n_total = len(transactions)
    n_disputed = disputes["transaction_id"].nunique()
    print(f"Total transactions: {n_total}")
    print(f"Disputed transactions (unique): {n_disputed}")
    print(f"Dispute rate: {n_disputed/n_total*100:.2f}%")

    print("\n=== Duplicate check ===")
    print(f"Duplicate transaction_ids in transactions.csv: {transactions['transaction_id'].duplicated().sum()}")
    print(f"Duplicate order_ids in orders.csv: {orders['order_id'].duplicated().sum()}")
    print(f"Duplicate dispute_ids in disputes.csv: {disputes['dispute_id'].duplicated().sum()}")
    print(f"Transactions appearing more than once in disputes.csv: {(disputes['transaction_id'].value_counts() > 1).sum()}")

    print("\n=== Time period distribution ===")
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    print("All orders date range:")
    print(f"  {orders['order_purchase_timestamp'].min()} to {orders['order_purchase_timestamp'].max()}")

    disputed_order_ids = disputes["canonical_order_id"].unique()
    disputed_orders = orders[orders["order_id"].isin(disputed_order_ids)]
    print("Disputed orders' purchase date range:")
    print(f"  {disputed_orders['order_purchase_timestamp'].min()} to {disputed_orders['order_purchase_timestamp'].max()}")

    print("\nOrders per year (all):")
    print(orders["order_purchase_timestamp"].dt.year.value_counts().sort_index())
    print("\nDisputed orders per year:")
    print(disputed_orders["order_purchase_timestamp"].dt.year.value_counts().sort_index())

    print("\n=== Is dispute rate uniform across subsets? ===")
    df = transactions.merge(
        orders[["order_id", "order_channel", "merchant_id"]],
        how="left", left_on="canonical_order_id", right_on="order_id"
    )
    df["is_disputed"] = df["transaction_id"].isin(disputes["transaction_id"]).astype(int)

    print("\nDispute rate by payment_type:")
    print(df.groupby("payment_type")["is_disputed"].mean().sort_values(ascending=False))

    print("\nDispute rate by order_channel:")
    print(df.groupby("order_channel")["is_disputed"].mean().sort_values(ascending=False))

    print("\nDispute rate by transaction_status:")
    print(df.groupby("transaction_status")["is_disputed"].mean().sort_values(ascending=False))

    print("\nTop 10 merchants by dispute rate (min 20 transactions):")
    merchant_stats = df.groupby("merchant_id")["is_disputed"].agg(["mean", "count"])
    merchant_stats = merchant_stats[merchant_stats["count"] >= 20].sort_values("mean", ascending=False)
    print(merchant_stats.head(10))
    print(f"\nMerchant dispute rate std dev: {merchant_stats['mean'].std():.4f}")
    print(f"Overall dispute rate (for comparison): {df['is_disputed'].mean():.4f}")


if __name__ == "__main__":
    main()