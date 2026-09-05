"""
Computes the risk-system metrics (Precision@10%, Recall@10%, Lift@10%,
PR-AUC) from the already-saved decision audit trail -- no retraining.

Run with: python src/investigation/business_metrics.py
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


def main():
    df = pd.read_csv(LOGS_DIR / "decision_audit_trail.csv")

    y_true = df["actual_disputed"]
    y_score = df["risk_score"]

    total = len(df)
    dispute_rate = y_true.mean()

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    reviewed = (df["decision"] == "escalate")
    n_reviewed = reviewed.sum()
    precision_at_10 = precision_score(y_true, reviewed)
    recall_at_10 = recall_score(y_true, reviewed)
    lift_at_10 = precision_at_10 / dispute_rate
    disputes_caught = int((y_true & reviewed).sum())

    print("=== Business-Facing Risk Metrics ===")
    print(f"Transactions evaluated: {total}")
    print(f"Dispute rate:           {dispute_rate*100:.1f}%")
    print()
    print(f"ROC-AUC:                {roc_auc:.3f}")
    print(f"PR-AUC:                 {pr_auc:.3f}")
    print()
    print(f"Transactions reviewed (top ~10%): {n_reviewed}")
    print(f"Precision@10%:          {precision_at_10*100:.1f}%")
    print(f"Recall@10%:             {recall_at_10*100:.1f}%")
    print(f"Lift@10%:               {lift_at_10:.2f}x")
    print(f"Disputes caught:        {disputes_caught} / {int(y_true.sum())}")


if __name__ == "__main__":
    main()
