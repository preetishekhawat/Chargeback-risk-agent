"""
Train the dispute risk classifier and evaluate honestly on a held-out
test set. Also runs the decision layer: top 10% highest-risk cases
get escalated to human review, the rest are auto-resolved.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"

DROP_COLS = [
    "transaction_id", "order_id", "canonical_order_id", "customer_id", "merchant_id",
    "is_disputed", "dispute_type", "dispute_reason", "dispute_amount", "dispute_status",
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]

ESCALATE_PERCENTILE = 0.90  # top 10% of risk scores get escalated


def main():
    df = pd.read_csv(PROCESSED_DIR / "features.csv", low_memory=False)

    y = df["is_disputed"]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = X.select_dtypes(include=["number", "bool"])
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    importances = pd.Series(clf.coef_[0], index=X.columns)
    print("\n=== Top 10 Feature Coefficients ===")
    print(importances.sort_values(ascending=False).head(10))

    # --- Decision layer + audit trail ---
    results = X_test.copy()
    results["risk_score"] = y_proba
    results["actual_disputed"] = y_test.values

    cutoff = results["risk_score"].quantile(ESCALATE_PERCENTILE)
    results["decision"] = results["risk_score"].apply(
        lambda s: "escalate" if s >= cutoff else "resolve"
    )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "decision_audit_trail.csv"
    results.to_csv(log_path, index=False)
    print(f"\nDecision audit trail saved to {log_path}")
    print(results["decision"].value_counts())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "risk_model.pkl")
    print(f"Model saved to {MODELS_DIR / 'risk_model.pkl'}")


if __name__ == "__main__":
    main()