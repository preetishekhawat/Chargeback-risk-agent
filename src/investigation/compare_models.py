"""
Model A: current transaction/order features -> AUC (baseline, 0.544)
Model B: customer-history features only -> AUC ?
Model C: both combined -> AUC ?

Does NOT touch or overwrite the locked train_model.py / risk_model.pkl.
This is purely a diagnostic comparison.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

DROP_COLS = [
    "transaction_id", "order_id", "canonical_order_id", "customer_id", "merchant_id",
    "is_disputed", "dispute_type", "dispute_reason", "dispute_amount", "dispute_status",
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]

HISTORY_COLS = [
    "customer_previous_order_count",
    "customer_previous_total_spend",
    "customer_previous_avg_order_value",
    "customer_previous_cancel_count",
    "customer_previous_cancellation_rate",
    "customer_previous_avg_delivery_delay",
    "customer_previous_delayed_order_rate",
    "customer_days_since_previous_order",
]


import numpy as np


def evaluate(X, y, label, model="logistic"):
    X = X.fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model == "logistic":
        scaler = StandardScaler()
        X_train_in = scaler.fit_transform(X_train)
        X_test_in = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    else:  # nonlinear
        from sklearn.ensemble import RandomForestClassifier
        X_train_in, X_test_in = X_train, X_test
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )

    clf.fit(X_train_in, y_train)
    y_proba = clf.predict_proba(X_test_in)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n=== {label} ({model}) ===")
    print(f"Features used: {X.shape[1]}")
    print(f"ROC-AUC: {auc:.4f}")
    return auc, y_test, y_proba


def bootstrap_auc_ci(y_test, y_proba, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_test = np.asarray(y_test)
    y_proba = np.asarray(y_proba)
    n = len(y_test)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_test[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_test[idx], y_proba[idx]))
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return np.mean(scores), lower, upper


def main():
    features = pd.read_csv(PROCESSED_DIR / "features.csv", low_memory=False)
    history = pd.read_csv(PROCESSED_DIR / "customer_history.csv")
    merchant_history = pd.read_csv(PROCESSED_DIR / "merchant_history.csv")

    joined = pd.read_csv(PROCESSED_DIR / "joined.csv", low_memory=False)
    joined = joined.merge(history, how="left", on="order_id")
    joined = joined.merge(merchant_history, how="left", on="order_id")

    y = joined["is_disputed"]

    current_cat_cols = [
        "payment_type", "authorization_status", "transaction_status",
        "order_status", "order_channel", "derived_delivery_state",
    ]
    current_num_cols = [
        "payment_value", "payment_total", "payment_installments", "payment_count",
        "delivery_delay_days", "purchase_to_delivery_days",
    ]
    current_encoded = pd.get_dummies(joined[current_cat_cols], dummy_na=True)
    X_a = pd.concat([joined[current_num_cols], current_encoded], axis=1)

    X_b = joined[HISTORY_COLS]

    merchant_cols = ["merchant_previous_order_count", "merchant_previous_dispute_count", "merchant_previous_dispute_rate"]
    X_d = joined[merchant_cols]

    X_c = pd.concat([X_a, X_b], axis=1)
    X_e = pd.concat([X_a, X_b, X_d], axis=1)

    auc_a, _, _ = evaluate(X_a, y, "Model A -- current transaction/order features")
    auc_b, _, _ = evaluate(X_b, y, "Model B -- customer history only")
    auc_c, _, _ = evaluate(X_c, y, "Model C -- current + customer history")
    auc_d, _, _ = evaluate(X_d, y, "Model D -- merchant history only")
    auc_e, _, _ = evaluate(X_e, y, "Model E -- current + customer history + merchant history")

    # Model F: same "everything" feature set, but a nonlinear model --
    # tests whether LogisticRegression was missing interaction effects
    auc_f, y_test_f, y_proba_f = evaluate(X_e, y, "Model F -- everything, RandomForest (nonlinear)", model="nonlinear")

    print("\n=== Summary ===")
    print(f"Model A (current, logistic):                    {auc_a:.4f}")
    print(f"Model B (customer history only, logistic):      {auc_b:.4f}")
    print(f"Model C (current + customer history, logistic): {auc_c:.4f}")
    print(f"Model D (merchant history only, logistic):      {auc_d:.4f}")
    print(f"Model E (everything, logistic):                 {auc_e:.4f}")
    print(f"Model F (everything, RandomForest/nonlinear):   {auc_f:.4f}")

    print("\n=== Bootstrap 95% CI for Model F (best/nonlinear) ===")
    mean_auc, lower, upper = bootstrap_auc_ci(y_test_f, y_proba_f)
    print(f"Mean bootstrap AUC: {mean_auc:.4f}")
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")


if __name__ == "__main__":
    main()