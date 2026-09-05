"""
Streamlit demo: pick a transaction from the test set, see its risk
score, top contributing factors, and the escalate/resolve decision.

Run with: streamlit run src/pipeline/app.py
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

st.set_page_config(page_title="Dispute Risk Scorer", layout="wide")

st.title("Chargeback / Dispute Risk Scorer")
st.caption("Track 2: AI Risk Manager — Razorpay Buildathon")

# --- Load model + audit trail (already-scored test set) ---
@st.cache_data
def load_data():
    df = pd.read_csv(LOGS_DIR / "decision_audit_trail.csv")
    return df

df = load_data()

st.markdown("### Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Transactions evaluated", len(df))
col2.metric("Escalated to review", int((df["decision"] == "escalate").sum()))
col3.metric("Auto-resolved", int((df["decision"] == "resolve").sum()))

st.markdown("---")

# --- Pick a case ---
st.markdown("### Inspect a case")
idx = st.selectbox("Pick a transaction (by row index)", df.index[:200])

row = df.loc[idx]

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Risk Score**")
    st.progress(min(float(row["risk_score"]), 1.0))
    st.write(f"{row['risk_score']:.3f}")

    st.markdown("**Decision**")
    if row["decision"] == "escalate":
        st.error("ESCALATE — routed to human review")
    else:
        st.success("RESOLVE — auto-cleared")

    st.markdown("**Actually disputed (ground truth)**")
    st.write("Yes" if row["actual_disputed"] == 1 else "No")

with col2:
    st.markdown("**Case details**")
    display_cols = [c for c in df.columns if c not in ["risk_score", "decision", "actual_disputed"]]
    st.dataframe(row[display_cols].astype(str), width="stretch")

st.markdown("---")
st.markdown("### Full audit trail (first 200 rows)")
st.dataframe(df.head(200), width="stretch")
