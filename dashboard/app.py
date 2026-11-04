
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"

ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_FILE = REPORT_DIR / "analyst_feedback.csv"

REVIEW_STATUSES = [
    "Under Review",
    "Confirmed Fraud",
    "False Positive"
]

st.set_page_config(
    page_title="NEXVIA | Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def load_feedback():
    if FEEDBACK_FILE.exists():
        try:
            return pd.read_csv(FEEDBACK_FILE).fillna("")
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return pd.DataFrame()

    return pd.DataFrame()


def save_prediction(record):
    """Persist a new transaction for analyst review."""

    feedback = load_feedback()

    new_record = {
        **record,
        "analyst_status": "Under Review",
        "analyst_note": "",
        "reviewed_at": ""
    }

    updated = pd.concat(
        [feedback, pd.DataFrame([new_record])],
        ignore_index=True
    )

    updated.to_csv(FEEDBACK_FILE, index=False)


def update_feedback(transaction_id, status, note):
    """Update the analyst decision for a transaction."""

    feedback = load_feedback()

    if feedback.empty:
        return False

    match = feedback["transaction_id"].astype(str) == str(transaction_id)

    if not match.any():
        return False

    feedback.loc[match, "analyst_status"] = status
    feedback.loc[match, "analyst_note"] = note
    feedback.loc[match, "reviewed_at"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    feedback.to_csv(FEEDBACK_FILE, index=False)
    return True


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🛡️ NEXVIA")
st.subheader("Real-Time Financial Fraud Detection Platform")

st.caption(
    "AI-powered transaction monitoring and fraud risk analysis"
)

# --------------------------------------------------
# API STATUS
# --------------------------------------------------

try:
    health = requests.get(HEALTH_URL, timeout=3)
    api_online = health.status_code == 200
except requests.RequestException:
    api_online = False

if api_online:
    st.success("API Status: Online")
else:
    st.error("API Status: Offline. Start your FastAPI server first.")

st.divider()

# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

history = st.session_state.history

total = len(history)

fraud_count = sum(
    item["prediction"] == 1 for item in history
)

legitimate_count = total - fraud_count

avg_probability = (
    sum(item["probability"] for item in history) / total
    if total else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Transactions Analyzed", total)
col2.metric("Fraud Detected", fraud_count)
col3.metric("Legitimate", legitimate_count)
col4.metric("Average Fraud Probability", f"{avg_probability:.2%}")

st.divider()

# --------------------------------------------------
# TRANSACTION INPUT
# --------------------------------------------------

st.header("🔍 Analyze a Transaction")

with st.form("transaction_form"):

    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input(
            "Transaction Time Step",
            min_value=1,
            value=100
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    with col2:
        oldbalanceOrg = st.number_input(
            "Sender's Previous Balance",
            min_value=0.0,
            value=60000.0,
            step=1000.0
        )

        oldbalanceDest = st.number_input(
            "Receiver's Previous Balance",
            min_value=0.0,
            value=10000.0,
            step=1000.0
        )

    submitted = st.form_submit_button(
        "🚀 Analyze Transaction",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    if not api_online:
        st.error("FastAPI server is unavailable.")

    else:

        payload = {
            "step": int(step),
            "transaction_type": transaction_type,
            "amount": float(amount),
            "oldbalanceOrg": float(oldbalanceOrg),
            "oldbalanceDest": float(oldbalanceDest)
        }

        try:
            with st.spinner("Analyzing transaction..."):

                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=15
                )

            if response.status_code == 200:

                result = response.json()

                transaction_id = str(uuid4())

                record = {
                    "transaction_id": transaction_id,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": result["transaction_type"],
                    "amount": amount,
                    "probability": result["fraud_probability"],
                    "prediction": result["prediction"],
                    "risk": result["risk_level"],
                    "response_ms": result["response_time_ms"]
                }

                st.session_state.history.append(record)

                save_prediction(record)

                st.subheader("Prediction Result")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Fraud Probability",
                    f"{result['fraud_probability']:.4%}"
                )

                c2.metric(
                    "Risk Level",
                    result["risk_level"]
                )

                c3.metric(
                    "Response Time",
                    f"{result['response_time_ms']:.3f} ms"
                )

                if result["prediction"] == 1:
                    st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
                else:
                    st.success("✅ TRANSACTION CLASSIFIED AS LEGITIMATE")

                st.caption(f"Transaction ID: {transaction_id}")

            else:
                st.error(f"API Error: {response.text}")

        except requests.RequestException as error:
            st.error(f"Could not connect to API: {error}")

# --------------------------------------------------
# TRANSACTION HISTORY
# --------------------------------------------------

st.divider()
st.header("📊 Transaction Monitoring")

if history:

    history_df = pd.DataFrame(history)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction Distribution")

        counts = (
            history_df["prediction"]
            .map({0: "Legitimate", 1: "Fraudulent"})
            .value_counts()
            .reset_index()
        )

        counts.columns = ["Category", "Count"]

        fig = px.pie(
            counts,
            names="Category",
            values="Count",
            hole=0.45,
            title="Analyzed Transactions"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud Probability")

        fig = px.bar(
            history_df,
            x="time",
            y="probability",
            color="risk",
            hover_data=["type", "amount"],
            title="Transaction Risk Analysis"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Transaction History")

    display_df = history_df.rename(columns={
        "time": "Time",
        "type": "Transaction Type",
        "amount": "Amount",
        "probability": "Fraud Probability",
        "prediction": "Prediction",
        "risk": "Risk Level",
        "response_ms": "Response Time (ms)"
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Transaction History",
        data=csv,
        file_name="fraud_transaction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear Dashboard History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No transactions analyzed yet. Submit a transaction above.")

# --------------------------------------------------
# FRAUD NETWORK INTELLIGENCE
# --------------------------------------------------

st.divider()
st.header("🕸️ Fraud Network Intelligence")

RING_FILE = ROOT / "reports" / "fraud_network" / "fraud_rings.csv"
ACCOUNT_FILE = ROOT / "reports" / "fraud_network" / "suspicious_accounts.csv"

if RING_FILE.exists() and ACCOUNT_FILE.exists():

    rings = pd.read_csv(RING_FILE)
    accounts = pd.read_csv(ACCOUNT_FILE)

    candidate_rings = rings[
        (rings["fraud_transactions"] >= 2)
        & (rings["accounts"] >= 3)
    ].copy()

    c1, c2, c3 = st.columns(3)

    c1.metric("Fraud-linked Groups", len(rings))
    c2.metric("Candidate Fraud Clusters", len(candidate_rings))
    c3.metric("Accounts Analyzed", len(accounts))

    st.subheader("Candidate Fraud Clusters")

    if not candidate_rings.empty:

        st.dataframe(
            candidate_rings[
                [
                    "ring_id",
                    "accounts",
                    "transactions",
                    "fraud_transactions",
                    "risk_level"
                ]
            ].sort_values(
                "fraud_transactions",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No candidate clusters found.")

    st.subheader("Top Suspicious Accounts")

    st.dataframe(
        accounts.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Network results are investigative indicators derived "
        "from known PaySim fraud labels, not confirmed criminal rings."
    )

else:
    st.warning(
        "Fraud network reports not found. "
        "Run src/graph/fraud_ring_detection.py first."
    )

# --------------------------------------------------
# ANALYST REVIEW & FEEDBACK
# --------------------------------------------------

st.divider()
st.header("🧑‍💼 Analyst Review & Feedback")

st.caption(
    "Review AI-generated transaction alerts and record the analyst's decision."
)

feedback = load_feedback()

if not feedback.empty:

    total_reviews = len(feedback)

    confirmed = (
        feedback["analyst_status"] == "Confirmed Fraud"
    ).sum()

    false_positives = (
        feedback["analyst_status"] == "False Positive"
    ).sum()

    under_review = (
        feedback["analyst_status"] == "Under Review"
    ).sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Alerts", total_reviews)
    c2.metric("Confirmed Fraud", int(confirmed))
    c3.metric("False Positives", int(false_positives))
    c4.metric("Under Review", int(under_review))

    st.subheader("Review Transaction Alerts")

    # Show newest transactions first
    review_records = feedback.iloc[::-1].head(50)

    for _, item in review_records.iterrows():

        transaction_id = str(item["transaction_id"])

        prediction_label = (
            "Fraudulent"
            if int(item["prediction"]) == 1
            else "Legitimate"
        )

        heading = (
            f"{item['type']} | "
            f"Amount: {float(item['amount']):,.2f} | "
            f"AI: {prediction_label} | "
            f"Risk: {item['risk']}"
        )

        with st.expander(heading):

            st.write(f"**Transaction ID:** `{transaction_id}`")
            st.write(f"**Time:** {item['time']}")
            st.write(f"**Fraud Probability:** {float(item['probability']):.4%}")
            st.write(f"**API Response Time:** {float(item['response_ms']):.3f} ms")

            current_status = item["analyst_status"]

            if current_status not in REVIEW_STATUSES:
                current_status = "Under Review"

            with st.form(f"review_form_{transaction_id}"):

                status = st.selectbox(
                    "Analyst Decision",
                    REVIEW_STATUSES,
                    index=REVIEW_STATUSES.index(current_status),
                    key=f"status_{transaction_id}"
                )

                note = st.text_area(
                    "Analyst Notes",
                    value=str(item["analyst_note"]),
                    placeholder="Enter reason for your decision...",
                    key=f"note_{transaction_id}"
                )

                save_button = st.form_submit_button(
                    "💾 Save Analyst Review"
                )

                if save_button:

                    success = update_feedback(
                        transaction_id,
                        status,
                        note
                    )

                    if success:
                        st.success("Analyst review saved successfully.")
                        st.rerun()
                    else:
                        st.error("Could not update this transaction.")

            if item["reviewed_at"]:
                st.caption(f"Last reviewed: {item['reviewed_at']}")
            else:
                st.caption("This transaction has not been reviewed yet.")

    st.subheader("Analyst Feedback Records")

    st.dataframe(
        feedback[
            [
                "transaction_id",
                "type",
                "amount",
                "probability",
                "prediction",
                "analyst_status",
                "analyst_note",
                "reviewed_at"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    feedback_csv = feedback.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Analyst Feedback",
        data=feedback_csv,
        file_name="analyst_feedback.csv",
        mime="text/csv"
    )

else:
    st.info(
        "No analyst review records yet. "
        "Analyze a transaction to create the first alert."
    )

st.caption(
    f"Feedback storage: {FEEDBACK_FILE}"
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "NEXVIA Financial Fraud Detection | "
    "Educational AI/ML Internship Project | "
    "Predictions are not financial decisions."
)
