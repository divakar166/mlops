import time
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Config
st.set_page_config(
    page_title="Fraud Detection Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.title("Settings")
    API_BASE = st.text_input("API Base URL", value="http://localhost:8000")
    API_KEY  = st.text_input("API Key", value="", type="password")
    st.divider()
    auto_refresh = st.toggle("Auto-refresh monitoring (10s)", value=False)
    if st.button("Refresh Now"):
        st.rerun()
    st.divider()
    st.caption("Fraud Detection API · v3.0.0")

HEADERS = {"x-api-key": API_KEY} if API_KEY else {}


def api_post(path: str, payload: dict) -> tuple[dict | None, str | None]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, headers=HEADERS, timeout=10)
        if r.ok:
            return r.json(), None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, str(e)


def api_get(path: str) -> tuple[dict | None, str | None]:
    try:
        r = requests.get(f"{API_BASE}{path}", headers=HEADERS, timeout=10)
        if r.ok:
            return r.json(), None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, str(e)


# Tabs
tab_predict, tab_monitor, tab_recent = st.tabs([
    "Predict", "Monitoring", "Recent Predictions"
])


# TAB 1 - PREDICT
with tab_predict:
    st.header("Transaction Fraud Scorer")
    st.caption("Fill in transaction details and score it against the live API.")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("Transaction Details")
        amount   = st.number_input("Amount ($)", min_value=0.01, value=150.00, step=0.01, format="%.2f")
        hour     = st.slider("Hour of Day", 0, 23, 14)
        dow      = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
            index=3,
        )
        merchant = st.selectbox(
            "Merchant Category",
            ["grocery", "restaurant", "retail", "online", "travel"],
            index=3,
        )

        st.divider()
        # Quick scenario presets
        st.caption("Quick scenarios")
        preset_cols = st.columns(3)
        with preset_cols[0]:
            if st.button("Legit", width='content'):
                st.session_state["preset"] = {"amount": 45.0, "hour": 14, "dow": 2, "merchant": "grocery"}
        with preset_cols[1]:
            if st.button("Suspicious", width='content'):
                st.session_state["preset"] = {"amount": 850.0, "hour": 2, "dow": 6, "merchant": "online"}
        with preset_cols[2]:
            if st.button("High Risk", width='content'):
                st.session_state["preset"] = {"amount": 2400.0, "hour": 3, "dow": 0, "merchant": "travel"}

        # Apply preset if clicked
        if "preset" in st.session_state:
            p = st.session_state.pop("preset")
            amount   = p["amount"]
            hour     = p["hour"]
            dow      = p["dow"]
            merchant = p["merchant"]

        score_btn = st.button("Score Transaction", type="primary", width='content')

    with col_result:
        st.subheader("Prediction Result")

        if score_btn:
            with st.spinner("Scoring…"):
                result, err = api_post("/predict", {
                    "amount": amount,
                    "hour": hour,
                    "day_of_week": dow,
                    "merchant_category": merchant,
                })

            if err:
                st.error(f"API Error: {err}")
            elif result:
                is_fraud = result["is_fraud"]
                prob     = result["fraud_probability"]

                # Big verdict banner
                if is_fraud:
                    st.error("FRAUD DETECTED")
                else:
                    st.success("LEGITIMATE")

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    title={"text": "Fraud Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#e74c3c" if is_fraud else "#2ecc71"},
                        "steps": [
                            {"range": [0,  30], "color": "#d5f5e3"},
                            {"range": [30, 60], "color": "#fdebd0"},
                            {"range": [60, 100],"color": "#fadbd8"},
                        ],
                        "threshold": {"line": {"color": "black","width": 3}, "thickness": 0.75, "value": 50},
                    },
                ))
                fig.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20))
                st.plotly_chart(fig, width='content')

                # Metadata
                meta_cols = st.columns(2)
                meta_cols[0].metric("Fraud Probability", f"{prob*100:.2f}%")
                meta_cols[1].metric("Model Version", result.get("model_version", "—"))
                meta_cols[0].metric("Feast Status", result.get("feast_status", "—"))
                meta_cols[1].metric("Decision Threshold", f"{result.get('decision_threshold', 0.30):.2f}")
        else:
            st.info("Fill in transaction details and click **Score Transaction**.")

            # Show a dummy gauge placeholder
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=0,
                title={"text": "Fraud Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#bdc3c7"},
                    "steps": [
                        {"range": [0,  30], "color": "#d5f5e3"},
                        {"range": [30, 60], "color": "#fdebd0"},
                        {"range": [60, 100],"color": "#fadbd8"},
                    ],
                },
            ))
            fig.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig, width='content')


# TAB 2 — MONITORING
with tab_monitor:
    st.header("Live Monitoring Dashboard")

    stats, stats_err = api_get("/monitoring/stats")
    drift, drift_err = api_get("/monitoring/drift")

    st.subheader("Prediction Stats")
    if stats_err:
        st.warning(f"Could not load stats: {stats_err}")
    elif stats:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Predictions", f"{stats.get('total', 0):,}")
        k2.metric("Total Fraud",       f"{stats.get('total_fraud', 0):,}")
        k3.metric("Fraud Rate",        f"{stats.get('fraud_rate_pct', 0):.2f}%")
        k4.metric("Avg Fraud Prob",    f"{stats.get('avg_fraud_prob_pct', 0):.2f}%")
        k5.metric("Feast Fallbacks",   f"{stats.get('feast_fallbacks', 0):,}")

        # Fraud rate by merchant category bar chart
        by_cat = stats.get("by_category", [])
        if by_cat:
            st.divider()
            cat_df = pd.DataFrame(by_cat)
            col_bar, col_pie = st.columns(2)
            with col_bar:
                st.caption("**Prediction volume by category**")
                fig = px.bar(
                    cat_df, x="merchant_category", y="count",
                    color="fraud_rate_pct",
                    color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                    labels={"merchant_category": "Category", "count": "Predictions", "fraud_rate_pct": "Fraud %"},
                )
                fig.update_layout(height=280, margin=dict(t=20, b=20), coloraxis_showscale=True)
                st.plotly_chart(fig, width='content')
            with col_pie:
                st.caption("**Fraud rate by category**")
                fig2 = px.bar(
                    cat_df, x="merchant_category", y="fraud_rate_pct",
                    color="fraud_rate_pct",
                    color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                    labels={"merchant_category": "Category", "fraud_rate_pct": "Fraud Rate (%)"},
                )
                fig2.update_layout(height=280, margin=dict(t=20, b=20))
                st.plotly_chart(fig2, width='content')

    st.divider()

    st.subheader("Data Drift Monitor")

    col_drift_summary, col_drift_chart = st.columns([1, 2], gap="large")

    if drift_err:
        st.warning(f"Could not load drift data: {drift_err}")
    elif drift:
        summary = drift.get("summary", {})
        alerts  = drift.get("alerts", [])
        history = drift.get("history", [])

        with col_drift_summary:
            if "message" in summary:
                st.info(summary["message"])
            else:
                st.metric("Total Drift Checks", summary.get("total_checks", 0))
                st.metric("Total Alerts",        summary.get("total_alerts", 0))
                st.metric("Max Drift Share",     f"{summary.get('max_drift_share', 0)*100:.1f}%")
                st.metric("Avg Drift Share",     f"{summary.get('avg_drift_share', 0)*100:.1f}%")

            if alerts:
                st.divider()
                st.caption(f"{len(alerts)} alert(s)")
                for a in alerts[-5:]:
                    severity_color = "🔴" if a["severity"] == "HIGH" else "🟡"
                    st.warning(f"{severity_color} [{a['severity']}] {a['message']}\n\nCols: {', '.join(a['drifted_columns'])}")

        with col_drift_chart:
            if history:
                hist_df = pd.DataFrame(history)
                hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
                hist_df["drift_pct"] = hist_df["drift_share"] * 100

                fig = px.line(
                    hist_df, x="timestamp", y="drift_pct",
                    title="Drift Share Over Time (%)",
                    labels={"drift_pct": "Drift Share (%)", "timestamp": "Time"},
                    markers=True,
                )
                fig.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="Alert threshold (10%)")
                fig.add_hline(y=30, line_dash="dash", line_color="red",    annotation_text="High severity (30%)")
                fig.update_layout(height=320, margin=dict(t=40, b=20))
                st.plotly_chart(fig, width='content')
            else:
                st.info("No drift history yet. Predictions will accumulate drift checks automatically.")

        # Manual batch drift check
        st.divider()
        col_btn, col_window = st.columns([1, 2])
        with col_window:
            window = st.number_input("Rows to check", min_value=10, max_value=1000, value=100, step=10)
        with col_btn:
            st.write("")   # vertical alignment spacer
            if st.button("Run Batch Drift Check", width='content'):
                with st.spinner("Running drift check…"):
                    result, err = api_get(f"/monitoring/drift/check?window={window}")
                if err:
                    st.error(f"Drift check failed: {err}")
                elif result:
                    if "message" in result:
                        st.info(result["message"])
                    else:
                        alert = result.get("alert", False)
                        ds    = result.get("drift_share", 0)
                        if alert:
                            st.error(f"Drift alert! {ds*100:.1f}% of features drifted: {result.get('drifted_columns')}")
                        else:
                            st.success(f"No significant drift. Drift share: {ds*100:.1f}%")
                        with st.expander("Column-level stats"):
                            st.json(result.get("column_stats", {}))

    if auto_refresh:
        time.sleep(10)
        st.rerun()


# TAB 3 — RECENT PREDICTIONS
with tab_recent:
    st.header("Recent Predictions")

    col_n, col_filter, _ = st.columns([1, 1, 3])
    with col_n:
        n_rows = st.number_input("Show last N", min_value=5, max_value=500, value=50, step=5)
    with col_filter:
        filter_fraud = st.selectbox("Filter", ["All", "Fraud only", "Legit only"])

    preds, err = api_get(f"/monitoring/recent?limit={n_rows}")
    if err:
        st.warning(f"Could not load predictions: {err}")
    elif preds:
        df = pd.DataFrame(preds)
        if filter_fraud == "Fraud only":
            df = df[df["is_fraud"] == 1]
        elif filter_fraud == "Legit only":
            df = df[df["is_fraud"] == 0]

        def highlight_fraud(row):
            if bool(row["is_fraud"]):
                return ["background-color: #fadbd8"] * len(row)
            return [""] * len(row)

        display_cols = [c for c in [
            "predicted_at", "merchant_category", "amount", "hour",
            "day_of_week", "is_fraud", "fraud_probability",
            "feast_status", "model_version",
        ] if c in df.columns]

        st.dataframe(
            df[display_cols].style.apply(highlight_fraud, axis=1),
            width='content',
            height=500,
        )
        st.caption(f"Showing {len(df)} rows  |  🔴 = fraud flagged")
    else:
        st.info("No predictions recorded yet.")