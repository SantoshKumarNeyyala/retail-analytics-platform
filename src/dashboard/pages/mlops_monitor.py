import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


def show_page():

    # ==========================================
    # PAGE CONFIG
    # ==========================================

    st.title("⚙️ MLOps Monitoring Dashboard")

    st.markdown(
        """
        Monitor model health, drift detection,
        forecast stability, and production ML metrics.
        """
    )

    # ==========================================
    # KPI SECTION
    # ==========================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Drift Status",
        "Detected",
        delta="PSI > 0.2",
    )

    col2.metric(
        "Forecast MAPE",
        "8.4%",
        delta="-1.2%",
    )

    col3.metric(
        "Model Version",
        "v2.0",
    )

    col4.metric(
        "Last Retrain",
        "2 days ago",
    )

    st.divider()

    # ==========================================
    # DRIFT TREND
    # ==========================================

    st.subheader("📈 Drift Monitoring Trend")

    drift_df = pd.DataFrame(
        {
            "Day": [
                "Mon",
                "Tue",
                "Wed",
                "Thu",
                "Fri",
                "Sat",
                "Sun",
            ],
            "PSI": [
                0.05,
                0.08,
                0.11,
                0.14,
                0.18,
                0.24,
                0.27,
            ],
        }
    )

    fig = px.line(
        drift_df,
        x="Day",
        y="PSI",
        markers=True,
        title="Population Stability Index (PSI)",
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    # ==========================================
    # FEATURE DRIFT TABLE
    # ==========================================

    st.subheader("🧪 Feature Drift Analysis")

    feature_drift = pd.DataFrame(
        {
            "Feature": [
                "Quantity",
                "UnitPrice",
                "Recency",
                "Frequency",
                "Monetary",
            ],
            "PSI Score": [
                0.31,
                0.09,
                0.14,
                0.22,
                0.07,
            ],
            "Status": [
                "High Drift",
                "Stable",
                "Moderate",
                "Moderate",
                "Stable",
            ],
        }
    )

    st.dataframe(
        feature_drift,
        width="stretch",
    )

    # ==========================================
    # DRIFT REPORT LINK
    # ==========================================

    st.subheader("📄 Evidently Drift Report")

    report_path = Path("artifacts/drift_report.html")

    if report_path.exists():

        with open(
            report_path,
            "rb",
        ) as file:

            st.download_button(
                label="⬇️ Download Drift Report",
                data=file,
                file_name="drift_report.html",
                mime="text/html",
            )

        st.success("Drift report available")

    else:

        st.warning("Drift report not found")

    # ==========================================
    # ALERT SECTION
    # ==========================================

    st.subheader("🚨 Active Alerts")

    st.error("Quantity feature drift exceeded PSI threshold (0.2)")

    st.warning("Forecast variance increased by 11%")

    st.success("Retraining pipeline operational")
