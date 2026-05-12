import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


def show_page():

    st.title("👥 Customer Hub")

    # ==========================================
    # LOAD DATA
    # ==========================================

    rfm = pd.read_parquet("data/silver/rfm.parquet")

    # ==========================================
    # MOCK CHURN
    # ==========================================

    rfm["ChurnRisk"] = np.where(
        rfm["Recency"] > 90, "High", np.where(rfm["Recency"] > 45, "Medium", "Low")
    )

    # ==========================================
    # CUSTOMER SEGMENTS
    # ==========================================

    rfm["Segment"] = pd.qcut(
        rfm["Monetary"], q=4, labels=["Bronze", "Silver", "Gold", "Platinum"]
    )

    # ==========================================
    # FILTER
    # ==========================================

    segment = st.sidebar.selectbox("Select Segment", rfm["Segment"].unique())

    filtered = rfm[rfm["Segment"] == segment]

    # ==========================================
    # KPI
    # ==========================================

    col1, col2, col3 = st.columns(3)

    col1.metric("Customers", filtered["CustomerID"].nunique())

    col2.metric("Avg Monetary", f"${filtered['Monetary'].mean():,.2f}")

    col3.metric("Avg Frequency", round(filtered["Frequency"].mean(), 2))

    st.divider()

    # ==========================================
    # CHURN DISTRIBUTION
    # ==========================================

    st.subheader("⚠️ Churn Risk Distribution")

    fig = px.histogram(
        filtered,
        x="ChurnRisk",
        color="ChurnRisk",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # RFM SCATTER
    # ==========================================

    st.subheader("📊 Customer Segmentation")

    fig2 = px.scatter(
        filtered,
        x="Frequency",
        y="Monetary",
        color="ChurnRisk",
        size="Recency",
        hover_data=["CustomerID"],
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # CUSTOMER 360
    # ==========================================

    st.subheader("🧾 Customer 360 View")

    customer = st.selectbox(
        "Select Customer", filtered["CustomerID"].astype(str).unique()
    )

    customer_data = filtered[filtered["CustomerID"].astype(str) == customer]

    st.dataframe(customer_data)

    # ==========================================
    # RETENTION ACTIONS
    # ==========================================

    st.subheader("🎯 Retention Recommendation")

    risk = customer_data["ChurnRisk"].values[0]

    if risk == "High":
        st.error("Offer 25% discount + loyalty rewards immediately.")

    elif risk == "Medium":
        st.warning("Send personalized email campaign.")

    else:
        st.success("Customer is healthy and loyal.")

    # ==========================================
    # EXPORT
    # ==========================================

    st.subheader("📥 CRM Export")

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Segment CSV",
        data=csv,
        file_name="customer_segment.csv",
        mime="text/csv",
    )

    st.success("✅ Customer Hub Loaded")
