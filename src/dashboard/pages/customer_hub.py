import streamlit as st
import pandas as pd
import plotly.express as px


def show_page():

    st.title("👥 Customer Hub")

    # =====================================
    # LOAD DATA
    # =====================================

    rfm = pd.read_parquet("data/silver/rfm.parquet")

    # =====================================
    # CREATE CHURN FLAG
    # =====================================

    rfm["ChurnRisk"] = rfm["Recency"].apply(
        lambda x: "High Risk" if x > 90 else "Low Risk"
    )

    # =====================================
    # KPI SECTION
    # =====================================

    total_customers = len(rfm)

    high_risk = len(rfm[rfm["ChurnRisk"] == "High Risk"])

    avg_monetary = rfm["Monetary"].mean()

    avg_frequency = rfm["Frequency"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Customers",
        total_customers,
    )

    col2.metric(
        "⚠️ High Risk",
        high_risk,
    )

    col3.metric(
        "💰 Avg Monetary",
        f"${avg_monetary:.2f}",
    )

    col4.metric(
        "🛒 Avg Frequency",
        f"{avg_frequency:.2f}",
    )

    st.divider()

    # =====================================
    # CHURN DISTRIBUTION
    # =====================================

    st.subheader("⚠️ Churn Risk Distribution")

    churn_fig = px.histogram(
        rfm,
        x="ChurnRisk",
        color="ChurnRisk",
        title="Customer Churn Risk",
    )

    st.plotly_chart(
        churn_fig,
        use_container_width=True,
    )

    # =====================================
    # RFM SCATTER
    # =====================================

    st.subheader("📊 RFM Customer Segments")

    scatter_fig = px.scatter(
        rfm,
        x="Frequency",
        y="Monetary",
        color="Recency",
        hover_data=["CustomerID"],
        title="Customer Segmentation",
    )

    st.plotly_chart(
        scatter_fig,
        use_container_width=True,
    )

    # =====================================
    # CUSTOMER SELECTOR
    # =====================================

    st.subheader("🔍 Customer 360 View")

    customer_ids = rfm["CustomerID"].astype(str).unique()

    selected_customer = st.selectbox(
        "Select Customer",
        customer_ids,
    )

    customer_data = rfm[rfm["CustomerID"].astype(str) == selected_customer]

    st.dataframe(
        customer_data,
        use_container_width=True,
    )

    # =====================================
    # RETENTION ACTIONS
    # =====================================

    st.subheader("🎯 Retention Recommendations")

    if customer_data["Recency"].values[0] > 90:

        st.error("High churn risk detected")

        st.write(
            """
            Recommended Actions:
            - Send discount coupon
            - Trigger email campaign
            - Offer loyalty rewards
            """
        )

    else:

        st.success("Customer engagement healthy")

        st.write(
            """
            Recommended Actions:
            - Upsell premium products
            - Recommend bundles
            - Continue loyalty program
            """
        )

    st.success("Customer Hub Loaded Successfully")
