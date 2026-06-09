import streamlit as st
import pandas as pd
import plotly.express as px


def show_page():

    st.title("📊 Executive Overview")

    # =========================
    # LOAD DATA
    # =========================

    df = pd.read_parquet("/app/data/silver/retail_features.parquet")

    # =========================
    # KPIs
    # =========================

    total_revenue = df["TotalPrice"].sum()

    total_orders = df["InvoiceNo"].nunique()

    total_customers = df["CustomerID"].nunique()

    avg_order_value = total_revenue / total_orders

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "💰 Revenue",
        f"${total_revenue:,.0f}",
    )

    col2.metric(
        "🧾 Orders",
        total_orders,
    )

    col3.metric(
        "👥 Customers",
        total_customers,
    )

    col4.metric(
        "📦 Avg Order Value",
        f"${avg_order_value:,.2f}",
    )

    st.divider()

    # =========================
    # SALES TREND
    # =========================

    st.subheader("📈 Revenue Trend")

    daily_sales = (
        df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()
    )

    fig_sales = px.line(
        daily_sales,
        x="InvoiceDate",
        y="TotalPrice",
        title="Daily Revenue",
    )

    st.plotly_chart(
        fig_sales,
        use_container_width=True,
    )

    # =========================
    # COUNTRY SALES
    # =========================

    st.subheader("🌍 Revenue by Country")

    country_sales = (
        df.groupby("Country")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_country = px.bar(
        country_sales,
        x="Country",
        y="TotalPrice",
        title="Top Countries",
    )

    st.plotly_chart(
        fig_country,
        use_container_width=True,
    )

    # =========================
    # TOP PRODUCTS
    # =========================

    st.subheader("🏆 Top Products")

    top_products = (
        df.groupby("Description")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_products = px.bar(
        top_products,
        x="TotalPrice",
        y="Description",
        orientation="h",
        title="Top Products",
    )

    st.plotly_chart(
        fig_products,
        use_container_width=True,
    )

    st.success("Executive Overview Loaded Successfully")
