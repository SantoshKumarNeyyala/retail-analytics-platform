import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Retail Analytics Platform", layout="wide")

st.title("🛒 Enterprise Retail Analytics Platform")

# =========================
# LOAD DATA
# =========================
df = pd.read_parquet("data/silver/retail_features.parquet")
rfm = pd.read_parquet("data/silver/rfm.parquet")

# =========================
# KPI SECTION
# =========================
total_revenue = df["TotalPrice"].sum()
total_orders = df["InvoiceNo"].nunique()
total_customers = df["CustomerID"].nunique()
avg_order_value = total_revenue / total_orders

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
col2.metric("🧾 Orders", total_orders)
col3.metric("👥 Customers", total_customers)
col4.metric("📦 Avg Order Value", f"${avg_order_value:,.2f}")

st.divider()

# =========================
# SALES TREND
# =========================
st.subheader("📈 Daily Sales Trend")

daily_sales = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()

fig_sales = px.line(daily_sales, x="InvoiceDate", y="TotalPrice", title="Daily Revenue")

st.plotly_chart(fig_sales, use_container_width=True)

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
    title="Top 10 Products",
)

st.plotly_chart(fig_products, use_container_width=True)

# =========================
# COUNTRY REVENUE
# =========================
st.subheader("🌍 Revenue by Country")

country_sales = (
    df.groupby("Country")["TotalPrice"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_country = px.pie(
    country_sales, names="Country", values="TotalPrice", title="Country-wise Revenue"
)

st.plotly_chart(fig_country, use_container_width=True)

# =========================
# CHURN DISTRIBUTION
# =========================
st.subheader("⚠️ Customer Churn Distribution")

rfm["Churn"] = rfm["Recency"].apply(lambda x: 1 if x > 90 else 0)

fig_churn = px.histogram(rfm, x="Churn", title="Churn Distribution")

st.plotly_chart(fig_churn, use_container_width=True)

# =========================
# DATA PREVIEW
# =========================
st.subheader("📄 Dataset Preview")

st.dataframe(df.head(20))

st.success("✅ Dashboard Loaded Successfully")
