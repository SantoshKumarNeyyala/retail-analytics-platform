import streamlit as st
import pandas as pd
import plotly.express as px


def show_page():

    st.title("📦 Inventory Monitor")

    # =====================================
    # LOAD DATA
    # =====================================

    df = pd.read_parquet("data/silver/retail_features.parquet")

    # =====================================
    # INVENTORY SIMULATION
    # =====================================

    inventory = df.groupby("Description")["Quantity"].sum().reset_index()

    inventory.columns = [
        "Product",
        "Stock",
    ]

    inventory["SafetyStock"] = 50

    inventory["ReorderNeeded"] = inventory["Stock"] < inventory["SafetyStock"]

    # =====================================
    # KPI SECTION
    # =====================================

    total_products = len(inventory)

    low_stock = len(inventory[inventory["ReorderNeeded"]])

    avg_stock = inventory["Stock"].mean()

    total_inventory = inventory["Stock"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📦 Products",
        total_products,
    )

    col2.metric(
        "⚠️ Low Stock",
        low_stock,
    )

    col3.metric(
        "📊 Avg Stock",
        f"{avg_stock:.0f}",
    )

    col4.metric(
        "🏭 Total Inventory",
        f"{total_inventory:.0f}",
    )

    st.divider()

    # =====================================
    # LOW STOCK ALERTS
    # =====================================

    st.subheader("🚨 Reorder Alerts")

    alerts = inventory[inventory["ReorderNeeded"]]

    st.dataframe(
        alerts,
        use_container_width=True,
    )

    # =====================================
    # TOP INVENTORY
    # =====================================

    st.subheader("🏆 Highest Stock Products")

    top_inventory = inventory.sort_values(
        by="Stock",
        ascending=False,
    ).head(10)

    fig = px.bar(
        top_inventory,
        x="Stock",
        y="Product",
        orientation="h",
        title="Top Inventory Products",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # =====================================
    # EOQ CALCULATOR
    # =====================================

    st.subheader("🧮 EOQ Calculator")

    demand = st.number_input(
        "Annual Demand",
        value=1000,
    )

    ordering_cost = st.number_input(
        "Ordering Cost",
        value=50,
    )

    holding_cost = st.number_input(
        "Holding Cost",
        value=5,
    )

    eoq = ((2 * demand * ordering_cost) / holding_cost) ** 0.5

    st.metric(
        "Recommended EOQ",
        f"{eoq:.2f}",
    )

    # =====================================
    # OVERSTOCK ANALYSIS
    # =====================================

    st.subheader("📉 Overstock Risk")

    overstock = inventory[inventory["Stock"] > inventory["Stock"].quantile(0.90)]

    st.dataframe(
        overstock,
        use_container_width=True,
    )

    st.success("Inventory Monitor Loaded")
