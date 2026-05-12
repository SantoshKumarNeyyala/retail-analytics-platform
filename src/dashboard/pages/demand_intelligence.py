import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def show_page():

    st.title("📈 Demand Intelligence")

    # ==========================================
    # LOAD DATA
    # ==========================================

    df = pd.read_parquet("data/silver/retail_features.parquet")

    # ==========================================
    # FILTERS
    # ==========================================

    st.sidebar.header("Forecast Filters")

    countries = df["Country"].dropna().unique()

    selected_country = st.sidebar.selectbox("Select Country", sorted(countries))

    filtered_df = df[df["Country"] == selected_country]

    # ==========================================
    # DAILY SALES
    # ==========================================

    daily_sales = (
        filtered_df.groupby(filtered_df["InvoiceDate"].dt.date)["TotalPrice"]
        .sum()
        .reset_index()
    )

    daily_sales.columns = ["Date", "Actual"]

    # ==========================================
    # MOCK FORECAST
    # ==========================================

    np.random.seed(42)

    daily_sales["Forecast"] = daily_sales["Actual"] * np.random.uniform(
        0.9, 1.1, len(daily_sales)
    )

    daily_sales["Upper"] = daily_sales["Forecast"] * 1.10
    daily_sales["Lower"] = daily_sales["Forecast"] * 0.90

    # ==========================================
    # FORECAST CHART
    # ==========================================

    fig = go.Figure()

    # Actual

    fig.add_trace(
        go.Scatter(
            x=daily_sales["Date"],
            y=daily_sales["Actual"],
            mode="lines",
            name="Actual Sales",
        )
    )

    # Forecast

    fig.add_trace(
        go.Scatter(
            x=daily_sales["Date"],
            y=daily_sales["Forecast"],
            mode="lines",
            name="Forecast",
        )
    )

    # Upper Bound

    fig.add_trace(
        go.Scatter(
            x=daily_sales["Date"],
            y=daily_sales["Upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )

    # Lower Bound + fill

    fig.add_trace(
        go.Scatter(
            x=daily_sales["Date"],
            y=daily_sales["Lower"],
            mode="lines",
            fill="tonexty",
            name="Confidence Interval",
            line=dict(width=0),
        )
    )

    fig.update_layout(
        title="Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Revenue",
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # WHAT-IF SIMULATOR
    # ==========================================

    st.subheader("🧠 What-If Simulator")

    promo_boost = st.slider("Promotion Impact %", 0, 100, 10)

    simulated_forecast = daily_sales["Forecast"] * (1 + promo_boost / 100)

    st.line_chart(simulated_forecast)

    # ==========================================
    # FORECAST TABLE
    # ==========================================

    st.subheader("📄 Forecast Table")

    st.dataframe(daily_sales.head(20))

    st.success("✅ Demand Intelligence Loaded")
