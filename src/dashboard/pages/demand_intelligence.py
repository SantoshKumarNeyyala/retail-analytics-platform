import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def show_page():

    st.title("📈 Demand Intelligence")

    # =====================================
    # LOAD DATA
    # =====================================

    df = pd.read_parquet("/app/data/silver/retail_features.parquet")

    # =====================================
    # FILTERS
    # =====================================

    st.sidebar.subheader("Forecast Filters")

    countries = sorted(df["Country"].dropna().unique())

    selected_country = st.sidebar.selectbox(
        "Select Country",
        countries,
    )

    filtered_df = df[df["Country"] == selected_country]

    products = sorted(filtered_df["Description"].dropna().unique())

    selected_product = st.sidebar.selectbox(
        "Select Product",
        products,
    )

    product_df = filtered_df[filtered_df["Description"] == selected_product]

    # =====================================
    # DAILY SALES
    # =====================================

    daily_sales = (
        product_df.groupby(product_df["InvoiceDate"].dt.date)["Quantity"]
        .sum()
        .reset_index()
    )

    daily_sales.columns = [
        "Date",
        "Actual Sales",
    ]

    # =====================================
    # SIMPLE FORECAST
    # =====================================

    daily_sales["Forecast"] = daily_sales["Actual Sales"].rolling(3).mean()

    daily_sales["Forecast"] = daily_sales["Forecast"].fillna(
        daily_sales["Actual Sales"]
    )

    # =====================================
    # FORECAST CHART
    # =====================================

    st.subheader("📊 Forecast vs Actual")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily_sales["Date"],
            y=daily_sales["Actual Sales"],
            mode="lines",
            name="Actual",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily_sales["Date"],
            y=daily_sales["Forecast"],
            mode="lines",
            name="Forecast",
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # =====================================
    # WHAT IF SIMULATOR
    # =====================================

    st.subheader("🧪 What-If Simulator")

    price_change = st.slider(
        "Price Change %",
        -50,
        50,
        0,
    )

    promo = st.checkbox("Promotion Enabled")

    base_forecast = daily_sales["Forecast"].mean()

    adjusted_forecast = base_forecast * (1 - (price_change / 100) * 0.5)

    if promo:

        adjusted_forecast *= 1.2

    st.metric(
        "Predicted Demand",
        f"{adjusted_forecast:.2f}",
    )

    # =====================================
    # LEADERBOARD
    # =====================================

    st.subheader("🏆 Forecast Accuracy Leaderboard")

    leaderboard = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    leaderboard.columns = [
        "Product",
        "Sales Volume",
    ]

    st.dataframe(
        leaderboard,
        use_container_width=True,
    )

    st.success("Demand Intelligence Loaded")
