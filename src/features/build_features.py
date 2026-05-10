import pandas as pd
import os


def build_features():
    print("Starting feature engineering...")

    input_path = "data/bronze/retail_data.parquet"
    output_path = "data/silver/retail_features.parquet"

    # Load data
    df = pd.read_parquet(input_path)

    print("Original shape:", df.shape)

    # ----------------------------
    # 🧹 Data Cleaning
    # ----------------------------

    # Remove missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Remove negative or zero quantity
    df = df[df["Quantity"] > 0]

    # Remove negative price
    df = df[df["UnitPrice"] > 0]

    # Remove cancelled invoices (start with 'C')
    df = df[~df["InvoiceNo"].str.startswith("C")]

    print("After cleaning:", df.shape)

    # ----------------------------
    # 🧠 Feature Engineering
    # ----------------------------

    # Total price
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Convert date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Date features
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["Month"] = df["InvoiceDate"].dt.month
    df["Hour"] = df["InvoiceDate"].dt.hour

    # ----------------------------
    # 💰 RFM Features
    # ----------------------------

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            {
                "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # Recency
                "InvoiceNo": "nunique",  # Frequency
                "TotalPrice": "sum",  # Monetary
            }
        )
        .reset_index()
    )

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    print("RFM shape:", rfm.shape)

    # ----------------------------
    # 💾 Save
    # ----------------------------

    os.makedirs("data/silver", exist_ok=True)

    df.to_parquet(output_path, index=False)
    rfm.to_parquet("data/silver/rfm.parquet", index=False)

    print(f"✅ Features saved to {output_path}")
    print("✅ RFM saved to data/silver/rfm.parquet")


if __name__ == "__main__":
    build_features()
