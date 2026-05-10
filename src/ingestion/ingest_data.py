import pandas as pd
import os


def ingest():
    print("Starting ingestion...")

    input_path = "data/raw/Online Retail.xlsx"
    output_path = "data/bronze/retail_data.parquet"

    # Read Excel
    df = pd.read_excel(input_path)

    print("Data Preview:")
    print(df.head())

    # Fix data types
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype(str)
    df["Country"] = df["Country"].astype(str)

    # Create bronze folder if not exists
    os.makedirs("data/bronze", exist_ok=True)

    # Save as parquet
    df.to_parquet(output_path, index=False)

    print(f"✅ Data successfully saved to {output_path}")


if __name__ == "__main__":
    ingest()
