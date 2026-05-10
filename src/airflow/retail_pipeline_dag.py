from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


def ingest_bronze():
    print("📥 Ingesting raw retail data into Bronze Layer")


def validate_data():
    import pandas as pd

    print("🔍 Running Data Quality Checks")

    df = pd.read_parquet("data/silver/retail_features.parquet")

    assert df.isnull().sum().sum() < 1000, "Too many null values detected"

    assert (df["Quantity"] > 0).all(), "Negative quantity found"

    assert (df["UnitPrice"] >= 0).all(), "Negative prices found"

    print("✅ Data validation passed")


def feature_engineering():
    print("⚙️ Creating RFM, lag, rolling, and date features")


def train_models():
    import os

    print("🤖 Running ML Training Pipeline")

    os.system("poetry run python src/models/train_models.py")

    print("✅ Models trained successfully")


def notify_success():
    print("📢 Pipeline completed successfully")


default_args = {
    "owner": "Santosh",
    "start_date": datetime(2026, 5, 1),
}


with DAG(
    dag_id="retail_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["retail", "ml", "analytics"],
) as dag:

    task1 = PythonOperator(
        task_id="ingest_bronze",
        python_callable=ingest_bronze,
    )

    task2 = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    task3 = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    task4 = PythonOperator(
        task_id="train_models",
        python_callable=train_models,
    )

    task5 = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
    )

    task1 >> task2 >> task3 >> task4 >> task5
