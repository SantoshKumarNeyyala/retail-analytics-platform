# 🛒 Enterprise Retail Analytics Platform

## 📌 Overview

An enterprise-grade retail analytics platform for demand forecasting, customer churn prediction, and business intelligence.

The platform includes:
- Data ingestion pipelines
- Feature engineering
- Machine learning models
- MLflow experiment tracking
- FastAPI serving
- Streamlit dashboards
- Dockerized infrastructure
- Airflow orchestration

---

# 🚀 Features

## 📥 Data Engineering
- CSV/Parquet ingestion pipelines
- Bronze/Silver layered architecture
- Feature engineering pipelines
- RFM customer analytics

## 🤖 Machine Learning
- Demand forecasting model
- Customer churn prediction
- MLflow experiment tracking
- Model versioning

## 📊 Visualization
- Interactive Streamlit dashboard
- KPI monitoring
- Sales analytics
- Country-wise revenue insights

## ⚡ Serving
- FastAPI prediction endpoints
- Swagger API documentation

## 🐳 Infrastructure
- Docker Compose setup
- PostgreSQL
- Redis
- MLflow
- Airflow orchestration

---

# 🧱 Architecture

```text
Data Sources
    ↓
Ingestion Pipeline
    ↓
Bronze Layer
    ↓
Silver Layer
    ↓
Feature Engineering
    ↓
ML Models
    ↓
MLflow Tracking
    ↓
FastAPI
    ↓
Streamlit Dashboard
```

---

# 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- MLflow
- FastAPI
- Streamlit
- PostgreSQL
- Redis
- Docker
- Airflow

---

# 📊 ML Models

## Demand Forecasting
- Linear Regression
- Lag Features
- Rolling Statistics

## Customer Churn Prediction
- Logistic Regression
- RFM Analytics

---

# 📸 Screenshots

## MLflow
(Add Screenshot)

## Dashboard
(Add Screenshot)

## Airflow DAG
(Add Screenshot)

## FastAPI Swagger
(Add Screenshot)

---

# ▶️ Run Project

## Install dependencies

```bash
poetry install
```

## Run MLflow

```bash
poetry run mlflow ui
```

## Run API

```bash
poetry run uvicorn src.api.main:app --reload
```

## Run Dashboard

```bash
poetry run streamlit run src/dashboard/app.py
```

---

# 👨‍💻 Author

Santosh Kumar Neyyala