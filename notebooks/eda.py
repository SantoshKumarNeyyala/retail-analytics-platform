# =========================
# 📊 EDA SCRIPT (FULL)
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("🚀 Starting EDA...")

# -------------------------
# 📥 Load Data
# -------------------------
df = pd.read_parquet("data/silver/retail_features.parquet")
rfm = pd.read_parquet("data/silver/rfm.parquet")

print("✅ Data Loaded")

# -------------------------
# 🔍 Basic Info
# -------------------------
print("\nShape:", df.shape)
print("\nColumns:", df.columns)

print("\nMissing Values:\n", df.isnull().sum())

print("\nSummary:\n", df.describe())

# -------------------------
# 📈 Sales by Month
# -------------------------
monthly_sales = df.groupby("Month")["TotalPrice"].sum()

plt.figure()
monthly_sales.plot(kind="bar")
plt.title("Sales by Month")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.show()

print("\n📌 Insight: Identify peak sales months")

# -------------------------
# 🕒 Sales by Hour
# -------------------------
hourly_sales = df.groupby("Hour")["TotalPrice"].sum()

plt.figure()
hourly_sales.plot(kind="line")
plt.title("Sales by Hour")
plt.xlabel("Hour")
plt.ylabel("Revenue")
plt.show()

print("\n📌 Insight: Identify peak shopping hours")

# -------------------------
# 🌍 Top Countries
# -------------------------
top_countries = (
    df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
)

plt.figure()
top_countries.plot(kind="barh")
plt.title("Top Countries by Revenue")
plt.xlabel("Revenue")
plt.show()

print("\n📌 Insight: Identify top markets")

# -------------------------
# 🛍️ Top Products
# -------------------------
top_products = (
    df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(10)
)

plt.figure()
top_products.plot(kind="barh")
plt.title("Top Products by Revenue")
plt.xlabel("Revenue")
plt.show()

print("\n📌 Insight: Identify best-selling products")

# -------------------------
# 👤 RFM Distribution
# -------------------------
plt.figure()
sns.histplot(rfm["Monetary"], bins=50)
plt.title("Customer Spending Distribution")
plt.xlabel("Spending")
plt.show()

print("\n📌 Insight: Understand customer spending behavior")

# -------------------------
# ⭐ Customer Segmentation
# -------------------------
rfm["Segment"] = pd.qcut(rfm["Monetary"], 3, labels=["Low", "Medium", "High"])

plt.figure()
rfm["Segment"].value_counts().plot(kind="bar")
plt.title("Customer Segments")
plt.xlabel("Segment")
plt.ylabel("Count")
plt.show()

print("\n📌 Insight: Customer segmentation complete")

print("\n✅ EDA COMPLETED SUCCESSFULLY")
