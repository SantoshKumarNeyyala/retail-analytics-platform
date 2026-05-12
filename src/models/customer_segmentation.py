# ==========================================
# 🚀 CUSTOMER SEGMENTATION PIPELINE
# KMEANS + DBSCAN + GMM
# ==========================================

import warnings

import os
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.filterwarnings("ignore")


# ==========================================
# LOAD DATA
# ==========================================

print("🚀 Loading RFM data...")

rfm = pd.read_parquet("data/silver/rfm.parquet")

print(rfm.head())

# ==========================================
# FEATURES
# ==========================================

X = rfm[["Recency", "Frequency", "Monetary"]]

# ==========================================
# SCALING
# ==========================================

print("\n⚙️ Scaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# CREATE ARTIFACT FOLDERS
# ==========================================

os.makedirs("artifacts/segmentation", exist_ok=True)

# ==========================================
# SETUP MLFLOW
# ==========================================

mlflow.set_experiment("Retail Analytics")

# ==========================================
# KMEANS CLUSTERING
# ==========================================

with mlflow.start_run(run_name="Customer Segmentation"):

    print("\n🚀 Running KMeans clustering...")

    silhouette_scores = []
    db_scores = []

    k_values = range(2, 11)

    for k in k_values:

        model = KMeans(n_clusters=k, random_state=42, n_init=10)

        labels = model.fit_predict(X_scaled)

        sil_score = silhouette_score(X_scaled, labels)

        db_score = davies_bouldin_score(X_scaled, labels)

        silhouette_scores.append(sil_score)
        db_scores.append(db_score)

        print(
            f"K={k} | "
            f"Silhouette={sil_score:.4f} | "
            f"Davies-Bouldin={db_score:.4f}"
        )

    # ==========================================
    # SELECT BEST K
    # ==========================================

    best_k = k_values[np.argmax(silhouette_scores)]

    print(f"\n🏆 Best K Selected: {best_k}")

    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)

    rfm["Cluster"] = final_kmeans.fit_predict(X_scaled)

    # ==========================================
    # LOG METRICS
    # ==========================================

    final_silhouette = silhouette_score(X_scaled, rfm["Cluster"])

    final_db = davies_bouldin_score(X_scaled, rfm["Cluster"])

    mlflow.log_metric("silhouette_score", final_silhouette)

    mlflow.log_metric("davies_bouldin_score", final_db)

    print(f"✅ Final Silhouette Score: {final_silhouette:.4f}")

    # ==========================================
    # SAVE MODEL
    # ==========================================

    mlflow.sklearn.log_model(final_kmeans, "kmeans_segmentation_model")

    # ==========================================
    # CLUSTER VISUALIZATION
    # ==========================================

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Cluster"])

    plt.xlabel("Frequency")
    plt.ylabel("Monetary")

    plt.title("Customer Segments")

    plt.colorbar(scatter)

    cluster_plot_path = "artifacts/segmentation/customer_segments.png"

    plt.savefig(cluster_plot_path)

    mlflow.log_artifact(cluster_plot_path)

    plt.close()

    # ==========================================
    # CLUSTER PROFILING
    # ==========================================

    cluster_profile = rfm.groupby("Cluster")[
        ["Recency", "Frequency", "Monetary"]
    ].mean()

    print("\n📊 CUSTOMER SEGMENT PROFILES")
    print(cluster_profile)

    profile_path = "artifacts/segmentation/cluster_profiles.csv"

    cluster_profile.to_csv(profile_path)

    mlflow.log_artifact(profile_path)

    # ==========================================
    # DBSCAN
    # ==========================================

    print("\n🚀 Running DBSCAN...")

    dbscan = DBSCAN(eps=0.8, min_samples=10)

    dbscan_labels = dbscan.fit_predict(X_scaled)

    unique_clusters = len(set(dbscan_labels))

    print(f"✅ DBSCAN Clusters: {unique_clusters}")

    # ==========================================
    # GMM
    # ==========================================

    print("\n🚀 Running Gaussian Mixture Model...")

    gmm = GaussianMixture(n_components=best_k, random_state=42)

    gmm.fit(X_scaled)

    gmm_labels = gmm.predict(X_scaled)

    print("✅ GMM Completed")

print("\n🎉 Customer Segmentation Completed!")
