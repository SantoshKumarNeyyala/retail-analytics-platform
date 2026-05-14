import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


# ==========================================
# LOAD DATA
# ==========================================

reference_data = pd.read_parquet("data/silver/retail_features.parquet")

current_data = reference_data.sample(
    frac=0.30,
    random_state=42,
).copy()

# Simulate drift
current_data["Quantity"] = current_data["Quantity"] * 1.5

# ==========================================
# CREATE REPORT
# ==========================================

report = Report(
    [
        DataDriftPreset(),
    ]
)

# ==========================================
# RUN REPORT
# ==========================================

my_eval = report.run(
    current_data=current_data,
    reference_data=reference_data,
)

# ==========================================
# SAVE REPORT
# ==========================================

my_eval.save_html("artifacts/drift_report.html")

print("✅ Drift report generated successfully")
