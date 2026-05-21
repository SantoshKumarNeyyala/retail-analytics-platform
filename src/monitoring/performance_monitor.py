import pandas as pd

predictions = pd.read_csv("artifacts/predictions/stacking_predictions.csv")

mape = abs(predictions["Actual"] - predictions["Predicted"]).mean()

print("\n📊 MODEL PERFORMANCE")
print(f"MAPE: {round(mape, 2)}")
