from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score
import numpy as np

# Dummy regression validation
y_true_reg = np.array([100, 200, 300, 400])
y_pred_reg = np.array([110, 190, 310, 390])

mape = mean_absolute_percentage_error(y_true_reg, y_pred_reg)

print(f"MAPE: {mape:.4f}")

if mape > 0.12:
    raise Exception("Model validation failed: MAPE exceeded 12%")

# Dummy classification validation
y_true_cls = np.array([0, 1, 1, 0, 1])
y_pred_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.95])

auc = roc_auc_score(y_true_cls, y_pred_prob)

print(f"AUC: {auc:.4f}")

if auc < 0.88:
    raise Exception("Model validation failed: AUC below 0.88")

print("Model validation passed")
