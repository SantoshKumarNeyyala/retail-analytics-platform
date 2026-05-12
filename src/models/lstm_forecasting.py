# ==========================================
# 🚀 ADVANCED LSTM DEMAND FORECASTING
# ==========================================

import warnings

import os
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ==========================================
# LOAD DATA
# ==========================================

print("🚀 Loading retail data...")

df = pd.read_parquet("data/silver/retail_features.parquet")

daily_sales = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()

daily_sales.columns = ["Date", "Sales"]

print(daily_sales.head())

# ==========================================
# SCALE DATA
# ==========================================

scaler = MinMaxScaler()

sales_scaled = scaler.fit_transform(daily_sales["Sales"].values.reshape(-1, 1))

# ==========================================
# CREATE SEQUENCES
# ==========================================

LOOKBACK = 28

X = []
y = []

for i in range(LOOKBACK, len(sales_scaled)):
    X.append(sales_scaled[i - LOOKBACK : i])
    y.append(sales_scaled[i])

X = np.array(X)
y = np.array(y)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ==========================================
# DATASET CLASS
# ==========================================


class SalesDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = SalesDataset(X_train, y_train)
test_dataset = SalesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ==========================================
# LSTM MODEL
# ==========================================


class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1, hidden_size=64, num_layers=2, batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


model = LSTMModel()

# ==========================================
# LOSS + OPTIMIZER
# ==========================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# MLFLOW SETUP
# ==========================================

mlflow.set_experiment("Retail Analytics")

# ==========================================
# TRAINING
# ==========================================

EPOCHS = 20

print("🚀 Starting training...")

with mlflow.start_run(run_name="Advanced LSTM Forecast"):

    mlflow.log_param("lookback", LOOKBACK)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("hidden_size", 64)

    losses = []

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        losses.append(avg_loss)

        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")

    # ==========================================
    # EVALUATION
    # ==========================================

    model.eval()

    predictions = []

    with torch.no_grad():

        for X_batch, _ in test_loader:

            pred = model(X_batch)

            predictions.extend(pred.numpy())

    predictions = np.array(predictions)

    # Inverse transform
    predictions = scaler.inverse_transform(predictions)
    y_actual = scaler.inverse_transform(y_test)

    # ==========================================
    # METRICS
    # ==========================================

    mape = mean_absolute_percentage_error(y_actual, predictions) * 100

    mlflow.log_metric("MAPE", mape)

    print(f"\n✅ Forecast MAPE: {mape:.2f}%")

    # ==========================================
    # SAVE MODEL
    # ==========================================

    os.makedirs("artifacts/models", exist_ok=True)

    torch.save(model.state_dict(), "artifacts/models/lstm_forecast.pth")

    mlflow.pytorch.log_model(model, "lstm_forecasting_model")

    # ==========================================
    # PLOT RESULTS
    # ==========================================

    os.makedirs("artifacts/plots", exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(y_actual, label="Actual Sales")
    plt.plot(predictions, label="Predicted Sales")

    plt.title("LSTM Demand Forecast")
    plt.xlabel("Time")
    plt.ylabel("Sales")

    plt.legend()

    plot_path = "artifacts/plots/lstm_forecast.png"

    plt.savefig(plot_path)

    mlflow.log_artifact(plot_path)

print("\n🎉 LSTM Forecasting Completed!")
