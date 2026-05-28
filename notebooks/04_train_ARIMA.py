import pandas as pd 
import numpy as np
import os
import warnings 
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
warnings.filterwarnings("ignore")

# Safe MAPE function
# This avoids division by zero when real sales are equal to 0, such as closed store days
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Paths
train_path = "data/processed/train_data.csv"
test_path = "data/processed/test_data.csv"
output_path = "outputs/arima_results.csv"

# Check if files exist
if not os.path.exists(train_path):
    raise FileNotFoundError("train_data.csv not found. Run 03_train_test_split.py first.")
if not os.path.exists(test_path):
    raise FileNotFoundError("test_data.csv not found. Run 03_train_test_split.py first.")

# Load train and test data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Convert Date column to datetime
train_df["Date"] = pd.to_datetime(train_df["Date"])
test_df["Date"] = pd.to_datetime(test_df["Date"])

print("Train and test dataset loaded successfully!")
print(f"Train records {len(train_df)}")
print(f"Test records {len(test_df)}")

# Prepare target variable
train_sales = train_df["Sales"].values
test_sales = test_df["Sales"].values

#Train ARIMA model
# ARIMA(1,1,1) is a simple classical time series model.
print("\nTraining ARIMA(1,1,1) model...")
model = ARIMA(train_sales, order=(1,1,1))
trained_model = model.fit()
print("ARIMA model trained successfully")

# Forecast the same number of days as the test set
print("\nMaking predictions...")
predictions = trained_model.forecast(steps=len(test_sales))
print(f"Predictions generated: {len(predictions)}")

# Calculate erros metrics
mae = mean_absolute_error(test_sales, predictions)
rmse = np.sqrt(mean_squared_error(test_sales, predictions))
mape = safe_mape(test_sales, predictions)

print("\nError Metrics - ARIMA(1,1,1)")
print("=" * 40)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print("=" * 40)

# Save results
os.makedirs("outputs", exist_ok=True)
result_arima = pd.DataFrame(
    {
        "Date": test_df["Date"].values,
        "real_sales": test_sales,
        "arima_predictions": predictions,
    }
)
result_arima.to_csv(output_path, index=False)

print(f"\nARIMA results saved at: {output_path}")

# Show first predictions
print("\nFirst 10 predictions vs real values:")
print(result_arima.head(10))
