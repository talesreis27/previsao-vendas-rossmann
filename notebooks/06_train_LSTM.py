import pandas as pd
import numpy as np
import os 
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Safe MAPE function
# This avoids division by zero when real sales are equal to 0, such as closed store days.
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Paths
processed_path = "data/processed/processed_data.csv"
model_path = "models/lstm_model.keras"
output_path = "outputs/lstm_results.csv"
graph_path = "outputs/graficos/lstm_forecast.png"
scaler_path = "models/sales_scaler.pkl"

# Check if processed dataset exist
if not os.path.exists(processed_path):
    raise FileNotFoundError("processed_data.csv not found. Run 02_process_features.py first.")

# Load processed dataset
df = pd.read_csv(processed_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Processed dataset loaded successfully!")
print(f"Total record: {len(df)}")
print(f"Initial date: {df['Date'].min()}")
print(f"Final date: {df['Date'].max()}")

# Select target variable
sales = df[["Sales"]].values

# Train/test split index for scaler fitting
# The scaler must be fitted only on the training period to avoid data leakage.
raw_split_index = int(len(df) * 0.8)

train_sales_for_scaler = sales[:raw_split_index]

# Normalize sales values between 0 and 1
scaler = MinMaxScaler()
scaler.fit(train_sales_for_scaler)

sales_scaled = scaler.transform(sales)

print("\nSales normalized successfully without data leakage.")
print(f"Minimum train sales used by scaler: {train_sales_for_scaler.min()}")
print(f"Maximum train sales used by scaler: {train_sales_for_scaler.max()}")

# Function to create sequences 
def create_sequences(data, timesteps):
    x = []
    y = []

    for i in range(len(data) - timesteps):
        x.append(data[i:i + timesteps])
        y.append(data[i + timesteps])
    return np.array(x), np.array(y)

# Use 30 previous days to predict the next day
timesteps = 30
x, y = create_sequences(sales_scaled, timesteps)

print("\nSequences created successfully!")
print(f"x shape: {x.shape}")
print(f"y shape: { y.shape}")

# Train/test split preserving temporal order
split_index = int(len(x) * 0.8)
x_train = x[:split_index]
x_test = x[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

print("\nTrain/Test split:")
print(f"x_train: {x_train.shape}")
print(f"x_test: {x_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

#Build LSTM model
model = Sequential(
    [
        LSTM(units =50, activation="relu", input_shape=(timesteps, 1)),
        Dropout(0.1),
        Dense(units=25, activation="relu"),
        Dense(units=1),
    ]
)
model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss="mean_squared_error"
)

print("\nTraining LSTM model...")
history =model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=16,
    verbose=1
)
print("LSTM model trained successfully!")

# Plot training loss
loss_graph_path = "outputs/graficos/lstm_training_loss.png"
os.makedirs("outputs/graficos", exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], color="blue")
plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(loss_graph_path)
plt.show()

print(f"LSTM training loss graph saved at: {loss_graph_path}")

# Make predictions
predictions_scaled = model.predict(x_test)

# Invert scaling to original sales unit 
predictions = scaler.inverse_transform(predictions_scaled).flatten()
y_test_real = scaler.inverse_transform(y_test).flatten()

# Calculate error metrics
mae = mean_absolute_error(y_test_real, predictions)
rmse = np.sqrt(mean_squared_error(y_test_real, predictions))
mape = safe_mape(y_test_real, predictions)

print("\nError Metrics - LSTM")
print("=" * 40)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print("=" * 40)

# Save model and results
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/graficos", exist_ok=True)
model.save(model_path)
with open(scaler_path, "wb") as file:
    pickle.dump(scaler, file)

print(f"\nLSTM model saved at: {model_path}")
print(f"Scaler saved at: {scaler_path}")

#Dates corresponding to y_test
test_dates = df["Date"].iloc[timesteps + split_index:].values
result_lstm = pd.DataFrame(
    {
        "Date": test_dates,
        "real_sales": y_test_real,
        "lstm_predictions": predictions,
    }
)
result_lstm.to_csv(output_path, index=False)

print(f"LSTM results saved at: {output_path}")

#plot real vs predicted sales
plt.figure(figsize=(14, 5))
plt.plot(result_lstm["Date"], result_lstm["real_sales"], label="Real Sales", color="black")
plt.plot(result_lstm["Date"], result_lstm["lstm_predictions"], label="LSTM Predictions", color="orange", linestyle="--")
plt.title("Real Sales vs LSTM Predictions")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(graph_path)
plt.show()

print(f"LSTM forecast graph saved at: { graph_path}")
print(f"\nFirst 10 predictions vs real values:")
print(result_lstm.head(10))
