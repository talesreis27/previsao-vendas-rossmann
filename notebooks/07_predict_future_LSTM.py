import pandas as pd
import numpy as np
import os 
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Paths
processed_path = "data/processed/processed_data.csv"
model_path = "models/lstm_model.keras"
output_path = "outputs/lstm_future_forecast.csv"
graph_path = "outputs/graficos/lstm_future_forecast_until_december.png"
scaler_path = "models/sales_scaler.pkl"

# Check if required files exist
if not os.path.exists(processed_path):
    raise FileNotFoundError("processed_data.csv not found. Run 02_process_features.py first.")
if not os.path.exists(model_path): 
    raise FileNotFoundError("lstm_model.keras not found. Run 06_train_lstm.py first.")
if not os.path.exists(scaler_path):
    raise FileNotFoundError("sales_scaler.pkl not found. Run 06_train_LSTM.py first.")

# Load processed dataset
df = pd.read_csv(processed_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("Processed dataset loaded successfully!")
print(f"Total records: {len(df)}")
print(f"Last available date: {df['Date'].max()}")

# Load train LSTM model
model = load_model(model_path)
print("LSTM model loaded successfully!")

# Prepare sales data
sales = df[["Sales"]].values

# Load scaler fitted only on training data
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

sales_scaled = scaler.transform(sales)

# Forecast configuration
timesteps = 30
future_days = 153

# Get the last 30 scaled sales values 
last_sequence = sales_scaled[-timesteps:].copy()
future_predictions_scaled = []

# Recursive prediction 
for day in range(future_days):
    # Reshape to model input format: (samples , timesteps, features)
    input_sequence = last_sequence.reshape(1, timesteps, 1)
    # Predict next day
    next_prediction_scaled = model.predict(input_sequence, verbose=0)[0][0]
    # Save prediction
    future_predictions_scaled.append(next_prediction_scaled)
    # Update sequence: remove oldest value and add new prediction
    last_sequence = np.append(last_sequence[1:], [[next_prediction_scaled]], axis=0)

# Convert predictions back to original sales scale
future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions_scaled).flatten()

# Create future dates 
last_date = df["Date"].max()
future_dates = pd.date_range( 
    start=last_date + pd.Timedelta(days=1), 
    periods=future_days,
    freq="D"
)

# Create forecast dataframe
future_df = pd.DataFrame(
    { 
        "Date": future_dates,
        "lstm_future_prediction": future_predictions,
    }
)

# Create output folders
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/graficos", exist_ok=True)

# Save full forecast
future_df.to_csv(output_path,index=False)

print(f"\nFuture forecast saved at: {output_path}")
print("\nFirst future predictions:")
print(future_df.head(10))

# Filter November and December forecast
future_df["month"] = future_df["Date"].dt.month
end_of_year_forecast = future_df[
    future_df["month"].isin([11, 12])
].copy()
end_of_year_path = "outputs/lstm_forecast_nov_dec_2015.csv"
end_of_year_forecast.to_csv(end_of_year_path, index=False)

print(f"\nNovember and December forecast saved at: {end_of_year_path}")
print("\nNovember and December forecast: ")
print(end_of_year_forecast)
print("\nEnd of the year Forecast summary:")
print(f"Average predicted sales: {end_of_year_forecast['lstm_future_prediction'].mean():.2f}")
print(f"Minimum predicted sales: {end_of_year_forecast['lstm_future_prediction'].min():.2f}")
print(f"Maximum predicted sales: {end_of_year_forecast['lstm_future_prediction'].max():.2f}")

monthly_totals = end_of_year_forecast.groupby(
    end_of_year_forecast["Date"].dt.month
)["lstm_future_prediction"].sum()

print("\nPredicted total sales by month:")
print(monthly_totals)

# Plot last historical sales and future forecast
historical_tail = df.tail(90)

plt.figure(figsize=(14, 5))
plt.plot( 
    historical_tail["Date"],
    historical_tail["Sales"],
    label="Historical Sales",
    color="black",
)
plt.plot(
    future_df["Date"],
    future_df["lstm_future_prediction"],
    label="LSTM Future Forecast",
    color="orange",
    linestyle="--",
)
plt.title("LSTM Future Sales Forecast - August to December 2015")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(graph_path)
plt.show()

print(f"Future forecast graph saved at: {graph_path}")

#Summary 
print("\nForecast summary:")
print(f"Forecast period: {future_df['Date'].min()} to {future_df['Date'].max()}")
print(f"Average predicted sales: {future_df['lstm_future_prediction'].mean():.2f}")
print(f"Minimum predicted sales: {future_df['lstm_future_prediction'].min():.2f}")
print(f"Maximum predicted sales: {future_df['lstm_future_prediction'].max():.2f}")



