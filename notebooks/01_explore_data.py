import pandas as pd
import matplotlib.pyplot as plt
import os


# Paths 
train_path = "data/raw/train.csv"
output_path = "outputs/graficos"

# Create output folder if it does not exist
os.makedirs(output_path, exist_ok=True)

# Load dataset
df = pd.read_csv(train_path, dtype={"StateHoliday": str}, low_memory=False)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

#select the store
store_id = 1
df_store = df[df["Store"] == store_id].copy()

# Keep only open days with positive sales 
df_store = df_store[
    (df_store["Open"] == 1) &
    (df_store["Sales"] > 0)
].copy()

# Sort chronologically
df_store = df_store.sort_values("Date").reset_index(drop=True)

#Basic information
print("Rossmann Store Sales - Exploratory Analysis")
print(f"Selected store: {store_id}")
print(f"Total valid records: {len(df_store)}")
print(f"Initial date: {df_store['Date'].min()}")
print(f"Final date: {df_store['Date'].max()}")
print("\nFirst rows:")
print(df_store.head())
print("\nLast rows:")
print(df_store.tail())
print("\nSales statistical summary:")
print(df_store["Sales"].describe())

# Create date features to analysis
df_store["month"] = df_store["Date"].dt.month
df_store["year"] = df_store["Date"].dt.year
df_store["day_of_week"] = df_store["Date"].dt.dayofweek

# Average sales by month
monthly_avg = df_store.groupby("month")["Sales"].mean()

# Average sales by day of week
weekday_avg = df_store.groupby("day_of_week")["Sales"].mean()

# Graph 1: Time series 
plt.figure(figsize=(14, 5))
plt.plot(df_store["Date"], df_store["Sales"], color="blue", linewidth=1)
plt.title("Daily Sales Time Series - Rossmann Store 1")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_path}/temporal_series_sales.png")
plt.show()

# Graph 2: Sales distribution 
plt.figure(figsize=(14, 5))
plt.hist(df_store["Sales"], bins=30, color="skyblue", edgecolor="black")
plt.title("Sales Distribution - Rossmann Store 1")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{output_path}/sales_distribution.png")
plt.show()

# Graph 3: Average sales by month
plt.figure(figsize=(14, 5))
plt.plot(monthly_avg.index, monthly_avg.values, marker="o", color="orange")
plt.title("Average Sales by Month - Rossmann Store 1")
plt.xlabel("Month")
plt.ylabel("Average Sales")
plt.xticks(range(1, 13))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_path}/month_average_sales.png")
plt.show()

# Graph 4: Average sales by day of week
plt.figure(figsize=(10, 5))
plt.bar(weekday_avg.index, weekday_avg.values, color="green")
plt.title("Average Sales by Day of Week - Rossmann Store 1")
plt.xlabel("Day of Week")
plt.ylabel("Average Sales")
plt.xticks(
    ticks=range(7),
    labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
)
plt.tight_layout()
plt.savefig(f"{output_path}/weekday_average_sales.png")
plt.show()

# Graph 5: Real sales vs Global mean baseline
mean_sales = df_store["Sales"].mean()
df_store["forecast_mean"] = mean_sales

plt.figure(figsize=(14, 5))
plt.plot(df_store["Date"], df_store["Sales"], label="Real Sales", color="blue", linewidth=1)
plt.plot(df_store["Date"], df_store["forecast_mean"], label="Global Mean Baseline", color="red", linestyle="--")
plt.title("Real Sales vs Global Mean Baseline")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_path}/real_vs_mean_forecast.png")
plt.show()

# Graph 6: Naive forecast usong previous day
df_store["naive_forecast"] = df_store["Sales"].shift(1)
plt.figure(figsize=(14, 5))
plt.plot(df_store["Date"], df_store["Sales"], label="Real Sales", color="blue")
plt.plot(df_store["Date"], df_store["naive_forecast"], label="Naive Forecast Previous Day", color="green", linestyle=":")
plt.title("Naive Forecast vs Previous Day Sale")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_path}/naive_vs_previous_day.png")
plt.show()

print("\nGraphs saved successfully in outputs/graficos/")
