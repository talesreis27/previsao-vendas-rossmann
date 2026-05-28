import pandas as pd
import os


# Paths
train_path = "data/raw/train.csv"
output_path = "data/processed/processed_data.csv"

# Check if the file exists
if not os.path.exists(train_path):
    raise FileNotFoundError("train.csv file not found in data/raw/")

# Load dataset
train_df = pd.read_csv(train_path, dtype={"StateHoliday": str}, low_memory=False)

print("Rossmann dataset loaded successfully!")
print(f"Train shape: {train_df.shape}")

# Convert Date column to datetime
train_df["Date"] = pd.to_datetime(train_df["Date"])

# Select one store for a clear time series problem
store_id = 1
df = train_df[train_df["Store"] == store_id].copy()

# Sort chronologically
df = df.sort_values("Date").reset_index(drop=True)

print(f"\nSelected store: {store_id}")
print(f"Records for selected store: {len(df)}")
print(f"Initial date: {df['Date'].min()}")
print(f"Final date: {df['Date'].max()}")

# Select useful columns
df = df[
    [
        "Store",
        "Date",
        "Sales",
        "Promo",
        "Open",
        "StateHoliday",
        "SchoolHoliday",
        "DayOfWeek",
    ]
].copy()

# Convert categorical columns into numeric features
df["StateHoliday"] = df["StateHoliday"].astype(str)
df["is_state_holiday"] = (df["StateHoliday"] != "0").astype(int)

# Calendar features
df["day"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month
df["year"] = df["Date"].dt.year
df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
df["is_november"] = (df["month"] == 11).astype(int)
df["is_december"] = (df["month"] == 12).astype(int)

# Lag features
# These help the model learn from previous sales values.
df["lag_1"] = df["Sales"].shift(1)
df["lag_7"] = df["Sales"].shift(7)
df["lag_14"] = df["Sales"].shift(14)
df["lag_30"] = df["Sales"].shift(30)

# Moving averages
# These represent short and medium-term sales trends.
df["MA7"] = df["Sales"].rolling(window=7).mean()
df["MA14"] = df["Sales"].rolling(window=14).mean()
df["MA30"] = df["Sales"].rolling(window=30).mean()

# Remove rows with missing values created by lag and moving average features
df = df.dropna().reset_index(drop=True)

# Guarantee that the output folder exists
os.makedirs("data/processed", exist_ok=True)

# Save processed dataset
df.to_csv(output_path, index=False)

print("\nProcessed dataset created successfully!")
print(f"Final shape: {df.shape}")
print(f"Saved at: {output_path}")
print("\nColumns in processed dataset:")
print(df.columns.tolist())
print("\nFirst rows:")
print(df.head())
print("\nLast rows:")
print(df.tail())

# Check if the processed dataset still has at least 500 records
if len(df) >= 500:
    print("\nOK: Processed dataset still has more than 500 records.")
else:
    print("\nWARNING: Processed dataset has fewer than 500 records.")