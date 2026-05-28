import pandas as pd
import os 

# Path of processed dataset
processed_path = "data/processed/processed_data.csv"

# Check if the processed dataset exists
if not os.path.exists(processed_path):
    raise FileNotFoundError("processed_data.csv not found. Run 02_process_features.py first.")

# Load processed dataset 
df = pd.read_csv(processed_path)

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort cronologically to keep the time series order
df = df.sort_values("Date").reset_index(drop=True)

print("Processed dataset loaded successfully.")
print(f"Total records: {len(df)}")
print(f"Initial date: {df['Date'].min()}")
print(f"Final date: {df['Date'].max()}")

# Split data into train and test sets 
# For time series, we do no shuffle the data.
split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\nTrain/Test split:")
print(f"Train records: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
print(f"Test records: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")
print("\nTrain period:")
print(f"From {train_df['Date'].min()} to {train_df['Date'].max()}")
print("\nTest period:")
print(f"From {test_df['Date'].min()} to {test_df['Date'].max()}")

# Save train an test datasets
os.makedirs("data/processed", exist_ok=True)
train_df.to_csv("data/processed/train_data.csv", index=False)
test_df.to_csv("data/processed/test_data.csv", index=False)

print("\nData saved successfully:")
print("- data/processed/train_data.csv")
print("- data/processed/test_data.csv")

# Quick check
if len(train_df) > len(test_df):
    print("\nOK: Train set is larger than test set.")
else:
    print("\nWARNING: Train set is not larger than test set.")

