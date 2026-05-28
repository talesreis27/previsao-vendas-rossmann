import pandas as pd
import os

# The way to the csv files
train_path = "data/raw/train.csv"
store_path = "data/raw/store.csv"

# Verify that the files exist
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Train file not found at data/raw")
if not os.path.exists(store_path):
    raise FileNotFoundError(f"file not found at data/raw/")  

#load the data
train_df = pd.read_csv(train_path)
store_df = pd.read_csv(store_path)    

# General information about the datasets
print("Dataset train.csv loaded successfully...")
print(f"Lines: {train_df.shape[0]}, Columns: {train_df.shape[1]}")
print(f"Columns: {train_df.shape[1]}")
print("\nColumns of train.csv:")
print(train_df.columns.tolist())
print("\nFirst lines of train.csv:")
print(train_df.head())
print("\nDataset store.csv loaded successfully...")
print(f"Lines: {store_df.shape[0]}")
print(f"Columns: {store_df.shape[1]}")
print("\nColumns of store.csv:")    
print(store_df.columns.tolist())

# Convert colum Date to datetime
train_df['Date'] = pd.to_datetime(train_df['Date'])

print("\nPeriod of dataset:")
print(f"From {train_df['Date'].min()} to {train_df['Date'].max()}")

# Verify the number of stores
print(f"\nQuantity of stores: {train_df['Store'].nunique()}")

#Count the registered sales per store
records_by_store = train_df.groupby('Store').size().sort_values(ascending=False)

print("\nTop 10 stores with more records:")
print(records_by_store.head(10))

# Chose the store for the project
store_id = 1
store_df_filtered = train_df[train_df['Store'] == store_id].copy()

print(f"\nRegisters of store id: {store_id}:")
print(len(store_df_filtered))

#Verify the days witcht the store was open and have positive sales
store_open = store_df_filtered[
    (store_df_filtered['Open'] == 1) &
      (store_df_filtered['Sales'] > 0)
].copy()    

print(f"Registers of store {store_id} with positive sales and open: {len(store_open)}")
print(f"First lines of store {store_id} with positive sales and open:")
print(store_open.head())
print(f"Statistic resume of sales of store chosen:")
print(store_open['Sales'].describe())

# Confirm the if have 500 records
if len(store_open) >= 500:
    print(f"\nStore {store_id} has more than 500 records with positive sales and open.")
else:
    print(f"\nStore {store_id} has less than 500 records with positive sales and open. Consider choosing another store.")
