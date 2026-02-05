import pandas as pd

# Load raw data
df = pd.read_csv("data/commodity_price.csv")

# Rename columns for sanity
df = df.rename(columns={
    "State": "state",
    "Commodity": "commodity",
    "Arrival_Date": "date",
    "Modal_x0020_Price": "price"
})

# Convert date column
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Drop rows with missing critical values
df = df.dropna(subset=["state", "commodity", "date", "price"])

# Extract year and month
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# Keep only required columns
df = df[["commodity", "state", "year", "month", "price"]]

# Aggregate to monthly state-level prices
df = (
    df.groupby(["commodity", "state", "year", "month"], as_index=False)
      .agg({"price": "mean"})
)

# Save processed dataset
df.to_csv("data/processed.csv", index=False)

print("Preprocessing complete.")
print(df.head())

