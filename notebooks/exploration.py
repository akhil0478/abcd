import pandas as pd

df = pd.read_csv("data/commodity_price.csv")

print("Columns:")
print(df.columns)

print("\nSample rows:")
print(df.head())

