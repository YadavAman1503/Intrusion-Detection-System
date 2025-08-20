import pandas as pd

df = pd.read_csv("data/Data.csv")  # or your actual path
print("🧾 Columns in your dataset:")
print(df.columns.tolist())
