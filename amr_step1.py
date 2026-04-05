import pandas as pd

# Load your dataset
df = pd.read_csv("your_file.csv")

# Show basic info
print("Columns:", df.columns)
print(df.head())