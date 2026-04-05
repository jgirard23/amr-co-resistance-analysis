import pandas as pd
import os

# Path to your E. coli date file
file_path = "/Users/jacobgirard-beaupre/Downloads/NCBI data/ecoli_dates.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path, low_memory=False, nrows=10)
    print("── COLUMNS FOUND ──")
    print(df.columns.tolist())
    print("\n── DATA PREVIEW ──")
    print(df)
else:
    print(f"File not found at: {file_path}")
    