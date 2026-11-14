import pandas as pd, re

df = pd.read_csv('dataset/extracted/train.csv')

for i in [2, 10, 25, 42, 100, 150, 200, 300, 400, 500]:
    row = df.iloc[i]
    row_id = row['id']
    lbl = row['label']
    evd = row['evidence'][:300]
    print(f"=== {row_id} (label={lbl}) ===")
    print("EVD:", evd)
    print()
