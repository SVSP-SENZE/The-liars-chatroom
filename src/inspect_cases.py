import os
import pandas as pd

df = pd.read_csv('dataset/extracted/train.csv')
indices = [3, 10, 25, 42, 100]

for idx in indices:
    row = df.iloc[idx]
    print('='*70)
    print(f"ID: {row['id']} | TRUE LIAR: {row['label']}")
    print(f"Scenario: {row['scenario']}")
    print(f"Speakers: {row['speakers']}")
    print(f"Evidence: {row['evidence']}")
    print("Chat:")
    for line in row['chat'].split('\n'):
        print("  ", line)
