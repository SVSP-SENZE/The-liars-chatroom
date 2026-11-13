import pandas as pd
from prompt_builder import format_prompt_for_llm

df = pd.read_csv('dataset/extracted/train.csv')
for i in [0, 5, 10]:
    row = df.iloc[i]
    print("="*60)
    print(f"CASE {row['id']} (True Label: {row['label']})")
    print(format_prompt_for_llm(row))
    print()
