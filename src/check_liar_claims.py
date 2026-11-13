import pandas as pd
from chat_parser import parse_speakers, parse_chat

df = pd.read_csv('data/train_folds.csv')

liar_claim_counts = []
for idx, row in df.iterrows():
    l2n, _ = parse_speakers(row['speakers'])
    msgs = parse_chat(row['chat'], l2n)
    liar = row['label']
    liar_claims = [m for m in msgs if m['sender'] == liar and not m['is_filler']]
    liar_claim_counts.append(len(liar_claims))

from collections import Counter
print("Number of claims sent by the liar in each case:")
print(Counter(liar_claim_counts))
