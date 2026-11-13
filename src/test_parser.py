import pandas as pd
from chat_parser import parse_speakers, parse_chat

df = pd.read_csv('dataset/extracted/train.csv')

total = len(df)
false_positive_eliminations = 0
filtered_stats = []

for idx, row in df.iterrows():
    l2n, n2l = parse_speakers(row['speakers'])
    msgs = parse_chat(row['chat'], l2n)
    
    # Check liar's candidate messages
    liar = row['label']
    liar_claims = [m for m in msgs if m['sender'] == liar and not m['is_filler']]
    
    if len(liar_claims) == 0:
        false_positive_eliminations += 1
        print(f"ERROR: Liar {liar} eliminated in row {row['id']}!")
        for m in msgs:
            if m['sender'] == liar:
                print(f"   Msg {m['num']}: {m['text']} (is_filler={m['is_filler']})")
                
    non_filler_senders = set(m['sender'] for m in msgs if not m['is_filler'])
    filtered_stats.append(len(non_filler_senders))

print(f"Total rows: {total}")
print(f"False negative eliminations (where true liar had 0 claims): {false_positive_eliminations}")
print(f"Average number of remaining candidate speakers per case: {sum(filtered_stats)/len(filtered_stats):.2f} / 4.0")
from collections import Counter
print("Distribution of candidate speaker counts:", Counter(filtered_stats))
