import os
import re
import pandas as pd
from collections import Counter

df = pd.read_csv('dataset/extracted/train.csv')

# Look at short messages or messages that appear frequently across different scenarios
all_messages = []
for chat in df['chat']:
    for line in chat.split('\n'):
        m = re.match(r'^\d+\.\s*([A-D]):\s*(.*)$', line.strip())
        if m:
            all_messages.append(m.group(2).strip())

counter = Counter(all_messages)
print("Top 30 most frequent messages across the entire dataset:")
for msg, cnt in counter.most_common(30):
    print(f"[{cnt}x] {msg}")
