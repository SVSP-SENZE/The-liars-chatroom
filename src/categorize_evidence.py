import pandas as pd, re

df = pd.read_csv('dataset/extracted/train.csv')

# Categorize evidence types
categories = {
    'time_window': 0,      # "between X and Y nobody could..."
    'calendar': 0,         # "X date was a Wednesday"
    'counts': 0,           # "exactly N items"
    'location': 0,         # "campus, bridge, stadium are separate"
    'person_access': 0,    # "only X and Y were at..."
    'code/key': 0,         # "only code X could unlock..."
    'rota': 0,             # "open Wednesday to Sunday"
    'transit': 0,          # "takes at least N minutes"
    'other': 0,
}

for _, row in df.iterrows():
    evd = row['evidence'].lower()
    if re.search(r'\b(wednesday|thursday|friday|monday|tuesday|saturday|sunday)\b', evd):
        categories['calendar'] += 1
    elif re.search(r'exactly \d+', evd):
        categories['counts'] += 1
    elif re.search(r'(only|nobody could enter|nobody could use)', evd) and re.search(r'(between|from).+to', evd):
        categories['time_window'] += 1
    elif re.search(r'(campus|bridge|stadium|route|at least \d+ minute)', evd):
        categories['location'] += 1
    elif re.search(r'only \w+ and \w+ were at', evd):
        categories['person_access'] += 1
    elif re.search(r'(code|unlock|key)', evd):
        categories['code/key'] += 1
    elif re.search(r'open only \w+day', evd):
        categories['rota'] += 1
    elif re.search(r'at least \d+ (hour|minute)', evd):
        categories['transit'] += 1
    else:
        categories['other'] += 1

for k, v in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"{k}: {v} ({v/25:.1f}%)")
