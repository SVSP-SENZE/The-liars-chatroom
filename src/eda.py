import os
import re
import pandas as pd

def run_eda():
    train_path = os.path.join('dataset', 'extracted', 'train.csv')
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df)} train cases.")
    
    total = len(df)
    liar_had_non_question = 0
    speakers_distribution = {k: 0 for k in ['A', 'B', 'C', 'D']}
    
    # Check candidates per case who only sent questions vs who made factual claims
    speakers_with_no_claims = 0
    total_candidates_eliminated = 0
    
    for idx, row in df.iterrows():
        lines = [l.strip() for l in row['chat'].split('\n') if l.strip()]
        label = row['label']
        speakers_distribution[label] += 1
        
        sender_msgs = {'A': [], 'B': [], 'C': [], 'D': []}
        for line in lines:
            m = re.match(r'^\d+\.\s*([A-D]):\s*(.*)$', line)
            if m:
                s, txt = m.group(1), m.group(2)
                sender_msgs[s].append(txt)
                
        liar_msgs = sender_msgs[label]
        non_questions = [m for m in liar_msgs if '?' not in m]
        if len(non_questions) > 0:
            liar_had_non_question += 1
        else:
            print(f"Case {row['id']}: Liar {label} had NO claims: {liar_msgs}")
            
        for s in ['A', 'B', 'C', 'D']:
            claims = [m for m in sender_msgs[s] if '?' not in m]
            if len(claims) == 0:
                speakers_with_no_claims += 1
                total_candidates_eliminated += 1
                
    print(f"Liar had non-question message: {liar_had_non_question} / {total} ({liar_had_non_question/total*100:.2f}%)")
    print(f"Total speakers with ZERO claims across all cases: {speakers_with_no_claims} (avg {speakers_with_no_claims/total:.2f} per case)")
    print("Label distribution in train:", speakers_distribution)

if __name__ == '__main__':
    run_eda()
