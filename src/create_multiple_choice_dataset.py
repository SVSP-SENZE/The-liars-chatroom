import os
import json
import pandas as pd
from chat_parser import parse_speakers, parse_chat

def create_dataset(input_csv, output_json, is_test=False):
    df = pd.read_csv(input_csv)
    records = []
    
    label_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    
    for _, row in df.iterrows():
        case_id = row['id']
        scenario = row['scenario']
        speakers = row['speakers']
        evidence = row['evidence']
        chat = row['chat']
        
        l2n, _ = parse_speakers(speakers)
        msgs = parse_chat(chat, l2n)
        
        # Group candidate claims for each speaker A, B, C, D
        speaker_claims = {'A': [], 'B': [], 'C': [], 'D': []}
        for m in msgs:
            if not m['is_filler']:
                speaker_claims[m['sender']].append(m['text'])
                
        # Format the 4 choices
        choices = []
        for s in ['A', 'B', 'C', 'D']:
            name = l2n.get(s, s)
            claims = speaker_claims[s]
            if len(claims) == 0:
                choice_text = f"Speaker {s} ({name}) made no claims."
            else:
                choice_text = f"Speaker {s} ({name}) claims: " + " ".join(claims)
            choices.append(choice_text)
            
        record = {
            'id': case_id,
            'context': f"Scenario: {scenario} Evidence: {evidence}",
            'choice_A': choices[0],
            'choice_B': choices[1],
            'choice_C': choices[2],
            'choice_D': choices[3]
        }
        
        if not is_test:
            record['label'] = label_to_idx[row['label']]
            if 'fold' in row:
                record['fold'] = int(row['fold'])
                
        records.append(record)
        
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
        
    print(f"Processed {len(records)} records -> {output_json}")

if __name__ == '__main__':
    create_dataset('data/train_folds.csv', 'data/mc_train.json', is_test=False)
    create_dataset('dataset/extracted/test.csv', 'data/mc_test.json', is_test=True)
