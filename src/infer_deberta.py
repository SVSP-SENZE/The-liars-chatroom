import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from train_deberta import DataCollatorForMultipleChoice

def run_inference(args):
    with open(args.test_json, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    print(f"Loaded {len(records)} test cases.")
    
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    model = AutoModelForMultipleChoice.from_pretrained(args.checkpoint_dir)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("Using device:", device)
    
    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples["context"]]
        second_sentences = [
            [examples["choice_A"][i], examples["choice_B"][i], examples["choice_C"][i], examples["choice_D"][i]]
            for i in range(len(examples["context"]))
        ]
        
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=args.max_length
        )
        
        return {
            k: [v[i : i + 4] for i in range(0, len(v), 4)]
            for k, v in tokenized_examples.items()
        }

    test_ds = Dataset.from_list(records)
    tokenized_ds = test_ds.map(preprocess_function, batched=True)
    
    collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    
    batch_size = args.batch_size
    all_logits = []
    
    idx_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    for i in range(0, len(tokenized_ds), batch_size):
        batch_records = [tokenized_ds[j] for j in range(i, min(i + batch_size, len(tokenized_ds)))]
        # Remove non-tensor columns for collator
        clean_batch = []
        for r in batch_records:
            clean_batch.append({
                'input_ids': r['input_ids'],
                'attention_mask': r['attention_mask'],
                'token_type_ids': r.get('token_type_ids')
            })
            if clean_batch[-1]['token_type_ids'] is None:
                del clean_batch[-1]['token_type_ids']
                
        batch = collator(clean_batch)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits.cpu().numpy()
            all_logits.append(logits)
            
    all_logits = np.vstack(all_logits)
    
    # Post-processing mask: if speaker made no claims, mask with -1e9
    predictions = []
    for idx, r in enumerate(records):
        logits = all_logits[idx].copy()
        if "made no claims" in r['choice_A']: logits[0] = -1e9
        if "made no claims" in r['choice_B']: logits[1] = -1e9
        if "made no claims" in r['choice_C']: logits[2] = -1e9
        if "made no claims" in r['choice_D']: logits[3] = -1e9
        
        pred_label = idx_to_label[int(np.argmax(logits))]
        predictions.append({'id': r['id'], 'label': pred_label})
        
    df_sub = pd.DataFrame(predictions)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_sub.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")
    print("Label distribution in predictions:")
    print(df_sub['label'].value_counts(normalize=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_json', type=str, default='data/mc_test.json')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='submissions/sub_deberta.csv')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=384)
    args = parser.parse_args()
    run_inference(args)
