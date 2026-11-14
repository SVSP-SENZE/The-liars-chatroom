"""
cv_validate.py
5-Fold stratified cross-validation using local HuggingFace inference.
Runs on CPU (slow but correct). Use to verify prompt quality before GPU submission.

Usage:
  python src/cv_validate.py --model microsoft/deberta-v3-base --folds 5
  python src/cv_validate.py --mode random   # sanity check baseline
"""

import os
import re
import sys
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from chat_parser import parse_speakers, parse_chat
from prompt_v2 import build_prompt, extract_answer

random.seed(42)

def majority_vote(candidates, valid):
    """Pick the most common letter among candidates, restricted to valid set."""
    filtered = [c for c in candidates if c in valid]
    if not filtered:
        return sorted(valid)[0] if valid else 'A'
    return Counter(filtered).most_common(1)[0][0]


def evaluate_random_baseline(df):
    """Random baseline (expect ~25%)."""
    correct = 0
    for _, row in df.iterrows():
        l2n, _ = parse_speakers(row['speakers'])
        msgs = parse_chat(row['chat'], l2n)
        valid = set(m['sender'] for m in msgs if not m['is_filler']) or {'A', 'B', 'C', 'D'}
        pred = random.choice(sorted(valid))
        if pred == row['label']:
            correct += 1
    acc = correct / len(df)
    print(f"Random baseline (from valid candidates): {acc*100:.2f}%")
    return acc


def evaluate_oracle_filter(df):
    """
    Oracle that always restricts to valid candidates (non-filler speakers).
    Shows theoretical max from filtering alone.
    """
    correct = 0
    total_valid = []
    for _, row in df.iterrows():
        l2n, _ = parse_speakers(row['speakers'])
        msgs = parse_chat(row['chat'], l2n)
        valid = set(m['sender'] for m in msgs if not m['is_filler']) or {'A', 'B', 'C', 'D'}
        total_valid.append(len(valid))
        # Oracle: pick correct if in valid set
        if row['label'] in valid:
            # uniform random among valid
            if len(valid) == 1:
                correct += 1
            else:
                correct += 1 / len(valid)
    acc = correct / len(df)
    avg_valid = np.mean(total_valid)
    print(f"Oracle uniform-random among valid candidates: {acc*100:.2f}%")
    print(f"  Average candidates per case: {avg_valid:.2f}")
    return acc


def run_hf_inference_on_fold(fold_df, model_name, max_new_tokens=400):
    """
    Runs HuggingFace generate() inference on a fold.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading {model_name} on {device}...")

    quant_cfg = None
    if torch.cuda.is_available():
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None
    )
    model.eval()

    correct = 0
    preds = []
    for idx, row in tqdm(fold_df.iterrows(), total=len(fold_df)):
        result = build_prompt(row)
        valid = result['valid_candidates']

        messages = [
            {'role': 'system', 'content': result['system']},
            {'role': 'user', 'content': result['user']},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        new_toks = out[0][inputs['input_ids'].shape[1]:]
        resp = tokenizer.decode(new_toks, skip_special_tokens=True)
        pred = extract_answer(resp, valid)
        preds.append(pred)
        if pred == row['label']:
            correct += 1

    acc = correct / len(fold_df)
    return acc, preds


def run_cv(args):
    df = pd.read_csv('data/train_folds.csv')
    print(f"Loaded {len(df)} rows with {args.folds} folds.")
    print()

    if args.mode == 'random':
        evaluate_random_baseline(df)
        evaluate_oracle_filter(df)
        return

    fold_accs = []
    all_preds = {}

    for fold in range(args.folds):
        val_df = df[df['fold'] == fold].reset_index(drop=True)
        print(f"\n--- Fold {fold}: {len(val_df)} validation cases ---")

        acc, preds = run_hf_inference_on_fold(val_df, args.model, args.max_new_tokens)
        fold_accs.append(acc)
        for i, (_, row) in enumerate(val_df.iterrows()):
            all_preds[row['id']] = preds[i]
        print(f"Fold {fold} Accuracy: {acc*100:.2f}%")

        if args.one_fold:
            break

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\n{'='*50}")
    print(f"Cross-Validation Results ({len(fold_accs)} folds):")
    for i, a in enumerate(fold_accs):
        print(f"  Fold {i}: {a*100:.2f}%")
    print(f"  Mean: {mean_acc*100:.2f}%  ±{std_acc*100:.2f}%")
    print(f"{'='*50}")

    # Save OOF predictions
    os.makedirs('submissions', exist_ok=True)
    oof_df = pd.DataFrame({'id': list(all_preds.keys()), 'pred': list(all_preds.values())})
    df_merged = df.merge(oof_df, on='id')
    df_merged['correct'] = df_merged['pred'] == df_merged['label']
    oof_path = 'submissions/oof_predictions.csv'
    df_merged.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='hf', choices=['hf', 'random'])
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--one_fold', action='store_true', help='Run only fold 0 for quick check')
    parser.add_argument('--max_new_tokens', type=int, default=400)
    args = parser.parse_args()
    run_cv(args)
