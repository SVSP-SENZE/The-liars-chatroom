"""
Quick local sanity test: runs the prompt builder on known training cases
and checks whether the computed facts alone would guide a model to the right answer.
No LLM needed - just validates the preprocessing pipeline.
"""
import sys, os
sys.path.insert(0, 'src')

import pandas as pd
from prompt_v2 import build_prompt, extract_answer

df = pd.read_csv('dataset/extracted/train.csv')

# Select hard cases: calendar, counts, travel time, person access
hard_indices = [2, 10, 25, 42, 75, 100, 150, 200, 300, 400]

print("Pipeline Sanity Test: Prompt Builder + Temporal Preprocessor")
print("="*70)

for i in hard_indices:
    row = df.iloc[i]
    result = build_prompt(row)
    
    print(f"\n[{row['id']}] True label: {row['label']}")
    print(f"Evidence: {row['evidence'][:100]}...")
    print(f"Valid candidates: {sorted(result['valid_candidates'])}")
    
    facts = result['computed_facts'].strip()
    if facts:
        print(f"Computed facts injected:\n{facts[:300]}")
    else:
        print("(No deterministic facts extracted)")
    
    # Simple keyword check: does the computed facts contain info that
    # directly exposes the lie?
    # For calendar cases: does it correctly classify the lied date as CLOSED?
    if 'CLOSED' in facts or 'earliest return' in facts or '% of' in facts:
        print("✅ Deterministic facts likely expose the contradiction directly.")
    else:
        print("ℹ️  LLM must reason from raw evidence (normal path).")
    
    print("-"*50)
