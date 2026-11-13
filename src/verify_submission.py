import os
import sys
import pandas as pd

def verify_submission(sub_path, test_path='dataset/extracted/test.csv'):
    if not os.path.exists(sub_path):
        print(f"FAILED: File does not exist: {sub_path}")
        return False
        
    df_sub = pd.read_csv(sub_path)
    df_test = pd.read_csv(test_path)
    
    # 1. Header check
    if list(df_sub.columns) != ['id', 'label']:
        print(f"FAILED: Header must be 'id,label', got {list(df_sub.columns)}")
        return False
        
    # 2. Row count check
    if len(df_sub) != len(df_test):
        print(f"FAILED: Row count mismatch! Submission has {len(df_sub)} rows, expected {len(df_test)}")
        return False
        
    # 3. ID match check
    if not (df_sub['id'] == df_test['id']).all():
        mismatches = (df_sub['id'] != df_test['id']).sum()
        print(f"FAILED: ID order does not match test.csv! {mismatches} mismatches found.")
        return False
        
    # 4. Null check
    if df_sub['label'].isnull().any():
        print(f"FAILED: Submission contains {df_sub['label'].isnull().sum()} null values!")
        return False
        
    # 5. Valid labels check
    valid_labels = {'A', 'B', 'C', 'D'}
    unique_labels = set(df_sub['label'].unique())
    invalid_labels = unique_labels - valid_labels
    if len(invalid_labels) > 0:
        print(f"FAILED: Invalid labels found: {invalid_labels}")
        return False
        
    print(f"SUCCESS: Submission {sub_path} passed ALL validation checks!")
    print(f"Total rows: {len(df_sub)}")
    print("Label distribution:")
    for lbl, count in df_sub['label'].value_counts().items():
        pct = count / len(df_sub) * 100
        print(f"   {lbl}: {count} ({pct:.2f}%)")
        
    return True

if __name__ == '__main__':
    sub_file = sys.argv[1] if len(sys.argv) > 1 else 'dataset/extracted/sample_submission.csv'
    verify_submission(sub_file)
