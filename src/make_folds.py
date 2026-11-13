import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def make_folds(n_splits=5, seed=42):
    train_path = os.path.join('dataset', 'extracted', 'train.csv')
    df = pd.read_csv(train_path)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        df.loc[val_idx, 'fold'] = fold
        
    os.makedirs('data', exist_ok=True)
    out_path = os.path.join('data', 'train_folds.csv')
    df.to_csv(out_path, index=False)
    print(f"Created {n_splits} folds and saved to {out_path}.")
    print("Fold counts per label:")
    print(pd.crosstab(df['fold'], df['label']))

if __name__ == '__main__':
    make_folds()
