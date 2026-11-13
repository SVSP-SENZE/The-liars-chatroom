import os
import argparse
import pandas as pd
from collections import Counter
from verify_submission import verify_submission

def ensemble_submissions(sub_files, weights=None, output_path='submissions/final_ensemble_submission.csv'):
    print(f"Ensembling {len(sub_files)} submissions:")
    for f in sub_files:
        print(f"  - {f}")
        
    dfs = [pd.read_csv(f) for f in sub_files]
    test_ids = dfs[0]['id'].tolist()
    
    if weights is None:
        weights = [1.0] * len(dfs)
        
    final_preds = []
    
    for i, cid in enumerate(test_ids):
        votes = Counter()
        for w, df in zip(weights, dfs):
            label = df.loc[i, 'label']
            votes[label] += w
            
        winner = votes.most_common(1)[0][0]
        final_preds.append({'id': cid, 'label': winner})
        
    df_out = pd.DataFrame(final_preds)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\nFinal ensembled submission written to: {output_path}")
    
    verify_submission(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True, help="List of submission CSVs")
    parser.add_argument('--weights', nargs='+', type=float, default=None, help="Weights for each submission")
    parser.add_argument('--output', type=str, default='submissions/final_ensemble_submission.csv')
    args = parser.parse_args()
    
    ensemble_submissions(args.files, args.weights, args.output)
