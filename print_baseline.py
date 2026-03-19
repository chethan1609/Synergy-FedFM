import pandas as pd
import numpy as np
from simulation import compute_central_baseline
from dataset_loader import read_and_clean_csvs

parquet = './cicids2017_clean.parquet'
try:
    df = pd.read_parquet(parquet)
except Exception:
    df = read_and_clean_csvs('./dataset')
labels = df['Label'].values
classes, encoded = np.unique(labels, return_inverse=True)
X = df.drop(columns=['Label']).values.astype('float32')
import traceback
try:
    baseline = compute_central_baseline(X, encoded, len(classes), X.shape[1], {'d_hidden':128,'R_max':16}, epochs=50, lr=0.01, batch_size=64)
    print(f"[LHS] Using oracle baseline_f1 = {baseline:.4f}")
except Exception as e:
    print("[LHS] compute_central_baseline raised an exception:")
    traceback.print_exc()
