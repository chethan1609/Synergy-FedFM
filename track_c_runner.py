import argparse
import pathlib
import time
import numpy as np
import pandas as pd
from simulation import partition_data, compute_central_baseline, run_flower_simulation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', default='./cicids2017_clean.parquet')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_rounds', type=int, default=20)
    parser.add_argument('--quantize_bits', type=int, default=32)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    labels = df['Label'].values
    classes, encoded = np.unique(labels, return_inverse=True)
    X = df.drop(columns=['Label']).values.astype(np.float32)

    streams = partition_data(X, encoded, args.num_clients, 0.1)

    hparams = {
        'd_hidden': 128, 'R_max': 16, 'r_k': 16,
        'epochs': 5, 'lr': 0.05, 'lambda_reg': 0.1,
        'batch_size': 64, 'quantize_bits': int(args.quantize_bits),
        'M_max': 500, 'gamma': 0.01,
    }

    baseline_f1 = compute_central_baseline(X, encoded, len(classes), X.shape[1], hparams, epochs=15, lr=0.05)

    args_dict = {
        'parquet': args.parquet,
        'num_rounds': args.num_rounds,
        'results_dir': args.results_dir,
        'exp_name': args.exp_name or f"track_c_{'no_quant' if args.quantize_bits==32 else 'lrea_arm'}_{int(time.time())}",
        'baseline_f1': float(baseline_f1),
    }

    print(f"[TRACK C] exp={args_dict['exp_name']} quantize_bits={args.quantize_bits} baseline_f1={baseline_f1:.4f}")
    history = run_flower_simulation(args_dict, streams, len(classes), X.shape[1], hparams)
    print("[TRACK C] Done", args_dict['exp_name'])

if __name__ == '__main__':
    main()
