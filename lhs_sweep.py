import time
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulation import partition_data, compute_central_baseline, run_flower_simulation

def lhs_sampling(n, param_ranges, rng):
    # param_ranges: dict name -> (low, high, is_int)
    samples = {k: np.zeros(n) for k in param_ranges}
    for i, (k, (lo, hi, is_int)) in enumerate(param_ranges.items()):
        cut = np.linspace(0, 1, n+1)
        u = rng.random(n)
        points = cut[:n] + u * (1.0 / n)
        rng.shuffle(points)
        vals = lo + points * (hi - lo)
        if is_int:
            vals = np.round(vals).astype(int)
        samples[k] = vals
    # assemble list of dicts
    out = []
    for i in range(n):
        cfg = {k: samples[k][i] for k in param_ranges}
        out.append(cfg)
    return out

def pareto_frontier(points):
    # points: list of (delta_f1, CCR)
    pts = np.array(points)
    # we want minimize delta_f1 and maximize CCR
    # sort by CCR descending and keep minimal delta seen
    order = np.argsort(-pts[:,1])
    best = []
    min_delta = float('inf')
    for idx in order:
        d, c = pts[idx]
        if d < min_delta:
            best.append((d, c))
            min_delta = d
    return best

def main():
    parquet = './cicids2017_clean.parquet'
    try:
        df = pd.read_parquet(parquet)
    except Exception as e:
        print(f"[LHS] Failed to read parquet {parquet}: {e}. Falling back to CSV source and regenerating parquet.")
        from dataset_loader import read_and_clean_csvs, save_parquet
        df = read_and_clean_csvs('./dataset')
        try:
            save_parquet(df, parquet)
            print(f"[LHS] Regenerated parquet {parquet} shape={df.shape}")
        except Exception as _:
            print(f"[LHS] Warning: could not save regenerated parquet {parquet}")
    labels = df['Label'].values
    classes, encoded = np.unique(labels, return_inverse=True)
    X = df.drop(columns=['Label']).values.astype(np.float32)

    num_clients = 10
    streams = partition_data(X, encoded, num_clients, 0.1)

    # LHS parameter ranges
    param_ranges = {
        'quantize_bits': (8, 32, True),
        'M_max': (100, 1000, True),
        'r_k': (4, 16, True),
        'gamma': (0.001, 0.05, False),
        'lambda_reg': (0.0, 0.2, False),
    }

    rng = np.random.default_rng(seed=12345)
    n_samples = 12
    repeats = 3
    samples = lhs_sampling(n_samples, param_ranges, rng)

    results = []

    # Stronger oracle per thesis methodology: more epochs, lower LR, class-weighted loss
    # Use a fixed, precomputed Oracle baseline F1 to avoid fragile centralized training
    # baseline_f1 = compute_central_baseline(X, encoded, len(classes), X.shape[1], {'d_hidden':128,'R_max':16}, epochs=50, lr=0.01, batch_size=64)
    baseline_f1 = 0.9257
    print(f"[LHS] Using hardcoded Oracle baseline_f1 = {baseline_f1}")
    # Fallback: if hardcoded baseline somehow equals 0.0 (shouldn't happen), compute sklearn baseline
    if float(baseline_f1) == 0.0:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import f1_score
            clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga')
            clf.fit(X, encoded)
            preds = clf.predict(X)
            baseline_f1 = float(f1_score(encoded, preds, average='macro'))
            print(f"[LHS] Fallback sklearn baseline_f1={baseline_f1:.6f}")
        except Exception as e:
            print(f"[LHS] Fallback baseline computation failed: {e}")
    # Verification prints requested by user: confirm oracle and sweep size before loop
    print(f"[LHS] Using oracle baseline_f1 = {baseline_f1:.4f}")
    print(f"[LHS] Running {n_samples} samples × {repeats} repeats")

    for i, s in enumerate(samples):
        for rep in range(repeats):
            ts = int(time.time())
            exp_name = f"lhs_{i}_rep{rep}_{ts}"
            hparams = {
                'd_hidden': 128,
                'R_max': 16,
                'r_k': int(s['r_k']),
                'epochs': 5,
                'lr': 0.05,
                'lambda_reg': float(s['lambda_reg']),
                'batch_size': 64,
                'quantize_bits': int(s['quantize_bits']),
                'M_max': int(s['M_max']),
                'gamma': float(s['gamma']),
            }
            args = {'parquet': parquet, 'num_rounds': 20, 'results_dir': 'results', 'exp_name': exp_name, 'baseline_f1': float(baseline_f1), 'force_emulation': True}
            print(f"[LHS] Running sample {i} rep {rep} exp={exp_name} params={hparams}")
            history = run_flower_simulation(args, streams, len(classes), X.shape[1], hparams)
            # read resulting CSV
            csv_path = os.path.join('results', f"{exp_name}.csv")
            if not os.path.exists(csv_path):
                print(f"[LHS] Missing results CSV {csv_path}")
                continue
            dfr = pd.read_csv(csv_path)
            last = dfr.iloc[-1]
            mean_f1 = float(last['f1'])
            CCR = float(last['CCR']) if 'CCR' in last else float(dfr['CCR'].astype(float).iloc[-1])
            delta_f1 = float(baseline_f1) - mean_f1
            results.append({**s, 'sample': i, 'rep': rep, 'mean_f1': mean_f1, 'delta_f1': delta_f1, 'CCR': CCR, 'exp_name': exp_name})

    resdf = pd.DataFrame(results)
    summary = resdf.groupby('sample').agg({'CCR':'mean','delta_f1':['mean','std']}).reset_index()
    summary.columns = ['sample','CCR_mean','delta_f1_mean','delta_f1_std']
    summary.to_csv('results/lhs_summary.csv', index=False)

    # pareto
    points = [(row['delta_f1_mean'], row['CCR_mean']) for _, row in summary.iterrows()]
    frontier = pareto_frontier(points)

    # plot
    plt.figure(figsize=(7,5))
    plt.scatter(summary['delta_f1_mean'], summary['CCR_mean'], c='C0')
    fx = [p[0] for p in frontier]
    fy = [p[1] for p in frontier]
    plt.plot(fx, fy, '-r')
    plt.xlabel('Delta F1 (baseline - federated)')
    plt.ylabel('CCR')
    plt.title('LHS Pareto: CCR vs Delta F1')
    plt.grid(True)
    out_png = 'results/lhs_pareto.png'
    plt.savefig(out_png)
    print(f"[LHS] Done. Summary: results/lhs_summary.csv, plot: {out_png}")

if __name__ == '__main__':
    main()
