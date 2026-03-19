import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_summary(path='results/lhs_summary.csv'):
    return pd.read_csv(path)

def plot(summary, out_png='results/lhs_pareto_pub.png', out_pdf='results/lhs_pareto_pub.pdf'):
    x = summary['delta_f1_mean']
    y = summary['CCR_mean']

    fig, ax = plt.subplots(figsize=(7,5))
    sc = ax.scatter(x, y, c='C0', s=80, edgecolor='k', alpha=0.9)

    # annotate top-3 CCR
    top_idx = np.argsort(-y)[:3]
    for i in top_idx:
        sample_id = int(summary.iloc[i]['sample'])
        ax.annotate(f"sample {sample_id}",
                    (x.iloc[i], y.iloc[i]), xytext=(5,5), textcoords='offset points', fontsize=9)

    # draw convex-like frontier by sorting CCR desc and keeping lower delta
    pts = np.column_stack((x.values, y.values))
    order = np.argsort(-pts[:,1])
    best = []
    min_delta = float('inf')
    for idx in order:
        d, c = pts[idx]
        if d < min_delta:
            best.append((d, c))
            min_delta = d
    if best:
        best = np.array(best)
        ax.plot(best[:,0], best[:,1], '-r', lw=2, label='Pareto frontier')

    ax.set_xlabel('Delta F1 (baseline - federated)', fontsize=12)
    ax.set_ylabel('CCR (higher is better)', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    out_dir = Path('results')
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"Saved plot: {out_png}, {out_pdf}")

if __name__ == '__main__':
    s = load_summary()
    plot(s)
