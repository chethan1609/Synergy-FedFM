import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Read summary
df = pd.read_csv('results/lhs_summary.csv')
fig, ax = plt.subplots(figsize=(7, 4.5))

colors = ['#1D9E75' if d <= 0.05 else '#378ADD' 
          for d in df['delta_f1_mean']]

ax.errorbar(df['CCR_mean'], df['delta_f1_mean'],
            yerr=df['delta_f1_std'], fmt='none',
            ecolor='#73726c', elinewidth=1, capsize=3, zorder=2)
ax.scatter(df['CCR_mean'], df['delta_f1_mean'],
           c=colors, s=60, zorder=3)

for _, row in df.iterrows():
    ax.annotate(f"S{int(row['sample'])}",
                (row['CCR_mean'], row['delta_f1_mean']),
                textcoords='offset points', xytext=(6, 4),
                fontsize=8, color='#444441')

ax.axhline(0.03, color='#E24B4A', linestyle='--',
           linewidth=1.2, label=r'$\tau_{tol}=0.03$')
ax.set_xscale('log')
ax.set_xlabel('Cumulative Compression Ratio (CCR)', fontsize=11)
ax.set_ylabel(r'$\Delta$F1 Degradation', fontsize=11)
ax.set_title('Synergy-FedFM Pareto Frontier', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()

out_pdf = 'results/lhs_pareto_pub.pdf'
out_tiff = 'results/lhs_pareto_pub.tiff'
out_png = 'results/lhs_pareto_pub.png'
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
plt.savefig(out_tiff, dpi=300, bbox_inches='tight')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Saved PDF, TIFF, PNG: {out_pdf}, {out_tiff}, {out_png}")
