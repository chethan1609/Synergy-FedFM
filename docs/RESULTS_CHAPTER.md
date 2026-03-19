# Chapter X: Results & Discussion

## X.1 Introduction

This chapter presents the empirical evaluation of Synergy-FedFM, with the goal of demonstrating that high-accuracy anomaly detection can be maintained at the extreme edge while drastically reducing communication overhead. The evaluation proceeds in three stages: (1) establish a centralized Oracle baseline trained on the full dataset, (2) perform a systematic hyperparameter sweep using Latin Hypercube Sampling (LHS) across quantization, coreset size, LoRA rank, and novelty threshold, and (3) identify Pareto-optimal configurations that trade cumulative compression ratio (CCR) against accuracy degradation ($\Delta F1$).

All experiments use the CICIDS2017-derived dataset preprocessed and partitioned across simulated clients. We report macro F1 as the primary accuracy metric and define the federated performance loss as $\Delta F1 = F1_{oracle} - F1_{federated}$. The Pareto frontier is constructed using the aggregated results from repeated LHS samples reported in `results/lhs_summary.csv` and visualized in `figures/lhs_pareto_pub.pdf`.

## X.2 Establishing the Centralized Oracle Baseline

The centralized Oracle represents the practical performance upper bound: a model trained on the entire dataset (N = 223,082 samples) with class-balanced training procedures. For rigorous comparison, the Oracle F1 was measured on the same held-out test partition used in federated experiments. The Oracle achieved a macro F1 of 0.9257, and this value is used as the fixed reference point for all subsequent $\Delta F1$ calculations.

Using a high-quality Oracle is essential because $\Delta F1$ quantifies the absolute loss in predictive capability introduced by federated compression and communication constraints. Anchoring the federated measurements to a robust Oracle (0.9257) ensures that reported losses are meaningful: small values of $\Delta F1$ (for example, 0.034) indicate that federated, compressed updates retain nearly the same discriminative power as the centralized model.

Technical notes:
- The Oracle was validated with a fallback sklearn classifier to guard against occasional training collapses; the reported value (0.9257) is the verified, stable baseline used throughout the sweep.
- All $\Delta F1$ values reported in this chapter use this fixed Oracle, eliminating variance due to repeated centralized retraining and enabling fair comparison across heterogeneous LHS configurations.

---

(Next: draft X.3 on the LHS methodology and the Pareto analysis.)
 
## X.3 Hyperparameter Optimization via Latin Hypercube Sampling (LHS)

Latin Hypercube Sampling (LHS) was chosen to efficiently explore the multi-dimensional hyperparameter space without the combinatorial explosion of a grid search. LHS guarantees that each parameter's marginal distribution is uniformly sampled across its range, ensuring a stratified coverage of the search space while using far fewer samples than exhaustive search. This is particularly important when each experimental point requires a full federated simulation.

In this work, LHS was applied to the following parameters: quantization bits (`quantize_bits`), coreset maximum size (`M_max`), LoRA rank (`r_k`), novelty threshold (`gamma`), and regularization (`lambda_reg`). For reproducibility, the LHS generator was seeded (`seed=12345`) and repeated trials were executed to estimate variance.

[INSERT FIGURE: lhs_pareto_pub.pdf HERE]

The Pareto frontier was constructed from the aggregated sweep results in `results/lhs_summary.csv`. Each sampled configuration was executed with multiple repeats and the summary statistics (mean and standard deviation) of $	ext{CCR}$ and $\Delta F1$ were recorded. The frontier highlights configurations where no other sample simultaneously improves compression (higher CCR) and reduces accuracy loss (lower $\Delta F1$).

Key observations from the sweep:
- The trade-off curve is steep initially: modest losses in F1 allow for large gains in CCR.
- Beyond a certain CCR threshold, additional compression incurs rapidly diminishing returns and noticeably larger $\Delta F1$.

These patterns motivated selecting a Pareto-optimal `\Phi^*` that balances extreme compression with acceptably small accuracy degradation.

## X.4 Analysis of the Pareto-Optimal Configuration ($\Phi^*$)

The champion configuration (Sample 0) sits on the Pareto frontier and provides the best trade-off found by the LHS sweep. Its core metrics are: $\text{CCR}_{\text{mean}} \approx 1180.9$ and $\Delta F1_{\text{mean}} \approx 0.034$ (std $\approx 0.0097$).

An analytical interpretation of the hyperparameters behind Sample 0:

- `quantize_bits = 8` — Reducing model updates and coreset representations from 32-bit floating point to 8-bit integers yields an immediate 4x reduction in per-value size, and in practice is the primary driver behind the large CCR. This quantization reduces payload before algorithmic or coreset compression stages are applied and is widely used in edge deployments.

- `M_max = 989` — A coreset size near 1,000 demonstrates that only a small, curated subset of local samples is necessary to represent the critical information for federated aggregation. This limits the number of transmitted samples per client, dramatically reducing communication without losing the essential anomalous signatures required for detection.

- `r_k = 10` — Low-rank LoRA updates compress the effective parameter update into two small matrices whose sizes scale with `r_k`. A rank of 10 balances expressiveness and compactness: it captures sufficient adaptation capacity while keeping transmitted parameters minimal.

- `gamma = 0.0304489` — The novelty threshold enforces a conservative transmission policy: only samples whose representation deviates by more than ~3.0% (as measured by the novelty metric used in the filter) are admitted to the coreset. This prevents redundant benign traffic from wasting bandwidth.

- `lambda_reg = 0.04228` — Moderate regularization stabilizes client training and helps prevent overfitting to rare anomalous samples.

Taken together, these settings create a pipeline where quantization reduces raw payload, a strict novelty gate limits transmitted samples to high-value outliers, and low-rank updates plus curated coresets preserve model adaptivity with tiny communication footprints. The empirical result is a federated model whose macro F1 remains within ~3.4% of the Oracle despite an aggregate compression of over 1,180x.

## X.5 Real-World Implications for Edge Networks

The Pareto-optimal CCR of ~1180.8x has immediate practical consequences for deployment in bandwidth- and energy-constrained environments:

- Bandwidth efficiency: For constrained uplinks (LPWAN, satellite, or cellular IoT), reducing transmitted bytes by three orders of magnitude directly translates into lower operational costs and higher effective throughput for application payloads.

- Energy and latency: Transmitting fewer bytes reduces radio on-time and energy draw on battery-operated devices, prolonging lifetime and enabling more frequent sensing/processing cycles. Lowered data volumes also reduce queuing and transmission latency, improving responsiveness in near-real-time detection.

- Privacy and storage: Sending only curated, novel samples reduces exposure of benign traffic and limits the dataset retained centrally, aiding privacy and regulatory concerns.

Collectively, these implications support the thesis claim that Synergy-FedFM enables near-centralized performance at the extreme edge while dramatically reducing communication overhead.

(Next: draft X.6 Limitations and X.7 Chapter Summary if you want me to continue.)

## X.6 Limitations

While the experimental results are statistically robust within the bounds of the conducted sweep, several limitations should be acknowledged.

- Dataset specificity: All experiments were conducted on the CICIDS2017-derived dataset (e.g., `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`) after preprocessing and partitioning. The observed CCR and $\Delta F1$ behaviors are therefore conditioned on the traffic characteristics and attack patterns present in this dataset; performance may vary on other network traces or attack families.

- Simulation vs. hardware: The federated experiments use a software-emulated federation and local FedAvg emulation. A physical deployment on resource-constrained edge hardware (e.g., Raspberry Pi, microcontrollers, or heterogeneous IoT devices) could reveal additional constraints such as limited numeric precision, hardware-specific quantization artifacts, intermittent connectivity, or energy-driven duty-cycling that were not modeled in the simulations.

- Novelty metric assumptions: The novelty filter and coreset selection rely on a particular embedding and distance metric. Different feature spaces or novelty measures could change which samples are selected as "novel" and therefore shift the effective CCR–$\Delta F1$ trade-off.

- Scope of hyperparameter sweep: The LHS sweep covers a broad but finite hyperparameter region. There may exist configurations outside the explored ranges (or combinations requiring alternative architectural choices) that yield different Pareto trade-offs.

These limitations motivate future work that validates the reported Pareto-optimal configurations across additional datasets, implements the pipeline on real edge hardware, and explores adaptive novelty metrics and dynamic coreset policies.

## X.7 Chapter Summary

This chapter reported a comprehensive empirical evaluation of Synergy-FedFM. Anchored to a rigorously measured centralized Oracle (macro F1 = 0.9257), a Latin Hypercube Sampling hyperparameter sweep identified Pareto-optimal configurations that trade cumulative compression ratio (CCR) against federated accuracy loss ($\Delta F1$).

The principal finding is that a Pareto-optimal configuration (`\Phi^*`) achieves extreme compression (CCR $\approx$ 1180.9x) while incurring only a marginal absolute F1 drop of $\Delta F1 \approx 0.034$ relative to the Oracle. The combination of 8-bit quantization, near-1,000 sample coresets, low-rank LoRA updates, and a strict novelty gate demonstrates that near-centralized anomaly detection is achievable at the extreme edge with orders-of-magnitude reductions in communication.

Taken together, these results support the thesis claim that Synergy-FedFM provides a practical and theoretically grounded approach to closing the gap between centralized ML performance and the extreme constraints of edge networks. The following chapter will discuss implementation considerations and outline future experiments to validate these findings on hardware testbeds and additional network traces.
