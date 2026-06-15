#!/usr/bin/env python3
"""
End‑to‑end federated learning with fairness evaluation and automatic mitigation.
Author: Your Name
Date: 2025-06-15
"""

import os
import json
import numpy as np
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

# -------------------------------
# Configuration
# -------------------------------
NODE_IDS = ["node_A_uuid", "node_B_uuid"]   # Replace with actual node IDs
THRESHOLD_DP_DIFF = 0.1                    # Maximum allowed demographic parity difference
THRESHOLD_EO_DIFF = 0.1                    # Maximum allowed equalised odds difference
REWEIGHTING_ROUNDS = 3                      # Additional rounds with re‑weighting

# Path to store fairness results
FAIRNESS_LOG = "fairness_log.json"

# -------------------------------
# Step 1: Standard Federated Training
# -------------------------------
print("=== Phase 1: Standard Federated Training ===")
exp = Experiment(
    tags=['diabetes', 'fairness_baseline'],
    training_plan_class='fair_training_plan.FairDiabetesTrainingPlan',
    training_plan_kwargs={'model_args': {'input_dim': 8}},
    round_limit=10,
    aggregator=FedAverage(),
)
exp.add_break(node_ids=NODE_IDS)
exp.run()

# -------------------------------
# Step 2: Fairness Evaluation (Baseline)
# -------------------------------
print("\n=== Phase 2: Evaluating Fairness (Baseline) ===")
# We request each node to evaluate fairness on its validation set.
# Since remote method calls are not trivial, we use a trick:
# We set a flag in the experiment's job_args that the training plan will detect.
# We then run a single "dummy" round where the training_step does nothing,
# but the plan's `after_training` method (custom) saves fairness metrics to a file.
# We'll then read those files.

# For simplicity, we assume each node writes fairness metrics to a JSON file
# accessible to the researcher (e.g., a shared network drive or via experiment logs).
# Here we simulate by reading from local files (adjust for your setup).

def collect_fairness_metrics(node_ids):
    """Collect fairness metrics from nodes (simulated with local files)."""
    all_metrics = {}
    for nid in node_ids:
        # In a real scenario, you would retrieve from node's output or a shared location.
        # For this example, we assume each node saves to `fairness_{nid}.json`.
        fname = f"fairness_{nid}.json"
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                all_metrics[nid] = json.load(f)
        else:
            print(f"Warning: fairness file for {nid} not found. Using default.")
            all_metrics[nid] = {"age_group": {"demographic_parity_diff": 1.0, "equalized_odds_diff": 1.0}}
    return all_metrics

baseline_metrics = collect_fairness_metrics(NODE_IDS)

# Aggregate disparities across nodes
dp_diffs = [baseline_metrics[nid]['age_group']['demographic_parity_diff'] for nid in NODE_IDS]
eo_diffs = [baseline_metrics[nid]['age_group']['equalized_odds_diff'] for nid in NODE_IDS]
avg_dp = np.mean(dp_diffs)
avg_eo = np.mean(eo_diffs)

print(f"Baseline fairness: avg DP diff = {avg_dp:.3f}, avg EO diff = {avg_eo:.3f}")

# -------------------------------
# Step 3: Decide if mitigation is needed
# -------------------------------
if avg_dp <= THRESHOLD_DP_DIFF and avg_eo <= THRESHOLD_EO_DIFF:
    print("\nFairness criteria satisfied. No mitigation needed.")
    exit(0)

print("\n=== Phase 3: Mitigation – Re‑weighted Training ===")

# -------------------------------
# Step 4: Re‑weighted Training
# -------------------------------
# We create a new experiment using the reweighted training plan.
# We also pass a flag to compute sample weights before training.
exp_rew = Experiment(
    tags=['diabetes', 'fairness_mitigation'],
    training_plan_class='reweighted_training_plan.ReweightedTrainingPlan',
    training_plan_kwargs={
        'model_args': {'input_dim': 8},
        'use_reweighting': True,            # custom flag to enable reweighting
        'fairness_metric': 'group_accuracy' # which metric to use for weights
    },
    round_limit=REWEIGHTING_ROUNDS,
    aggregator=FedAverage(),
)
exp_rew.add_break(node_ids=NODE_IDS)
# Optionally, we could initialise with the previous model weights (warm start).
# In Fed‑BioMed, you can set `exp_rew.set_initial_model(exp.model())`.
exp_rew.run()

# -------------------------------
# Step 5: Re‑evaluate Fairness after Mitigation
# -------------------------------
print("\n=== Phase 4: Re‑evaluating Fairness ===")
mitigated_metrics = collect_fairness_metrics(NODE_IDS)
dp_diffs_m = [mitigated_metrics[nid]['age_group']['demographic_parity_diff'] for nid in NODE_IDS]
eo_diffs_m = [mitigated_metrics[nid]['age_group']['equalized_odds_diff'] for nid in NODE_IDS]
avg_dp_m = np.mean(dp_diffs_m)
avg_eo_m = np.mean(eo_diffs_m)

print(f"Mitigated fairness: avg DP diff = {avg_dp_m:.3f} (improvement = {avg_dp - avg_dp_m:.3f})")
print(f"Mitigated fairness: avg EO diff = {avg_eo_m:.3f} (improvement = {avg_eo - avg_eo_m:.3f})")

# Save final fairness report
report = {
    "baseline": {"avg_dp_diff": float(avg_dp), "avg_eo_diff": float(avg_eo)},
    "mitigated": {"avg_dp_diff": float(avg_dp_m), "avg_eo_diff": float(avg_eo_m)},
    "thresholds": {"dp": THRESHOLD_DP_DIFF, "eo": THRESHOLD_EO_DIFF},
    "nodes": baseline_metrics
}
with open(FAIRNESS_LOG, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nFull fairness report saved to {FAIRNESS_LOG}")
