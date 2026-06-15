# run_fair_experiment.py
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
import numpy as np

# List of node IDs (you need to obtain them from your running nodes)
NODE_IDS = ["node_A_uuid", "node_B_uuid"]

# Step 1: Federated training (as before)
exp = Experiment(
    tags=['diabetes', 'fairness'],
    training_plan_class='fair_training_plan.FairDiabetesTrainingPlan',
    training_plan_kwargs={'model_args': {'input_dim': 8}},
    round_limit=10,
    aggregator=FedAverage(),
)
exp.add_break(node_ids=NODE_IDS)
exp.run()

# Step 2: After training, evaluate fairness on each node.
# We send a request to each node to run the fairness_evaluation method.
# Fed‑BioMed allows calling remote methods via `experiment.get_nodes()` and `node.remote_method_call()`.
# Here we assume we have a helper to send a command.
# The actual implementation depends on Fed‑BioMed's API. Below is a conceptual pattern.

fairness_results = {}

for node_id in NODE_IDS:
    # Get the node object (pseudo‑code – consult Fed‑BioMed docs for exact method)
    node = exp.get_node(node_id)
    # Call the fairness_evaluation method on the node. We need to pass the validation loader.
    # In practice, you would have pre‑registered validation datasets on each node.
    # For simplicity, we assume the node has a validation set named 'diabetes_val' with sensitive columns.
    result = node.remote_method_call(
        'fairness_evaluation',
        val_dataset='diabetes_val',
        sensitive_columns=['age_group', 'gender']   # columns present in that dataset
    )
    fairness_results[node_id] = result

# Step 3: Aggregate fairness metrics across nodes
# For each sensitive attribute, compute average DP diff, EO diff, and worst‑group accuracy across nodes.
aggregated = {}
for attr in ['age_group', 'gender']:
    dp_diffs = [fairness_results[n][attr]['demographic_parity_diff'] for n in NODE_IDS]
    eo_diffs = [fairness_results[n][attr]['equalized_odds_diff'] for n in NODE_IDS]
    worst_accs = [fairness_results[n][attr]['worst_group_accuracy'] for n in NODE_IDS]
    aggregated[attr] = {
        'avg_dp_diff': np.mean(dp_diffs),
        'max_dp_diff': np.max(dp_diffs),   # worst‑case disparity
        'avg_eo_diff': np.mean(eo_diffs),
        'max_eo_diff': np.max(eo_diffs),
        'avg_worst_acc': np.mean(worst_accs),
        'min_worst_acc': np.min(worst_accs)   # most disadvantaged group across nodes
    }

print("Aggregated fairness metrics across hospitals:")
print(aggregated)
