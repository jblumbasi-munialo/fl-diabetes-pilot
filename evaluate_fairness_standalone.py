# Run this on each node after training to generate fairness JSON
from fair_training_plan import FairDiabetesTrainingPlan
import torch
# Load model
model = FairDiabetesTrainingPlan()
model.load_state(torch.load('latest_model.pt'))
# Load validation set
val_loader = ... # your DataLoader
sensitive_cols = ['age_group', 'gender']
output_path = f"fairness_{node_id}.json"
model.fairness_evaluation(val_loader, sensitive_cols, output_path)
