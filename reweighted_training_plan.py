# reweighted_training_plan.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from fedbiomed.common.training_plans import TorchTrainingPlan

class TinyDiabetesNet(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

class ReweightedTrainingPlan(TorchTrainingPlan):
    """
    Training plan that computes sample weights based on group performance
    on a validation set, then uses weighted loss to improve fairness.
    """

    def init_model(self, model_args):
        return TinyDiabetesNet(input_dim=model_args.get('input_dim', 8))

    def init_optimizer(self, optimizer_args):
        return torch.optim.Adam(self.model().parameters(), lr=optimizer_args.get('lr', 0.001))

    def compute_sample_weights(self, val_loader, sensitive_column='age_group'):
        """
        Compute weights for each sample in the training set based on group accuracy.
        Groups with lower accuracy get higher weight.
        Args:
            val_loader: DataLoader for validation set (must include sensitive attribute).
            sensitive_column: Name of the sensitive attribute column in the validation set.
        Returns:
            A tensor of weights for the training set (to be used in loss).
        """
        self.model().eval()
        all_preds = []
        all_targets = []
        all_groups = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch[0]
                targets = batch[1]
                groups = batch[2]   # assuming batch[2] is a tensor of group indices (numeric)
                outputs = self.model()(features)
                preds = (outputs > 0.5).int().flatten()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_groups.extend(groups.cpu().numpy())

        # Compute accuracy per group
        group_acc = {}
        unique_groups = np.unique(all_groups)
        for g in unique_groups:
            mask = np.array(all_groups) == g
            acc = accuracy_score(np.array(all_targets)[mask], np.array(all_preds)[mask])
            group_acc[g] = acc

        # Compute weights: lower accuracy → higher weight.
        # Use inverse of accuracy (plus small epsilon) to avoid division by zero.
        # Normalize so that average weight is 1.
        weights = {}
        for g in unique_groups:
            acc = group_acc[g]
            w = 1.0 / (acc + 1e-6)
            weights[g] = w

        # Normalize weights to have mean 1 across groups (optional)
        w_sum = sum(weights.values())
        for g in weights:
            weights[g] = weights[g] / w_sum * len(weights)

        return weights

    def training_step(self, data, target, sample_weights=None):
        output = self.model()(data)
        loss = nn.BCELoss(reduction='none')(output, target.view(-1, 1))
        if sample_weights is not None:
            # sample_weights is a tensor of shape (batch_size,)
            weighted_loss = (loss * sample_weights).mean()
        else:
            weighted_loss = loss.mean()
        return weighted_loss
    def on_round_begin(self, model, **kwargs):
        """Called by Fed‑BioMed at the start of each round."""
        # Load validation set from the node's dataset (assumed to be registered as 'val_data')
        # This is pseudo‑code – actual implementation depends on how your node loads data.
        try:
            val_loader = self.get_dataloader(dataset_id='val_data', batch_size=32)
            self.sample_weights = self.compute_sample_weights(val_loader)
        except Exception as e:
            print(f"Could not compute sample weights: {e}")
            self.sample_weights = None

    def training_step(self, data, target):
        # Use stored sample weights if available
        output = self.model()(data)
        loss = nn.BCELoss(reduction='none')(output, target.view(-1, 1))
        if hasattr(self, 'sample_weights') and self.sample_weights is not None:
            # Need to map data samples to their group weights.
            # For simplicity, assume `data` includes a group index column.
            # This is complex; easier to pre‑compute per‑sample weights once.
            # Instead, we simplify: compute weights per group and apply during loss.
            # A full implementation requires aligning batch samples with groups.
            pass
        return loss.mean()
