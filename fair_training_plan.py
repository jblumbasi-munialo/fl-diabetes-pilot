# fair_training_plan.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
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

class FairDiabetesTrainingPlan(TorchTrainingPlan):
    """Training plan that also computes fairness metrics on validation set."""

    def init_model(self, model_args):
        return TinyDiabetesNet(input_dim=model_args.get('input_dim', 8))

    def init_optimizer(self, optimizer_args):
        return torch.optim.Adam(self.model().parameters(), lr=optimizer_args.get('lr', 0.001))

    def training_step(self, data, target):
        output = self.model()(data)
        loss = nn.BCELoss()(output, target.view(-1, 1))
        return loss

    def fairness_evaluation(self, val_loader, sensitive_columns):
        """
        Evaluate fairness metrics on the local validation set.
        Args:
            val_loader: DataLoader containing features, target, and sensitive attributes.
            sensitive_columns: List of column names (strings) that contain sensitive groups.
        Returns:
            Dictionary with fairness metrics per sensitive attribute.
        """
        self.model().eval()
        all_preds = []
        all_targets = []
        sensitive_values = {col: [] for col in sensitive_columns}

        with torch.no_grad():
            for batch in val_loader:
                # Assuming batch is a tuple (features, target, sens_df)
                # Here we need to adapt to your actual data loader structure.
                # For simplicity, we assume batch[0] = features, batch[1] = target,
                # and batch[2] is a dict or list of sensitive tensors.
                # Modify according to your DataLoader implementation.
                features = batch[0]
                targets = batch[1]
                outputs = self.model()(features)
                preds = (outputs > 0.5).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy().flatten())
                # Collect sensitive attributes (as strings or categories)
                for i, col in enumerate(sensitive_columns):
                    # Assume batch[2][i] contains the sensitive values for this batch
                    sensitive_values[col].extend(batch[2][i].cpu().numpy().flatten())

        # Convert to arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        fairness_results = {}
        for col in sensitive_columns:
            sens = np.array(sensitive_values[col])
            # Convert to pandas Series for fairlearn
            sens_series = pd.Series(sens)
            dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sens_series)
            eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sens_series)
            # Per-group accuracy
            groups = np.unique(sens)
            group_acc = {}
            for g in groups:
                mask = (sens == g)
                if mask.sum() > 0:
                    acc = accuracy_score(y_true[mask], y_pred[mask])
                    group_acc[str(g)] = acc
            fairness_results[col] = {
                "demographic_parity_diff": dp_diff,
                "equalized_odds_diff": eo_diff,
                "group_accuracies": group_acc,
                "worst_group_accuracy": min(group_acc.values()) if group_acc else np.nan
            }
        return fairness_results
