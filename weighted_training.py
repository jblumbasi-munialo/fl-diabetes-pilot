# weighted_training.py
import torch
import pandas as pd
from fair_training_plan import TinyDiabetesNet  # reuse your model

def assign_weights(training_df, validation_df, model, sensitive_col='age_group'):
    """
    Compute group accuracies on validation set and assign weights to training samples.
    Args:
        training_df: DataFrame with features, target, and sensitive_col.
        validation_df: DataFrame with same columns.
        model: trained PyTorch model.
        sensitive_col: column name for group labels.
    Returns:
        training_df with added column 'sample_weight'.
    """
    model.eval()
    # Evaluate on validation set
    groups = validation_df[sensitive_col].unique()
    group_acc = {}
    with torch.no_grad():
        for g in groups:
            val_g = validation_df[validation_df[sensitive_col] == g]
            X = torch.tensor(val_g.drop(columns=['target', sensitive_col]).values, dtype=torch.float32)
            y = torch.tensor(val_g['target'].values, dtype=torch.float32).view(-1,1)
            outputs = model(X)
            preds = (outputs > 0.5).float()
            acc = (preds == y).float().mean().item()
            group_acc[g] = acc

    # Assign weights: inverse accuracy (normalised)
    weights = {g: 1.0 / (acc + 1e-6) for g, acc in group_acc.items()}
    # Normalize weights
    total = sum(weights.values())
    for g in weights:
        weights[g] = weights[g] / total * len(weights)

    training_df['sample_weight'] = training_df[sensitive_col].map(weights)
    return training_df

# Example usage inside a node's local training loop:
# model = TinyDiabetesNet()
# model.load_state_dict(torch.load('global_model.pt'))
# train_df = pd.read_csv('train_data.csv')
# val_df = pd.read_csv('val_data.csv')
# train_df = assign_weights(train_df, val_df, model, sensitive_col='age_group')
# Then train using weighted loss: loss = (BCELoss(output, target) * weight).mean()
