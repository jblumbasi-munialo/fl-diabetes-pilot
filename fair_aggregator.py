from fedbiomed.researcher.aggregators import Aggregator
import numpy as np

class FairFedAvg(Aggregator):
    """Aggregator that weights nodes by inverse fairness disparity."""
    def __init__(self, fairness_metric='demographic_parity_diff', **kwargs):
        super().__init__(**kwargs)
        self.fairness_metric = fairness_metric

    def aggregate(self, models, fairness_scores=None):
        """
        fairness_scores: dict {node_id: score} where lower is better.
        """
        if fairness_scores is None:
            # fallback to standard FedAvg
            return super().aggregate(models)

        # Convert scores to weights (lower disparity -> higher weight)
        scores = np.array([fairness_scores[nid] for nid in self.node_ids])
        # Invert and normalise
        weights = 1.0 / (scores + 1e-6)
        weights = weights / weights.sum()
        # Weighted average of model parameters
        avg_params = {}
        for key in models[0].keys():
            avg_params[key] = sum(w * m[key].float() for w, m in zip(weights, models))
        return avg_params
