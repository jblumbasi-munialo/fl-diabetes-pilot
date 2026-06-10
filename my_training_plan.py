import torch
import torch.nn as nn
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

class DiabetesTrainingPlan(TorchTrainingPlan):
    def init_model(self, model_args):
        return TinyDiabetesNet(input_dim=model_args.get('input_dim', 8))
    
    def init_optimizer(self, optimizer_args):
        return torch.optim.Adam(self.model().parameters(), lr=optimizer_args.get('lr', 0.001))
    
    def training_step(self, data, target):
        output = self.model()(data)
        loss = nn.BCELoss()(output, target.view(-1, 1))
        return loss
