import torch
import torch.nn as nn

class MSDNetLoss(nn.Module):
    def __init__(self, exit_weights=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.exit_weights = exit_weights

    def forward(self, logits_list, targets):
        total_loss = 0.0
        n_exits = len(logits_list)

        if self.exit_weights is None:
            weights = [1.0 / n_exits] * n_exits
        else:
            weights = self.exit_weights

        for logit, w in zip(logits_list, weights):
            total_loss += w * self.ce_loss(logit, targets)

        return total_loss
