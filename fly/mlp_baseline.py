"""Plain feed-forward policy with the same inputs/outputs as FlyBrain, for comparison."""
import torch
from torch import nn

from game.env import C, G, NUM_ACTIONS, V


class MLPPolicy(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C * G + V, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, NUM_ACTIONS),
        )
        with torch.no_grad():
            self.net[-1].bias[0] = 4.0   # same "mostly wait" start as the fly

    def initial_state(self, batch):
        return torch.zeros(batch, 1)

    def forward(self, spatial, vector, mask, state):
        x = torch.cat([spatial.flatten(1), vector], 1)
        logits = self.net(x).masked_fill(~mask, torch.finfo(torch.float32).min)
        return logits, state
