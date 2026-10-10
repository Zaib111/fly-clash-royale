"""Value network: estimates how likely the player is to win from the current observation.

Trained alongside the policy (it supplies PPO's advantages) but never used to pick moves;
the viewers show it as a "who's winning" bar.
"""
import torch
from torch import nn

from game.env import C, V


class Critic(nn.Module):
    """Value function on the raw observation (only used for training, not for playing)."""

    def __init__(self, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(32 * 8 * 5 + V, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, spatial, vector):
        x = self.conv(spatial).flatten(1)
        return self.head(torch.cat([x, vector], 1)).squeeze(1)
