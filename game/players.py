"""Players for the viewers: human, bots, and trained policies (fly or MLP)."""
import os
import sys

import numpy as np

from .bots import RandomBot, ScriptedBot
from .cards import CARDS
from .env import action_mask, decode_action, observe


class PolicyAgent:
    def __init__(self, path, threads=2):
        import torch
        from fly.brain import set_threads
        from fly.policies import load_checkpoint
        self.torch = torch
        set_threads(threads)
        self.policy, ck = load_checkpoint(path)
        self.cfg = ck["cfg"]
        self.critic = None            # value network, if the checkpoint has one
        if "critic" in ck:
            from fly.critic import Critic
            self.critic = Critic()
            self.critic.load_state_dict(ck["critic"])
            self.critic.eval()
        self.is_fly = self.cfg["policy"] == "fly"
        self.kind = "fly" if self.is_fly else "mlp"
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        self.name = f"{'Fly' if self.is_fly else 'MLP'} ({run})"
        self.reset()

    def reset(self):
        self.state = self.policy.initial_state(1)
        self.activity = None      # descending + motor neuron rates (fly only)
        self.top = []             # [(description, prob)] for the last decision
        self.probs = None         # full action distribution of the last decision
        self.obs = None           # spatial observation the fly saw
        self.last_action = 0

    def act(self, m, p):
        torch = self.torch
        sp, vec = observe(m, p)
        mask = action_mask(m, p)
        with torch.no_grad():
            logits, self.state = self.policy(torch.from_numpy(sp)[None], torch.from_numpy(vec)[None],
                                             torch.from_numpy(mask)[None], self.state)
            probs = torch.softmax(logits[0], 0)
            a = int(torch.multinomial(probs, 1))
        # group the 577 raw actions into readable options ("Knight - left, bridge") and keep the top 3
        options = {}
        for i in np.flatnonzero(probs.numpy() > 1e-4):
            d = self.describe(m, p, int(i))
            options[d] = options.get(d, 0.0) + float(probs[i])
        self.top = sorted(options.items(), key=lambda kv: -kv[1])[:3]
        self.probs, self.obs, self.last_action = probs.numpy(), sp, a
        if self.is_fly:
            self.activity = self.state[0, self.policy.read_idx].numpy()
        return a

    def win_chance(self, m, p):
        """Value network's estimate (0-1) that player p wins from here, or None."""
        if self.critic is None:
            return None
        torch = self.torch
        sp, vec = observe(m, p)
        with torch.no_grad():
            v = float(self.critic(torch.from_numpy(sp)[None], torch.from_numpy(vec)[None]))
        return min(1.0, max(0.0, (v + 1) / 2))

    def describe(self, m, p, a):
        """Human-readable action, e.g. 'Knight - left lane, bridge'."""
        d = decode_action(a)
        if d is None:
            return "Wait"
        slot, cx, cy = d
        card = CARDS[m.hand(p)[slot]].name
        lane = "left" if cx <= 2 else "middle" if cx <= 5 else "right"
        depth = ("back" if cy <= 2 else "mid" if cy <= 5 else "bridge" if cy == 6
                 else "river" if cy <= 8 else "enemy side")
        return f"{card} - {lane}, {depth}"


class Human:
    name = "You"
    kind = "human"

    def __init__(self):
        self.selected = None

    def reset(self):
        self.selected = None

    def act(self, m, p):
        return 0   # humans play through clicks, not on the decision clock


def make_player(spec, seed, threads=2):
    if spec == "human":
        return Human()
    if spec == "random":
        b = RandomBot(seed=seed)
        b.name, b.kind = "Random Bot", "bot"
        return b
    if spec == "scripted":
        b = ScriptedBot(seed=seed)
        b.name, b.kind = "Scripted Bot", "bot"
        return b
    if spec.endswith(".pt"):
        return PolicyAgent(spec, threads)
    sys.exit(f"unknown player {spec!r} (use human | random | scripted | path/to/checkpoint.pt)")
