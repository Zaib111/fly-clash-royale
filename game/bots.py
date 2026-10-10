"""Non-learning opponents used to *measure* the fly (never as training targets)."""
import math
import random

import numpy as np

from .cards import CARD_INDEX, CARDS
from .env import NUM_ACTIONS, action_mask, encode_action, world_to_cell
from .sim import BRIDGES, HEIGHT, Match


class RandomBot:
    """Plays a random legal card at a random legal spot, with probability `play_prob`."""

    def __init__(self, play_prob=0.15, seed=None):
        self.play_prob = play_prob
        self.rng = np.random.default_rng(seed)

    def act(self, m: Match, p: int) -> int:
        mask = action_mask(m, p)
        mask[0] = False
        if not mask.any() or self.rng.random() > self.play_prob:
            return 0
        return int(self.rng.choice(np.flatnonzero(mask)))


def _own_y(p, y):
    """World y -> y from player p's perspective (own king at the bottom)."""
    return y if p == 0 else HEIGHT - y


def _world_y(p, y):
    return y if p == 0 else HEIGHT - y


ANTI_AIR = ("Musketeer", "Archers", "Minions")
ANTI_TANK = ("MiniPekka", "Knight", "Minions")
GENERAL = ("Knight", "Musketeer", "MiniPekka", "Archers", "Goblins", "Minions")


class ScriptedBot:
    """Simple rule-based player: defends what crosses the river, fireballs clumps,
    and builds a Giant push when it has a full elixir bar."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def _action(self, m, p, slot, x, own_y):
        cx, cy = world_to_cell(p, x, _world_y(p, own_y))
        return encode_action(slot, cx, cy)

    def _slot_for(self, m, p, names):
        hand = m.hand(p)
        for name in names:
            cid = CARD_INDEX[name]
            if cid in hand and m.elixir[p] >= CARDS[cid].cost:
                return hand.index(cid)
        return None

    def act(self, m: Match, p: int) -> int:
        enemy = 1 - p
        threats = [u for u in m.units
                   if u.owner == enemy and u.deploy <= 0 and _own_y(p, u.y) < 17.5]
        if threats:
            # fireball a clump of 3+
            fb = self._slot_for(m, p, ("Fireball",))
            if fb is not None:
                for u in threats:
                    near = [o for o in threats if math.hypot(o.x - u.x, o.y - u.y) < 2.5]
                    if len(near) >= 3:
                        x = sum(o.x for o in near) / len(near)
                        y = sum(_own_y(p, o.y) for o in near) / len(near)
                        return self._action(m, p, fb, x, y)
            # counter the most advanced threat, unless we already have defenders there
            t = min(threats, key=lambda u: _own_y(p, u.y))
            defenders = [u for u in m.units if u.owner == p and
                         math.hypot(u.x - t.x, u.y - t.y) < 5 and not CARDS[u.card].buildings_only]
            if len(defenders) < 2:
                tc = CARDS[t.card]
                prefs = ANTI_AIR if tc.is_air else ANTI_TANK if tc.hp > 1000 else GENERAL
                slot = self._slot_for(m, p, prefs)
                if slot is not None:
                    y = max(1.0, min(13.0, _own_y(p, t.y) - 3.0))
                    return self._action(m, p, slot, t.x, y)
            return 0

        if m.elixir[p] >= 9.5:
            # push down the lane of the weaker enemy princess tower
            et = m.towers[enemy]
            left = et[1].hp if et[1].alive else 0
            right = et[2].hp if et[2].alive else 0
            bx = BRIDGES[0] if left <= right else BRIDGES[1]
            slot = self._slot_for(m, p, ("Giant",))
            if slot is not None:
                return self._action(m, p, slot, bx, 12.5)
            support = [u for u in m.units if u.owner == p and CARDS[u.card].buildings_only]
            y = 9.0 if support else 12.5
            x = support[0].x if support else bx
            slot = self._slot_for(m, p, GENERAL)
            if slot is not None:
                return self._action(m, p, slot, x, y)
        return 0
