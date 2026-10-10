"""Batched two-player environment on top of `Match`.

Every observation and action is from the acting player's own perspective: the
arena is mirrored vertically for player 1 so both always "sit" at the bottom.

Observation:
  spatial: (C, GH, GW) planes on a 9x16 grid of 2x2-tile cells
  vector:  (V,) elixir, hand, next card, clock, crowns, tower HP
Action (discrete, A = 1 + 4 * GH * GW):
  0         wait
  1 + s*G + cy*GW + cx   play hand slot s at cell (cx, cy)
"""
import numpy as np

from .cards import CARDS, NUM_CARDS, TROOP_IDS
from .sim import HEIGHT, MATCH_SECONDS, Match, TICKS_PER_SEC

GW, GH = 9, 16
CELL = 2.0
G = GW * GH
NUM_ACTIONS = 1 + 4 * G
DECISION_TICKS = TICKS_PER_SEC // 2          # decide every 0.5s
TROOP_ROWS = 7                               # own-perspective rows where troops may land
_TROOP_PLANE = {cid: i for i, cid in enumerate(TROOP_IDS)}
NT = len(TROOP_IDS)
NUM_TROOP_TYPES = NT
C = 2 * NT + 4
V = 1 + 4 * NUM_CARDS + NUM_CARDS + 4 + 1 + 1 + 2 + 2 + 6
SHAPING = 0.5                                # reward per whole enemy tower-HP pool removed


def cell_to_world(p, cx, cy):
    x = cx * CELL + CELL / 2
    y = cy * CELL + CELL / 2
    return x, (y if p == 0 else HEIGHT - y)


def world_to_cell(p, x, y):
    if p == 1:
        y = HEIGHT - y
    cx = min(GW - 1, max(0, int(x // CELL)))
    cy = min(GH - 1, max(0, int(y // CELL)))
    return cx, cy


def decode_action(a):
    """-> None for wait, else (slot, cx, cy)."""
    if a == 0:
        return None
    a -= 1
    slot, rem = divmod(a, G)
    cy, cx = divmod(rem, GW)
    return slot, cx, cy


def encode_action(slot, cx, cy):
    return 1 + slot * G + cy * GW + cx


def apply_action(m: Match, p, a):
    d = decode_action(int(a))
    if d is None:
        return False
    slot, cx, cy = d
    x, y = cell_to_world(p, cx, cy)
    return m.play(p, slot, x, y)


def action_mask(m: Match, p, out=None):
    mask = np.zeros(NUM_ACTIONS, dtype=bool) if out is None else out
    mask[:] = False
    mask[0] = True
    if m.done:
        return mask
    for s, cid in enumerate(m.hand(p)):
        card = CARDS[cid]
        if m.elixir[p] < card.cost:
            continue
        base = 1 + s * G
        if card.kind == "spell":
            mask[base:base + G] = True
        else:
            mask[base:base + TROOP_ROWS * GW] = True
    return mask


def observe(m: Match, p, spatial=None, vector=None):
    sp = np.zeros((C, GH, GW), dtype=np.float32) if spatial is None else spatial
    vec = np.zeros(V, dtype=np.float32) if vector is None else vector
    sp[:] = 0
    vec[:] = 0
    for u in m.units:
        cx, cy = world_to_cell(p, u.x, u.y)
        plane = _TROOP_PLANE[u.card] + (0 if u.owner == p else NT)
        sp[plane, cy, cx] += u.hp / 1000.0
    for owner in (0, 1):
        plane = 2 * NT + (0 if owner == p else 1)
        for t in m.towers[owner]:
            cx, cy = world_to_cell(p, t.x, t.y)
            sp[plane, cy, cx] = t.hp / t.max_hp if t.alive else -1.0
    for s in m.spells:
        cx, cy = world_to_cell(p, s.x, s.y)
        sp[2 * NT + 2 + (0 if s.owner == p else 1), cy, cx] = 1.0

    q = 1 - p
    i = 0
    vec[i] = m.elixir[p] / 10.0; i += 1
    hand = m.hand(p)
    for s, cid in enumerate(hand):
        vec[i + s * NUM_CARDS + cid] = 1.0
    i += 4 * NUM_CARDS
    vec[i + m.next_card(p)] = 1.0; i += NUM_CARDS
    for s, cid in enumerate(hand):
        vec[i + s] = float(m.elixir[p] >= CARDS[cid].cost)
    i += 4
    vec[i] = m.time_left() / MATCH_SECONDS; i += 1
    vec[i] = float(m.double_elixir()); i += 1
    vec[i] = m.crowns(p) / 3.0; vec[i + 1] = m.crowns(q) / 3.0; i += 2
    vec[i] = float(m.towers[p][0].active); vec[i + 1] = float(m.towers[q][0].active); i += 2
    for k, t in enumerate(m.towers[p] + m.towers[q]):
        vec[i + k] = t.hp / t.max_hp
    i += 6
    assert i == V
    return sp, vec


class VecEnv:
    """N matches stepped in lockstep. Player 0 and player 1 each get an action per match."""

    def __init__(self, n, seed=0):
        self.n = n
        self._seed = seed
        self._next_seed = seed
        self.matches = [self._new_match() for _ in range(n)]
        self.spatial = np.zeros((n, 2, C, GH, GW), dtype=np.float32)
        self.vector = np.zeros((n, 2, V), dtype=np.float32)
        self.mask = np.zeros((n, 2, NUM_ACTIONS), dtype=bool)
        self._hp = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            self._refresh(i)

    def _new_match(self):
        self._next_seed += 1
        return Match(seed=self._next_seed)

    def _refresh(self, i):
        m = self.matches[i]
        for p in (0, 1):
            observe(m, p, self.spatial[i, p], self.vector[i, p])
            action_mask(m, p, self.mask[i, p])
            self._hp[i, p] = m.tower_hp_frac(p)

    def obs(self):
        return self.spatial, self.vector, self.mask

    def step(self, a0, a1, shaping=SHAPING):
        """Apply both players' actions, advance 0.5s.

        Returns (reward[n, 2], done[n], winner[n]). Finished matches are reset
        automatically; `winner` is 0/1, -1 for a draw, or -2 if still running.
        """
        n = self.n
        reward = np.zeros((n, 2), dtype=np.float32)
        done = np.zeros(n, dtype=bool)
        winner = np.full(n, -2, dtype=np.int64)
        for i, m in enumerate(self.matches):
            apply_action(m, 0, a0[i])
            apply_action(m, 1, a1[i])
            m.step(DECISION_TICKS)
            hp0, hp1 = m.tower_hp_frac(0), m.tower_hp_frac(1)
            r = shaping * ((self._hp[i, 1] - hp1) - (self._hp[i, 0] - hp0))
            if m.done:
                done[i] = True
                winner[i] = m.winner
                if m.winner in (0, 1):
                    r += 1.0 if m.winner == 0 else -1.0
                self.matches[i] = self._new_match()
            reward[i, 0], reward[i, 1] = r, -r
            self._refresh(i)
        return reward, done, winner
