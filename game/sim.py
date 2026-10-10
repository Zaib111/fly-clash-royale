"""Deterministic tick-based Clash Royale-like arena.

World coordinates: x in [0, 18), y in [0, 32). Player 0 lives at the bottom
(y < 15), player 1 at the top (y > 17). The river spans 15 <= y <= 17 and can
only be crossed by ground troops on the two bridges.
"""
import math
import random

from .cards import (CARDS, DEPLOY_TIME, ELIXIR_PER_SEC, KING, MAX_ELIXIR, NUM_CARDS,
                    PRINCESS, START_ELIXIR)

WIDTH, HEIGHT = 18.0, 32.0
RIVER_LO, RIVER_HI = 15.0, 17.0
BRIDGES = (3.5, 14.5)
TICKS_PER_SEC = 10
DT = 1.0 / TICKS_PER_SEC
MATCH_SECONDS = 180.0
DOUBLE_ELIXIR_AT = 120.0   # last 60s


def mirror_y(y: float) -> float:
    return HEIGHT - y


class Tower:
    __slots__ = ("owner", "is_king", "x", "y", "hp", "max_hp", "damage", "hit_speed",
                 "range", "radius", "cooldown", "target", "active", "alive")

    def __init__(self, owner, is_king, x, y):
        s = KING if is_king else PRINCESS
        self.owner, self.is_king, self.x, self.y = owner, is_king, x, y
        self.hp = self.max_hp = s.hp
        self.damage, self.hit_speed, self.range, self.radius = s.damage, s.hit_speed, s.range, s.radius
        self.cooldown = 0.0
        self.target = None
        self.active = not is_king
        self.alive = True

    is_air = False
    is_building = True


class Unit:
    __slots__ = ("owner", "card", "x", "y", "hp", "max_hp", "deploy", "cooldown", "target",
                 "alive")

    is_building = False

    def __init__(self, owner, card_id, x, y):
        c = CARDS[card_id]
        self.owner, self.card, self.x, self.y = owner, card_id, x, y
        self.hp = self.max_hp = c.hp
        self.deploy = DEPLOY_TIME
        self.cooldown = 0.0
        self.target = None
        self.alive = True

    @property
    def stats(self):
        return CARDS[self.card]

    @property
    def is_air(self):
        return CARDS[self.card].is_air

    @property
    def radius(self):
        return CARDS[self.card].radius


class PendingSpell:
    __slots__ = ("owner", "card", "x", "y", "delay")

    def __init__(self, owner, card_id, x, y):
        self.owner, self.card, self.x, self.y = owner, card_id, x, y
        self.delay = CARDS[card_id].spell_delay


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


class Match:
    """One 1v1 match. Call `play()` for each player's decision, then `step()`."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.time = 0.0
        self.ticks = 0
        self.units: list[Unit] = []
        self.spells: list[PendingSpell] = []
        self.towers: list[list[Tower]] = [[], []]
        for p in (0, 1):
            ys = (lambda y: y) if p == 0 else mirror_y
            self.towers[p] = [
                Tower(p, True, 9.0, ys(3.0)),
                Tower(p, False, 3.5, ys(6.5)),
                Tower(p, False, 14.5, ys(6.5)),
            ]
        self.elixir = [START_ELIXIR, START_ELIXIR]
        self.cycle = []
        for _ in (0, 1):
            deck = list(range(NUM_CARDS))
            self.rng.shuffle(deck)
            self.cycle.append(deck)   # first 4 = hand, 5th = next card
        self.winner = None            # 0, 1, or -1 for draw
        self.done = False
        self.tower_damage_dealt = [0.0, 0.0]
        self.events = None            # set to a list to record visual events (viewer only)

    def _emit(self, *ev):
        if self.events is not None:
            self.events.append((self.time,) + ev)

    # ------------------------------------------------------------------ queries
    def hand(self, p):
        return self.cycle[p][:4]

    def next_card(self, p):
        return self.cycle[p][4]

    def crowns(self, p):
        """Crowns earned by player p (enemy towers destroyed; king counts 3)."""
        enemy = self.towers[1 - p]
        if not enemy[0].alive:
            return 3
        return sum(1 for t in enemy[1:] if not t.alive)

    def double_elixir(self):
        return self.time >= DOUBLE_ELIXIR_AT

    def time_left(self):
        return max(0.0, MATCH_SECONDS - self.time)

    @staticmethod
    def can_deploy_troop(p, x, y):
        if not (0 <= x < WIDTH):
            return False
        return 0 <= y < 14.0 if p == 0 else HEIGHT - 14.0 < y < HEIGHT

    def can_play(self, p, slot, x, y):
        if self.done or not (0 <= slot < 4):
            return False
        card = CARDS[self.cycle[p][slot]]
        if self.elixir[p] < card.cost:
            return False
        if card.kind == "spell":
            return 0 <= x < WIDTH and 0 <= y < HEIGHT
        return self.can_deploy_troop(p, x, y)

    # ------------------------------------------------------------------ actions
    def play(self, p, slot, x, y):
        """Play the card in hand slot `slot` at (x, y). Returns False if illegal."""
        if not self.can_play(p, slot, x, y):
            return False
        # the next card fills the played slot; the played card goes to the back of the cycle
        cyc = self.cycle[p]
        card_id = cyc[slot]
        cyc[slot] = cyc.pop(4)
        cyc.append(card_id)
        card = CARDS[card_id]
        self.elixir[p] -= card.cost
        if card.kind == "spell":
            self.spells.append(PendingSpell(p, card_id, x, y))
            self._emit("cast", card_id, p, x, y)
        else:
            for i in range(card.count):
                if card.count == 1:
                    ox = oy = 0.0
                else:
                    ang = 2 * math.pi * i / card.count
                    ox, oy = 0.6 * math.cos(ang), 0.6 * math.sin(ang)
                ux = min(max(x + ox, 0.3), WIDTH - 0.3)
                self.units.append(Unit(p, card_id, ux, y + oy))
        return True

    # ------------------------------------------------------------------ simulation
    def step(self, ticks=1):
        for _ in range(ticks):
            if self.done:
                return
            self._tick()

    def _tick(self):
        rate = ELIXIR_PER_SEC * (2.0 if self.double_elixir() else 1.0)
        for p in (0, 1):
            self.elixir[p] = min(MAX_ELIXIR, self.elixir[p] + rate * DT)

        self._resolve_spells()
        for u in self.units:
            if u.alive:
                self._unit_tick(u)
        for p in (0, 1):
            for t in self.towers[p]:
                if t.alive and t.active:
                    self._tower_tick(t)
        self.units = [u for u in self.units if u.alive]

        self.ticks += 1
        self.time = self.ticks * DT
        self._check_end()

    def _damage(self, attacker_owner, target, amount):
        if not target.alive:
            return
        target.hp -= amount
        if target.is_building:
            self.tower_damage_dealt[attacker_owner] += min(amount, target.hp + amount)
            # King wakes up when hit
            if target.is_king:
                target.active = True
        if target.hp <= 0:
            target.alive = False
            target.hp = 0.0
            self._emit("death", target.is_building, target.owner, target.x, target.y,
                       None if target.is_building else target.card)
            if target.is_building and not target.is_king:
                self.towers[target.owner][0].active = True

    def _resolve_spells(self):
        keep = []
        for s in self.spells:
            s.delay -= DT
            if s.delay > 0:
                keep.append(s)
                continue
            card = CARDS[s.card]
            enemy = 1 - s.owner
            self._emit("boom", s.card, s.x, s.y)
            for u in self.units:
                if u.owner == enemy and u.alive and u.deploy <= 0 and \
                        math.hypot(u.x - s.x, u.y - s.y) <= card.spell_radius + u.radius:
                    self._damage(s.owner, u, card.damage)
            for t in self.towers[enemy]:
                if t.alive and math.hypot(t.x - s.x, t.y - s.y) <= card.spell_radius + t.radius:
                    self._damage(s.owner, t, card.tower_damage)
        self.spells = keep

    def _valid_target(self, u, other):
        c = u.stats
        if not other.alive or other.owner == u.owner:
            return False
        if other.is_building:
            return True
        if c.buildings_only or other.deploy > 0:
            return False
        return c.hits_air or not other.is_air

    def _find_target(self, u):
        c = u.stats
        enemy = 1 - u.owner
        best, best_d = None, c.sight
        if not c.buildings_only:
            for o in self.units:
                if o.owner == enemy and self._valid_target(u, o):
                    d = _dist(u, o) - o.radius
                    if d < best_d:
                        best, best_d = o, d
        if best is None:
            # fall back to the nearest enemy tower (lane push)
            best_d = float("inf")
            for t in self.towers[enemy]:
                if t.alive:
                    d = _dist(u, t)
                    if d < best_d:
                        best, best_d = t, d
        return best

    def _waypoint(self, u, tx, ty):
        if u.is_air:
            return tx, ty
        my_bottom = u.y < 16.0
        t_bottom = ty < 16.0
        in_river = RIVER_LO <= u.y <= RIVER_HI
        if my_bottom == t_bottom and not in_river:
            return tx, ty
        if in_river:
            # on a bridge: keep walking straight across toward the target's side
            return u.x, (RIVER_LO - 0.6 if t_bottom else RIVER_HI + 0.6)
        bx = min(BRIDGES, key=lambda b: abs(u.x - b) + abs(b - tx))
        if abs(u.x - bx) > 0.2 or abs(u.y - 16.0) > 1.6:
            return bx, (RIVER_LO - 0.5 if my_bottom else RIVER_HI + 0.5)
        return bx, (RIVER_HI + 0.6 if my_bottom else RIVER_LO - 0.6)

    def _unit_tick(self, u):
        if u.deploy > 0:
            u.deploy -= DT
            return
        c = u.stats
        t = u.target
        # retarget unless locked onto something alive and in range
        if t is None or not t.alive or not self._in_range(u, t):
            t = self._find_target(u)
            if t is not u.target:
                u.cooldown = max(u.cooldown, c.hit_speed * 0.5)
            u.target = t
        if t is None:
            return
        if self._in_range(u, t):
            u.cooldown -= DT
            if u.cooldown <= 0:
                self._emit("attack", u.card, u.owner, u.x, u.y, t.x, t.y)
                self._damage(u.owner, t, c.damage)
                u.cooldown = c.hit_speed
            return
        u.cooldown = max(0.0, u.cooldown - DT)
        wx, wy = self._waypoint(u, t.x, t.y)
        dx, dy = wx - u.x, wy - u.y
        d = math.hypot(dx, dy)
        if d > 1e-6:
            step = min(d, c.speed * DT)
            u.x += dx / d * step
            u.y += dy / d * step

    @staticmethod
    def _in_range(u, t):
        return _dist(u, t) - t.radius - u.radius <= u.stats.range

    def _tower_tick(self, t):
        tgt = t.target
        if tgt is None or not tgt.alive or _dist(t, tgt) - tgt.radius > t.range:
            tgt, best = None, t.range
            for o in self.units:
                if o.owner != t.owner and o.alive and o.deploy <= 0:
                    d = _dist(t, o) - o.radius
                    if d <= best:
                        tgt, best = o, d
            t.target = tgt
        if tgt is None:
            t.cooldown = max(0.0, t.cooldown - DT)
            return
        t.cooldown -= DT
        if t.cooldown <= 0:
            self._emit("tower_attack", t.is_king, t.owner, t.x, t.y, tgt.x, tgt.y)
            self._damage(t.owner, tgt, t.damage)
            t.cooldown = t.hit_speed

    def _check_end(self):
        for p in (0, 1):
            if not self.towers[1 - p][0].alive:
                self.winner, self.done = p, True
                return
        if self.time >= MATCH_SECONDS:
            c0, c1 = self.crowns(0), self.crowns(1)
            if c0 != c1:
                self.winner = 0 if c0 > c1 else 1
            else:
                # tiebreak: whoever dealt more tower damage
                d0, d1 = self.tower_damage_dealt
                self.winner = -1 if abs(d0 - d1) < 1e-6 else (0 if d0 > d1 else 1)
            self.done = True

    def tower_hp_frac(self, p):
        tot = sum(t.max_hp for t in self.towers[p])
        return sum(t.hp for t in self.towers[p]) / tot
