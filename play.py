"""Watch or play matches.

    python play.py                                    # scripted bot vs random bot
    python play.py --p1 human --p2 scripted           # you vs the scripted bot
    python play.py --p1 runs/fly1/latest.pt --p2 human
    python play.py --p1 runs/fly1/latest.pt --p2 runs/fly1/iter00050.pt

Players: human | random | scripted | <checkpoint.pt>
Human controls: click a card (or press 1-4), then click the arena. Space pauses,
+/- change speed, R restarts, Esc quits.
"""
import argparse
import os
import sys

import numpy as np
import pygame
import torch

from game.bots import RandomBot, ScriptedBot
from game.cards import CARDS, MAX_ELIXIR
from game.env import DECISION_TICKS, action_mask, apply_action, observe
from game.sim import BRIDGES, HEIGHT, RIVER_HI, RIVER_LO, TICKS_PER_SEC, WIDTH, Match

TILE = 22
AW, AH = int(WIDTH * TILE), int(HEIGHT * TILE)
TOP, BOTTOM = 40, 110
PANEL = 300
COLORS = {0: (60, 120, 230), 1: (220, 60, 60)}
CARD_COLORS = {
    "Knight": (200, 170, 90), "Archers": (230, 120, 190), "Giant": (200, 140, 60),
    "Goblins": (90, 200, 90), "Musketeer": (150, 110, 210), "MiniPekka": (90, 160, 220),
    "Fireball": (250, 140, 40), "Minions": (110, 110, 240),
}


class PolicyAgent:
    def __init__(self, path):
        from fly.brain import set_threads
        from fly.policies import load_checkpoint
        set_threads(2)
        self.policy, ck = load_checkpoint(path)
        self.is_fly = ck["cfg"]["policy"] == "fly"
        self.name = f"{ck['cfg']['policy']}:{os.path.basename(os.path.dirname(path))}"
        self.reset()

    def reset(self):
        self.state = self.policy.initial_state(1)
        self.activity = None

    @torch.no_grad()
    def act(self, m, p):
        sp, vec = observe(m, p)
        mask = action_mask(m, p)
        logits, self.state = self.policy(torch.from_numpy(sp)[None], torch.from_numpy(vec)[None],
                                         torch.from_numpy(mask)[None], self.state)
        if self.is_fly:
            self.activity = self.state[0, self.policy.read_idx].numpy()
        return int(torch.distributions.Categorical(logits=logits).sample())


class Human:
    name = "human"

    def __init__(self):
        self.selected = None
        self.pending = None   # (slot, x, y) in world coords

    def reset(self):
        self.selected = self.pending = None

    def act(self, m, p):
        return 0   # humans play through on_click, not on the decision clock


def make_player(spec, seed):
    if spec == "human":
        return Human()
    if spec == "random":
        b = RandomBot(seed=seed)
        b.name = "random"
        return b
    if spec == "scripted":
        b = ScriptedBot(seed=seed)
        b.name = "scripted"
        return b
    if spec.endswith(".pt"):
        return PolicyAgent(spec)
    sys.exit(f"unknown player {spec!r}")


class Viewer:
    def __init__(self, players, view, speed, show_brain):
        self.players = players
        self.view = view
        self.speed = speed
        flies = [p for p in players if isinstance(p, PolicyAgent) and p.is_fly]
        self.brain_agent = flies[0] if (flies and show_brain) else None
        width = AW + (PANEL if self.brain_agent else 0)
        self.screen = pygame.display.set_mode((width, TOP + AH + BOTTOM))
        pygame.display.set_caption("Fly Clash Royale")
        self.font = pygame.font.SysFont("menlo", 13)
        self.big = pygame.font.SysFont("menlo", 30, bold=True)
        self.seed = 0
        self.new_match()

    def new_match(self):
        self.seed += 1
        self.m = Match(seed=self.seed)
        for pl in self.players:
            pl.reset()
        self.paused = False

    # ------------------------------------------------------------------ coords
    def to_screen(self, x, y):
        if self.view == 1:
            y = HEIGHT - y
        return int(x * TILE), int(TOP + (HEIGHT - y) * TILE)

    def to_world(self, sx, sy):
        x = sx / TILE
        y = HEIGHT - (sy - TOP) / TILE
        if self.view == 1:
            y = HEIGHT - y
        return x, y

    # ------------------------------------------------------------------ loop
    def run(self, max_frames=None, screenshot=None):
        clock = pygame.time.Clock()
        acc, frames = 0.0, 0
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    return
                if ev.type == pygame.KEYDOWN:
                    self.on_key(ev.key)
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    self.on_click(*ev.pos)
            dt = clock.tick(60) / 1000
            if not self.paused and not self.m.done:
                acc += dt * self.speed * TICKS_PER_SEC
                while acc >= 1 and not self.m.done:
                    acc -= 1
                    if self.m.ticks % DECISION_TICKS == 0:
                        for p, pl in enumerate(self.players):
                            apply_action(self.m, p, pl.act(self.m, p))
                    self.m.step()
            self.draw()
            pygame.display.flip()
            frames += 1
            if max_frames and frames >= max_frames:
                if screenshot:
                    pygame.image.save(self.screen, screenshot)
                return

    def human_seat(self):
        for p, pl in enumerate(self.players):
            if isinstance(pl, Human):
                return p, pl
        return None, None

    def on_key(self, key):
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_PLUS, pygame.K_EQUALS):
            self.speed = min(16, self.speed * 2)
        elif key == pygame.K_MINUS:
            self.speed = max(0.25, self.speed / 2)
        elif key == pygame.K_r:
            self.new_match()
        elif pygame.K_1 <= key <= pygame.K_4:
            _, h = self.human_seat()
            if h:
                h.selected = key - pygame.K_1

    def on_click(self, sx, sy):
        p, h = self.human_seat()
        if h is None or self.m.done:
            return
        if sy > TOP + AH and sx < AW:
            slot = (sx - 90) // ((AW - 100) // 4)
            if 0 <= slot < 4:
                h.selected = slot
            return
        if h.selected is not None and TOP <= sy <= TOP + AH and sx < AW:
            x, y = self.to_world(sx, sy)
            if self.m.play(p, h.selected, x, y):
                h.selected = None

    # ------------------------------------------------------------------ drawing
    def draw(self):
        s, m = self.screen, self.m
        s.fill((25, 25, 30))
        pygame.draw.rect(s, (70, 140, 70), (0, TOP, AW, AH))
        _, ry0 = self.to_screen(0, RIVER_HI if self.view == 0 else RIVER_LO)
        pygame.draw.rect(s, (60, 110, 200), (0, ry0, AW, int((RIVER_HI - RIVER_LO) * TILE)))
        for bx in BRIDGES:
            pygame.draw.rect(s, (140, 100, 60), (int((bx - 1.5) * TILE), ry0, 3 * TILE,
                                                  int((RIVER_HI - RIVER_LO) * TILE)))
        hp_seat, h = self.human_seat()
        if h and h.selected is not None:
            card = CARDS[m.hand(hp_seat)[h.selected]]
            if card.kind == "troop":
                y0 = 0 if hp_seat == 0 else HEIGHT - 14
                a = self.to_screen(0, y0 + 14)
                zone = pygame.Surface((AW, 14 * TILE), pygame.SRCALPHA)
                zone.fill((255, 255, 255, 35))
                s.blit(zone, (0, a[1]))

        for p in (0, 1):
            for t in m.towers[p]:
                if not t.alive:
                    continue
                x, y = self.to_screen(t.x, t.y)
                r = int(t.radius * TILE)
                col = COLORS[p] if t.active else tuple(c // 2 for c in COLORS[p])
                pygame.draw.rect(s, col, (x - r, y - r, 2 * r, 2 * r), border_radius=4)
                self.hp_bar(x, y - r - 6, 2 * r, t.hp / t.max_hp, p)
        for sp in m.spells:
            x, y = self.to_screen(sp.x, sp.y)
            pygame.draw.circle(s, (250, 140, 40), (x, y), int(CARDS[sp.card].spell_radius * TILE), 2)
        for u in m.units:
            x, y = self.to_screen(u.x, u.y)
            c = CARDS[u.card]
            r = max(5, int(c.radius * TILE))
            if u.deploy > 0:
                pygame.draw.circle(s, COLORS[u.owner], (x, y), r, 1)
                continue
            pygame.draw.circle(s, CARD_COLORS[c.name], (x, y), r)
            pygame.draw.circle(s, COLORS[u.owner], (x, y), r, 3 if c.is_air else 2)
            self.hp_bar(x - r, y - r - 5, 2 * r, u.hp / u.max_hp, u.owner)

        top_seat = 1 - self.view
        mins, secs = divmod(int(m.time_left()), 60)
        txt = (f"{self.players[top_seat].name}  crowns {m.crowns(top_seat)}   "
               f"{mins}:{secs:02d}{' x2' if m.double_elixir() else ''}   "
               f"elixir {m.elixir[top_seat]:.1f}   speed x{self.speed:g}")
        s.blit(self.font.render(txt, True, COLORS[top_seat]), (8, 12))
        self.draw_hand(self.view)
        if m.done:
            msg = "DRAW" if m.winner == -1 else f"{self.players[m.winner].name} WINS"
            img = self.big.render(msg, True, (255, 255, 255))
            s.blit(img, (AW // 2 - img.get_width() // 2, TOP + AH // 2 - 20))
            s.blit(self.font.render("R: rematch   Esc: quit", True, (230, 230, 230)),
                   (AW // 2 - 80, TOP + AH // 2 + 20))
        if self.paused:
            s.blit(self.big.render("PAUSED", True, (255, 255, 255)), (AW // 2 - 55, TOP + 20))
        if self.brain_agent:
            self.draw_brain()

    def hp_bar(self, x, y, w, frac, owner):
        pygame.draw.rect(self.screen, (20, 20, 20), (x, y, w, 4))
        pygame.draw.rect(self.screen, COLORS[owner], (x, y, int(w * max(0, frac)), 4))

    def draw_hand(self, p):
        s, m = self.screen, self.m
        y0 = TOP + AH + 8
        s.blit(self.font.render(f"{self.players[p].name}  crowns {m.crowns(p)}", True, COLORS[p]),
               (8, y0))
        el = m.elixir[p]
        pygame.draw.rect(s, (50, 30, 60), (8, y0 + 82, AW - 16, 14))
        pygame.draw.rect(s, (200, 70, 220), (8, y0 + 82, int((AW - 16) * el / MAX_ELIXIR), 14))
        s.blit(self.font.render(f"{el:.1f}", True, (255, 255, 255)), (12, y0 + 81))
        cw = (AW - 100) // 4
        _, h = self.human_seat()
        for i, cid in enumerate(m.hand(p)):
            c = CARDS[cid]
            x = 90 + i * cw
            col = CARD_COLORS[c.name] if el >= c.cost else (90, 90, 90)
            pygame.draw.rect(s, col, (x, y0 + 20, cw - 8, 56), border_radius=6)
            if h and isinstance(self.players[p], Human) and h.selected == i:
                pygame.draw.rect(s, (255, 255, 255), (x, y0 + 20, cw - 8, 56), 3, border_radius=6)
            s.blit(self.font.render(c.name, True, (20, 20, 20)), (x + 4, y0 + 26))
            s.blit(self.font.render(f"[{i + 1}] {c.cost}", True, (20, 20, 20)), (x + 4, y0 + 46))
        nxt = CARDS[m.next_card(p)]
        s.blit(self.font.render("next:", True, (200, 200, 200)), (8, y0 + 30))
        s.blit(self.font.render(nxt.name, True, CARD_COLORS[nxt.name]), (8, y0 + 48))

    def draw_brain(self):
        s = self.screen
        x0 = AW + 10
        s.blit(self.font.render("fly descending + motor", True, (230, 230, 230)), (x0, 12))
        s.blit(self.font.render("neurons (live)", True, (230, 230, 230)), (x0, 28))
        act = self.brain_agent.activity
        if act is None:
            return
        n = len(act)
        cols = int(np.ceil(np.sqrt(n * 0.8)))
        cell = max(2, (PANEL - 20) // cols)
        top = 60
        a = np.clip(act / (act.max() + 1e-6), 0, 1)
        for i, v in enumerate(a):
            r, c = divmod(i, cols)
            col = (int(30 + 225 * v), int(30 + 180 * v ** 2), int(60 + 40 * v))
            pygame.draw.rect(s, col, (x0 + c * cell, top + r * cell, cell - 1, cell - 1))
        s.blit(self.font.render(f"mean rate {act.mean():.3f}", True, (200, 200, 200)),
               (x0, top + (n // cols + 1) * cell + 8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", default="scripted", help="bottom player (seat 0)")
    ap.add_argument("--p2", default="random", help="top player (seat 1)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-brain", action="store_true", help="hide the neuron activity panel")
    ap.add_argument("--screenshot", help="run headless for --frames frames and save a PNG")
    ap.add_argument("--frames", type=int, default=600)
    args = ap.parse_args()
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    players = [make_player(args.p1, 1), make_player(args.p2, 2)]
    view = 1 if isinstance(players[1], Human) and not isinstance(players[0], Human) else 0
    v = Viewer(players, view, args.speed, not args.no_brain)
    v.run(max_frames=args.frames if args.screenshot else None, screenshot=args.screenshot)
    pygame.quit()


if __name__ == "__main__":
    main()
