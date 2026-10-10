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
import math
import os
import sys

import numpy as np
import pygame
import torch

from game.bots import RandomBot, ScriptedBot
from game.cards import CARDS, MAX_ELIXIR
from game.env import DECISION_TICKS, action_mask, apply_action, observe
from game.sim import DT, HEIGHT, TICKS_PER_SEC, WIDTH, Match
from ui import art
from ui.scene import Arena, Effects

TILE = 20
MARGIN = 14
HEADER = 68
FOOTER = 174
AW, AH = int(WIDTH * TILE), int(HEIGHT * TILE)
WIN_W = AW + 2 * MARGIN
WIN_H = HEADER + AH + FOOTER
PANEL = 290
BG = (34, 28, 48)


class PolicyAgent:
    def __init__(self, path):
        from fly.brain import set_threads
        from fly.policies import load_checkpoint
        set_threads(2)
        self.policy, ck = load_checkpoint(path)
        self.is_fly = ck["cfg"]["policy"] == "fly"
        run = os.path.basename(os.path.dirname(os.path.abspath(path)))
        self.name = f"{'Fly' if self.is_fly else 'MLP'} ({run})"
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
    name = "You"

    def __init__(self):
        self.selected = None

    def reset(self):
        self.selected = None

    def act(self, m, p):
        return 0   # humans play through clicks, not on the decision clock


def make_player(spec, seed):
    if spec == "human":
        return Human()
    if spec == "random":
        b = RandomBot(seed=seed)
        b.name = "Random Bot"
        return b
    if spec == "scripted":
        b = ScriptedBot(seed=seed)
        b.name = "Scripted Bot"
        return b
    if spec.endswith(".pt"):
        return PolicyAgent(spec)
    sys.exit(f"unknown player {spec!r}")


class Viewer:
    def __init__(self, players, view, speed, show_brain):
        self.players = players
        self.view = view            # which seat sits at the bottom of the screen
        self.speed = speed
        flies = [p for p in players if isinstance(p, PolicyAgent) and p.is_fly]
        self.brain_agent = flies[0] if (flies and show_brain) else None
        width = WIN_W + (PANEL if self.brain_agent else 0)
        self.screen = pygame.display.set_mode((width, WIN_H))
        pygame.display.set_caption("Fly Clash Royale")
        self.arena = Arena(TILE, MARGIN, HEADER)
        self.seed = 0
        self.new_match()

    def new_match(self):
        self.seed += 1
        self.m = Match(seed=self.seed)
        self.m.events = []
        self.fx = Effects()
        self.prev = {}               # id(obj) -> (x, y) before the last tick, for smooth motion
        self.last_hp = {}
        self.flash = {}
        for pl in self.players:
            pl.reset()
        self.paused = False
        self.alpha = 0.0

    # ------------------------------------------------------------------ coords
    def to_screen(self, x, y):
        if self.view == 1:
            y = HEIGHT - y
        return MARGIN + x * TILE, HEADER + (HEIGHT - y) * TILE

    def to_world(self, sx, sy):
        x = (sx - MARGIN) / TILE
        y = HEIGHT - (sy - HEADER) / TILE
        if self.view == 1:
            y = HEIGHT - y
        return x, y

    def team(self, seat):
        """Colour index: whoever sits at the bottom is blue, like in Clash Royale."""
        return 0 if seat == self.view else 1

    def now(self):
        return self.m.time + self.alpha * DT

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
            if max_frames:
                dt = 1 / 60
            if not self.paused and not self.m.done:
                acc += dt * self.speed * TICKS_PER_SEC
                while acc >= 1 and not self.m.done:
                    acc -= 1
                    self.tick()
                self.alpha = min(1.0, acc)
            self.fx.ingest(self.m.events, self.m)
            self.draw(frames / 60)
            pygame.display.flip()
            frames += 1
            if max_frames and frames >= max_frames:
                if screenshot:
                    pygame.image.save(self.screen, screenshot)
                return

    def tick(self):
        m = self.m
        if m.ticks % DECISION_TICKS == 0:
            for p, pl in enumerate(self.players):
                apply_action(m, p, pl.act(m, p))
        self.prev = {id(u): (u.x, u.y) for u in m.units}
        m.step()
        for obj in m.units + m.towers[0] + m.towers[1]:
            k = id(obj)
            if obj.hp < self.last_hp.get(k, obj.hp):
                self.flash[k] = self.now()
            self.last_hp[k] = obj.hp

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
                slot = key - pygame.K_1
                h.selected = None if h.selected == slot else slot

    def card_rects(self):
        cw, ch = 70, 88
        x0 = MARGIN + 70
        gap = (WIN_W - x0 - MARGIN - 4 * cw) // 3
        y = HEADER + AH + 18
        return [pygame.Rect(x0 + i * (cw + gap), y, cw, ch) for i in range(4)]

    def on_click(self, sx, sy):
        p, h = self.human_seat()
        if h is None or self.m.done or p != self.view:
            return
        for i, r in enumerate(self.card_rects()):
            if r.inflate(0, 20).collidepoint(sx, sy):
                h.selected = None if h.selected == i else i
                return
        if h.selected is not None and HEADER <= sy <= HEADER + AH and MARGIN <= sx <= MARGIN + AW:
            x, y = self.to_world(sx, sy)
            if self.m.play(p, h.selected, x, y):
                h.selected = None

    # ------------------------------------------------------------------ drawing
    def draw(self, wall_t):
        s, m = self.screen, self.m
        now = self.now()
        s.fill(BG)
        self.arena.draw(s, wall_t)
        self.draw_placement_hint()

        # draw back-to-front so nearer things overlap farther ones
        drawables = []
        for p in (0, 1):
            for tw in m.towers[p]:
                drawables.append((self.to_screen(tw.x, tw.y)[1], 0, tw))
        for u in m.units:
            px, py = self.prev.get(id(u), (u.x, u.y))
            x = px + (u.x - px) * self.alpha
            y = py + (u.y - py) * self.alpha
            sy = self.to_screen(x, y)[1] + (1000 if u.is_air else 0)
            drawables.append((sy, 1, (u, x, y)))
        drawables.sort(key=lambda d: (d[0], d[1]))
        for _, kind, obj in drawables:
            if kind == 0:
                self.draw_tower(obj, now, wall_t)
            else:
                self.draw_unit(*obj, now, wall_t)
        self.fx.draw(s, now, self.to_screen, TILE)
        self.draw_header()
        self.draw_footer(wall_t)
        self.draw_side_crowns()
        if m.done:
            self.draw_result()
        if self.paused:
            art.text(s, "PAUSED", (MARGIN + AW // 2, HEADER + 40), 30, anchor="center")
        if self.brain_agent:
            self.draw_brain()

    def flash_amount(self, obj, now):
        t0 = self.flash.get(id(obj))
        if t0 is None:
            return 0.0
        return min(1.0, max(0.0, 1 - (now - t0) / 0.15))

    def draw_tower(self, tw, now, wall_t):
        x, y = self.to_screen(tw.x, tw.y)
        half = tw.radius * TILE * 1.05
        if not tw.alive:
            art.draw_rubble(self.screen, (x, y), half, self.team(tw.owner))
            return
        art.draw_tower(self.screen, (x, y), half, self.team(tw.owner), tw.is_king, tw.active, wall_t,
                       self.flash_amount(tw, now))
        art.hp_bar(self.screen, x, int(y - half - 20), half * 2.1, tw.hp / tw.max_hp, self.team(tw.owner),
                   tw.hp, show_number=True)

    def draw_unit(self, u, x, y, now, wall_t):
        c = CARDS[u.card]
        sx, sy = self.to_screen(x, y)
        r = max(6.0, c.radius * TILE * 1.25)
        tgt = u.target if (u.target is not None and u.target.alive) else None
        if tgt is not None:
            tx, ty = self.to_screen(tgt.x, tgt.y)
            facing = math.atan2(ty - sy, tx - sx)
        else:
            ex, ey = self.to_screen(x, y + (1 if u.owner == 0 else -1))
            facing = math.atan2(ey - sy, ex - sx)
        lift = TILE * 0.9 if c.is_air else 0.0
        if u.deploy > 0:
            art.draw_unit(self.screen, c.name, (sx, sy), r, self.team(u.owner), facing, wall_t, alpha=120,
                          air_lift=lift)
            frac = u.deploy / 1.0
            rect = pygame.Rect(0, 0, int(r * 1.6), int(r * 1.6))
            rect.center = (int(sx), int(sy - lift))
            pygame.draw.arc(self.screen, (255, 255, 255), rect, math.pi / 2,
                            math.pi / 2 + 2 * math.pi * frac, 3)
            return
        art.draw_unit(self.screen, c.name, (sx, sy), r, self.team(u.owner), facing, wall_t + id(u) % 7,
                      flash=self.flash_amount(u, now), air_lift=lift)
        if u.hp < u.max_hp:
            art.hp_bar(self.screen, sx, int(sy - lift - r - 10), max(18, r * 2), u.hp / u.max_hp, self.team(u.owner))

    def draw_placement_hint(self):
        p, h = self.human_seat()
        if h is None or h.selected is None or self.m.done:
            return
        s = self.screen
        card = CARDS[self.m.hand(p)[h.selected]]
        mx, my = pygame.mouse.get_pos()
        if card.kind == "troop":
            # red tint where troops can't go (enemy half + river)
            y_lo = 14.0 if p == 0 else 0.0
            y_hi = HEIGHT if p == 0 else HEIGHT - 14.0
            a = self.to_screen(0, y_hi)
            b = self.to_screen(WIDTH, y_lo)
            r = pygame.Rect(int(min(a[0], b[0])), int(min(a[1], b[1])), int(abs(b[0] - a[0])),
                            int(abs(b[1] - a[1])))
            tint = pygame.Surface(r.size, pygame.SRCALPHA)
            tint.fill((230, 40, 40, 60))
            s.blit(tint, r.topleft)
        if HEADER <= my <= HEADER + AH and MARGIN <= mx <= MARGIN + AW:
            wx, wy = self.to_world(mx, my)
            ok = self.m.can_play(p, h.selected, wx, wy)
            if card.kind == "spell":
                rad = card.spell_radius * TILE
                ring = pygame.Surface((int(rad * 2 + 4), int(rad * 2 + 4)), pygame.SRCALPHA)
                pygame.draw.circle(ring, (255, 255, 255, 60), ring.get_rect().center, int(rad))
                pygame.draw.circle(ring, (255, 255, 255, 200), ring.get_rect().center, int(rad), 2)
                s.blit(ring, (mx - rad - 2, my - rad - 2))
            else:
                art.draw_unit(s, card.name, (mx, my), max(6, card.radius * TILE * 1.25), self.team(p),
                              -math.pi / 2, 0, alpha=150 if ok else 70)

    def draw_header(self):
        s, m = self.screen, self.m
        top_seat = 1 - self.view
        pygame.draw.rect(s, (24, 20, 36), (0, 0, s.get_width(), HEADER))
        banner = pygame.Rect(MARGIN, 10, 190, 40)
        pygame.draw.rect(s, art.TEAM_DARK[1], banner.move(0, 3), border_radius=8)
        pygame.draw.rect(s, art.TEAM[1], banner, border_radius=8)
        art.text(s, self.players[top_seat].name, (banner.x + 10, banner.centery), 17, anchor="midleft")
        # timer
        box = pygame.Rect(WIN_W - MARGIN - 112, 8, 112, 46)
        pygame.draw.rect(s, (12, 10, 20), box, border_radius=8)
        art.text(s, "Time left:", (box.centerx, box.y + 3), 12, (230, 220, 170), anchor="midtop")
        mins, secs = divmod(int(math.ceil(m.time_left())), 60)
        col = (255, 150, 60) if m.double_elixir() else (255, 255, 255)
        art.text(s, f"{mins}:{secs:02d}", (box.centerx, box.bottom - 2), 22, col, anchor="midbottom")
        if m.double_elixir():
            art.elixir_drop(s, (box.x - 22, box.centery), 11, "x2", 12)

    def draw_side_crowns(self):
        s, m = self.screen, self.m
        x = MARGIN + AW - 26
        for seat, y in ((1 - self.view, HEADER + AH // 2 - 70), (self.view, HEADER + AH // 2 + 22)):
            r = pygame.Rect(x - 4, y, 34, 48)
            pygame.draw.rect(s, (0, 0, 0), r.move(0, 2), border_radius=8)
            pygame.draw.rect(s, art.TEAM[self.team(seat)], r, border_radius=8)
            art.crown(s, (r.centerx, r.y + 15), 20, art.GOLD if m.crowns(seat) else (90, 90, 110))
            art.text(s, m.crowns(seat), (r.centerx, r.bottom - 2), 18, anchor="midbottom")

    def draw_footer(self, wall_t):
        s, m = self.screen, self.m
        p = self.view
        y0 = HEADER + AH + 6
        pygame.draw.rect(s, (52, 34, 70), (0, y0, s.get_width(), WIN_H - y0))
        pygame.draw.rect(s, (80, 56, 105), (0, y0, s.get_width(), 4))
        _, h = self.human_seat()
        is_human = isinstance(self.players[p], Human)
        el = m.elixir[p]
        # next card
        art.text(s, "Next:", (MARGIN + 28, y0 + 22), 13, anchor="center")
        nxt = CARDS[m.next_card(p)]
        art.draw_card(s, (MARGIN + 6, y0 + 34, 46, 58), nxt.name, nxt.cost, small=True, t=wall_t)
        for i, (cid, rect) in enumerate(zip(m.hand(p), self.card_rects())):
            c = CARDS[cid]
            art.draw_card(s, rect, c.name, c.cost, affordable=el >= c.cost,
                          selected=is_human and h is not None and h.selected == i, t=wall_t,
                          elixir_frac=min(1.0, el / c.cost))
        bar = pygame.Rect(MARGIN + 30, WIN_H - 34, WIN_W - 2 * MARGIN - 34, 22)
        art.draw_elixir_bar(s, bar, el, MAX_ELIXIR)
        name = pygame.Rect(0, 0, 150, 22)
        name.bottomright = (WIN_W - MARGIN, bar.top - 8)
        pygame.draw.rect(s, art.TEAM[0], name, border_radius=6)
        art.text(s, self.players[p].name, name.center, 13, anchor="center")

    def draw_result(self):
        s, m = self.screen, self.m
        shade = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 120))
        s.blit(shade, (0, 0))
        cx, cy = MARGIN + AW // 2, HEADER + AH // 2
        if m.winner == -1:
            art.text(s, "DRAW", (cx, cy - 20), 44, anchor="center")
        else:
            col = art.TEAM[self.team(m.winner)]
            banner = pygame.Rect(0, 0, 300, 70)
            banner.center = (cx, cy - 20)
            pygame.draw.rect(s, art.TEAM_DARK[self.team(m.winner)], banner.move(0, 4), border_radius=12)
            pygame.draw.rect(s, col, banner, border_radius=12)
            art.text(s, f"{self.players[m.winner].name}", (cx, cy - 34), 22, anchor="center")
            art.text(s, "WINS!", (cx, cy - 8), 26, art.GOLD, anchor="center")
            for k in range(3):
                art.crown(s, (cx - 50 + 50 * k, cy + 50), 38,
                          art.GOLD if k < m.crowns(m.winner) else (80, 80, 100))
        art.text(s, "R: rematch    Esc: quit", (cx, cy + 100), 15, anchor="center")

    def draw_brain(self):
        s = self.screen
        x0 = WIN_W
        pygame.draw.rect(s, (16, 14, 26), (x0, 0, PANEL, WIN_H))
        art.text(s, "Fly brain", (x0 + 14, 14), 20)
        art.text(s, "descending + motor neurons, live", (x0 + 14, 42), 12, (190, 190, 210))
        act = self.brain_agent.activity
        if act is None:
            return
        n = len(act)
        cols = int(np.ceil(np.sqrt(n * 0.85)))
        cell = max(2, (PANEL - 28) // cols)
        top = 70
        a = np.clip(act / (act.max() + 1e-6), 0, 1)
        for i, v in enumerate(a):
            r, c = divmod(i, cols)
            col = (int(25 + 230 * v), int(20 + 200 * v ** 1.5), int(60 + 120 * v ** 3))
            pygame.draw.rect(s, col, (x0 + 14 + c * cell, top + r * cell, cell - 1, cell - 1))
        y = top + (n // cols + 1) * cell + 12
        art.text(s, f"{n:,} neurons   mean rate {act.mean():.3f}", (x0 + 14, y), 12, (200, 200, 220))


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
