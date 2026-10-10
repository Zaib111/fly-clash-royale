"""Arena background and combat effects."""
import math
import random

import pygame

from game.cards import CARDS
from game.sim import BRIDGES, HEIGHT, RIVER_HI, RIVER_LO, WIDTH
from . import art

GRASS_A = (122, 196, 78)
GRASS_B = (110, 184, 70)
DIRT = (214, 184, 122)
DIRT_DARK = (178, 146, 90)
WATER = (60, 150, 225)
WATER_DARK = (40, 110, 190)
WOOD = (160, 108, 60)
WOOD_DARK = (110, 70, 35)


class Arena:
    """Pre-rendered arena (grass, lanes, river banks, bridges, border); the water animates."""

    def __init__(self, tile, ox, oy):
        self.tile, self.ox, self.oy = tile, ox, oy
        self.w, self.h = int(WIDTH * tile), int(HEIGHT * tile)
        self.surface = self._render()

    def rect_world(self, x0, y0, x1, y1):
        """Screen rect for a world-space box (y is symmetric, so view flips don't matter)."""
        t = self.tile
        return pygame.Rect(int(x0 * t), int((HEIGHT - y1) * t), int((x1 - x0) * t), int((y1 - y0) * t))

    def _render(self):
        t = self.tile
        s = pygame.Surface((self.w, self.h))
        for gx in range(int(WIDTH)):
            for gy in range(int(HEIGHT)):
                s.fill(GRASS_A if (gx + gy) % 2 == 0 else GRASS_B, (gx * t, gy * t, t, t))
        # lanes: princess tower -> bridge, and across the back to the king tower
        for back, river_edge in ((3.0, RIVER_LO), (HEIGHT - 3.0, RIVER_HI)):
            lo, hi = min(back, river_edge), max(back, river_edge)
            pieces = [self.rect_world(bx - 0.9, lo - 0.9 * (back < 16), bx + 0.9, hi + 0.9 * (back > 16))
                      for bx in BRIDGES]
            pieces.append(self.rect_world(BRIDGES[0] - 0.9, back - 0.9, BRIDGES[1] + 0.9, back + 0.9))
            for r in pieces:
                pygame.draw.rect(s, DIRT_DARK, r.inflate(4, 4), border_radius=8)
            for r in pieces:
                pygame.draw.rect(s, DIRT, r, border_radius=8)
        # river with banks
        river = self.rect_world(0, RIVER_LO, WIDTH, RIVER_HI)
        pygame.draw.rect(s, (90, 140, 60), river.inflate(0, 8))
        pygame.draw.rect(s, WATER_DARK, river.inflate(0, 2))
        pygame.draw.rect(s, WATER, river.inflate(0, -6))
        # decorations: bushes and rocks along the sides
        rng = random.Random(3)
        for _ in range(26):
            x = rng.choice([rng.uniform(0.3, 1.2), rng.uniform(WIDTH - 1.2, WIDTH - 0.3)])
            y = rng.uniform(1, HEIGHT - 1)
            if RIVER_LO - 1 < y < RIVER_HI + 1:
                continue
            px, py = int(x * t), int((HEIGHT - y) * t)
            if rng.random() < 0.7:
                for dx, dy, rr in ((0, 0, 0.42), (-0.3, 0.15, 0.3), (0.3, 0.12, 0.32)):
                    art.circle(s, (60, 135, 50), (px + dx * t, py + dy * t + 2), rr * t)
                    art.circle(s, (80, 160, 60), (px + dx * t, py + dy * t), rr * t)
            else:
                art.circle(s, (110, 110, 120), (px, py + 2), 0.32 * t)
                art.circle(s, (160, 160, 170), (px, py), 0.32 * t)
        # bridges (drawn over the water)
        for bx in BRIDGES:
            r = self.rect_world(bx - 1.3, RIVER_LO - 0.35, bx + 1.3, RIVER_HI + 0.35)
            pygame.draw.rect(s, WOOD_DARK, r.move(0, 3), border_radius=4)
            pygame.draw.rect(s, WOOD, r, border_radius=4)
            for k in range(1, 8):
                yy = r.top + k * r.h // 8
                pygame.draw.line(s, WOOD_DARK, (r.left + 3, yy), (r.right - 3, yy), 1)
            for xx in (r.left, r.right - 5):
                pygame.draw.rect(s, WOOD_DARK, (xx, r.top - 2, 5, r.h + 4), border_radius=2)
        # border vignette
        edge = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        for k in range(6):
            pygame.draw.rect(edge, (0, 0, 0, 22), (k, k, self.w - 2 * k, self.h - 2 * k), 1)
        s.blit(edge, (0, 0))
        return s

    def draw(self, surf, now):
        surf.blit(self.surface, (self.ox, self.oy))
        t = self.tile
        river = self.rect_world(0, RIVER_LO, WIDTH, RIVER_HI).move(self.ox, self.oy)
        # animated ripples
        for row in range(3):
            y = river.top + 8 + row * (river.h - 16) / 2
            pts = []
            for k in range(0, river.w + 1, 6):
                pts.append((river.left + k, y + 2.5 * math.sin(k * 0.08 + now * 3 + row * 2)))
            for a, b in zip(pts[::2], pts[1::2]):
                if not any(abs(a[0] - (bx * t + self.ox)) < 1.3 * t for bx in BRIDGES):
                    pygame.draw.line(surf, (130, 195, 245), a, b, 2)
        # frame
        pygame.draw.rect(surf, (55, 45, 40), (self.ox - 6, self.oy - 6, self.w + 12, self.h + 12), 6,
                         border_radius=8)


# ------------------------------------------------------------------ effects
PROJECTILE_SPEED = 14.0   # tiles / second (visual only)


class Effects:
    def __init__(self):
        self.items = []

    def ingest(self, events, match):
        for ev in events:
            t0, kind = ev[0], ev[1]
            if kind == "attack":
                _, _, card, owner, x, y, tx, ty = ev
                c = CARDS[card]
                if c.range > 1.5:
                    style = {"Archers": "arrow", "Musketeer": "bullet", "Minions": "orb"}.get(c.name, "orb")
                    self._projectile(t0, style, x, y, tx, ty)
                else:
                    self.items.append(("slash", t0, 0.25, tx, ty, c.name == "MiniPekka" or c.name == "Giant"))
            elif kind == "tower_attack":
                _, _, is_king, owner, x, y, tx, ty = ev
                self._projectile(t0, "cannon" if is_king else "arrow", x, y, tx, ty)
            elif kind == "cast":
                _, _, card, owner, x, y = ev
                k = match.towers[owner][0]
                self.items.append(("fireball", t0, CARDS[card].spell_delay, k.x, k.y, x, y))
            elif kind == "boom":
                _, _, card, x, y = ev
                self.items.append(("boom", t0, 0.6, x, y, CARDS[card].spell_radius))
            elif kind == "death":
                _, _, is_building, owner, x, y, card = ev
                if is_building:
                    self.items.append(("boom", t0, 0.9, x, y, 2.5))
                else:
                    self.items.append(("puff", t0, 0.45, x, y))
        events.clear()

    def _projectile(self, t0, style, x, y, tx, ty):
        d = math.hypot(tx - x, ty - y)
        self.items.append(("proj", t0, max(0.08, d / PROJECTILE_SPEED), style, x, y, tx, ty))

    def draw(self, surf, now, to_screen, tile):
        keep = []
        for it in self.items:
            kind, t0, dur = it[0], it[1], it[2]
            p = (now - t0) / dur
            if p >= 1:
                if kind == "proj":   # spark on impact
                    keep.append(("spark", t0 + dur, 0.15, it[6], it[7]))
                continue
            keep.append(it)
            if p < 0:
                continue
            if kind == "proj":
                _, _, _, style, x, y, tx, ty = it
                cx, cy = x + (tx - x) * p, y + (ty - y) * p
                sx, sy = to_screen(cx, cy)
                hx, hy = to_screen(x + (tx - x) * max(0, p - 0.12), y + (ty - y) * max(0, p - 0.12))
                lift = math.sin(p * math.pi) * tile * (0.6 if style == "cannon" else 0.3)
                if style == "arrow":
                    pygame.draw.line(surf, (90, 60, 30), (hx, hy - lift), (sx, sy - lift), 3)
                    art.circle(surf, (230, 230, 230), (sx, sy - lift), 2)
                elif style == "bullet":
                    pygame.draw.line(surf, (255, 240, 160), (hx, hy), (sx, sy), 2)
                    art.circle(surf, (255, 220, 80), (sx, sy), 3)
                elif style == "cannon":
                    art.circle(surf, (40, 40, 45), (sx, sy - lift), tile * 0.28)
                    art.circle(surf, (110, 110, 120), (sx - 2, sy - lift - 2), tile * 0.1)
                else:
                    art.circle(surf, (190, 120, 255), (sx, sy - lift), tile * 0.2)
                    art.circle(surf, (240, 220, 255), (sx, sy - lift), tile * 0.09)
            elif kind == "spark":
                sx, sy = to_screen(it[3], it[4])
                r = tile * (0.2 + 0.4 * p)
                _ring(surf, (255, 250, 200), (sx, sy), r, int(220 * (1 - p)), 3)
            elif kind == "slash":
                _, _, _, x, y, big = it
                sx, sy = to_screen(x, y)
                r = tile * (0.9 if big else 0.6)
                s = pygame.Surface((int(r * 3), int(r * 3)), pygame.SRCALPHA)
                a0 = -2.4 + p * 1.6
                pygame.draw.arc(s, (255, 255, 255, int(255 * (1 - p))), (r * 0.5, r * 0.5, r * 2, r * 2),
                                a0, a0 + 1.6, 4 if big else 3)
                surf.blit(s, (sx - r * 1.5, sy - r * 1.5))
            elif kind == "fireball":
                _, _, _, x, y, tx, ty = it
                cx, cy = x + (tx - x) * p, y + (ty - y) * p
                sx, sy = to_screen(cx, cy)
                lift = math.sin(p * math.pi) * tile * 3
                for k in range(5):
                    q = max(0, p - k * 0.04)
                    qx, qy = to_screen(x + (tx - x) * q, y + (ty - y) * q)
                    art.circle(surf, (255, 160 - k * 20, 40), (qx, qy - math.sin(q * math.pi) * tile * 3),
                               tile * (0.55 - k * 0.08))
                art.draw_unit(surf, "Fireball", (sx, sy - lift), tile * 0.6, 0, shadow=False)
            elif kind == "boom":
                _, _, _, x, y, rad = it
                sx, sy = to_screen(x, y)
                r = rad * tile * (0.4 + 0.6 * math.sqrt(p))
                a = int(230 * (1 - p))
                s = pygame.Surface((int(r * 2 + 4), int(r * 2 + 4)), pygame.SRCALPHA)
                c = (int(r + 2), int(r + 2))
                pygame.draw.circle(s, (255, 120, 30, a), c, int(r))
                pygame.draw.circle(s, (255, 210, 90, a), c, int(r * 0.65))
                pygame.draw.circle(s, (255, 250, 220, a), c, int(r * 0.3))
                surf.blit(s, (sx - c[0], sy - c[1]))
            elif kind == "puff":
                sx, sy = to_screen(it[3], it[4])
                for k, (dx, dy) in enumerate(((0, 0), (-0.4, -0.2), (0.4, -0.25), (0, -0.5))):
                    r = tile * (0.3 + 0.5 * p) * (1 - 0.15 * k)
                    _disc(surf, (235, 235, 240), (sx + dx * tile * p, sy + dy * tile * p), r,
                          int(200 * (1 - p)))
        self.items = keep


def _disc(surf, color, center, r, alpha):
    r = max(1, int(r))
    s = pygame.Surface((2 * r + 2, 2 * r + 2), pygame.SRCALPHA)
    pygame.draw.circle(s, color + (max(0, alpha),), (r + 1, r + 1), r)
    surf.blit(s, (center[0] - r - 1, center[1] - r - 1))


def _ring(surf, color, center, r, alpha, width):
    r = max(2, int(r))
    s = pygame.Surface((2 * r + 2, 2 * r + 2), pygame.SRCALPHA)
    pygame.draw.circle(s, color + (max(0, alpha),), (r + 1, r + 1), r, width)
    surf.blit(s, (center[0] - r - 1, center[1] - r - 1))
