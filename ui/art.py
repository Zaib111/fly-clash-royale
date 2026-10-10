"""Procedural Clash Royale-style art: everything is drawn from shapes, no image assets."""
import math

import pygame
from pygame import gfxdraw

BLUE = (52, 120, 246)
BLUE_DARK = (25, 60, 150)
RED = (232, 52, 62)
RED_DARK = (130, 20, 30)
TEAM = {0: BLUE, 1: RED}
TEAM_DARK = {0: BLUE_DARK, 1: RED_DARK}
GOLD = (255, 205, 50)
ELIXIR = (214, 60, 220)
ELIXIR_DARK = (110, 20, 120)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SKIN = (245, 200, 160)
STONE = (150, 150, 162)
STONE_DARK = (95, 95, 108)
STONE_LIGHT = (190, 190, 200)

_fonts = {}


def font(size, bold=True):
    key = (size, bold)
    if key not in _fonts:
        for name in ("arialroundedmtbold", "supercell-magic", "avenirnext", "arial"):
            path = pygame.font.match_font(name, bold=bold)
            if path:
                _fonts[key] = pygame.font.Font(path, size)
                break
        else:
            _fonts[key] = pygame.font.SysFont(None, size, bold=bold)
    return _fonts[key]


def text(surf, s, pos, size=16, color=WHITE, outline=2, anchor="topleft"):
    """White text with a dark outline, the way Clash Royale draws everything."""
    f = font(size)
    base = f.render(str(s), True, color)
    w, h = base.get_width() + 2 * outline, base.get_height() + 2 * outline
    img = pygame.Surface((w, h), pygame.SRCALPHA)
    if outline:
        edge = f.render(str(s), True, (20, 20, 30))
        for dx in range(-outline, outline + 1):
            for dy in range(-outline, outline + 1):
                if dx * dx + dy * dy <= outline * outline + 1:
                    img.blit(edge, (outline + dx, outline + dy))
    img.blit(base, (outline, outline))
    r = img.get_rect(**{anchor: pos})
    surf.blit(img, r)
    return r


def circle(surf, color, center, r, width=0):
    x, y, r = int(center[0]), int(center[1]), max(1, int(r))
    if width == 0:
        gfxdraw.filled_circle(surf, x, y, r, color)
        gfxdraw.aacircle(surf, x, y, r, color)
    else:
        for k in range(width):
            gfxdraw.aacircle(surf, x, y, r - k, color)


def poly(surf, color, pts):
    pts = [(int(x), int(y)) for x, y in pts]
    gfxdraw.filled_polygon(surf, pts, color)
    gfxdraw.aapolygon(surf, pts, color)


def ellipse_alpha(surf, color, rect, alpha):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.ellipse(s, color + (alpha,), (0, 0, rect[2], rect[3]))
    surf.blit(s, (rect[0], rect[1]))


def shade(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c)


def rot(dx, dy, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    return dx * ca - dy * sa, dx * sa + dy * ca


def elixir_drop(surf, center, r, value=None, size=None):
    x, y = center
    pts = [(x, y - r * 1.55)] + [(x + r * math.cos(math.radians(a)), y + r * math.sin(math.radians(a)))
                                 for a in range(-30, 211, 12)]
    poly(surf, ELIXIR_DARK, [(px, py + 1.5) for px, py in pts])
    poly(surf, ELIXIR, pts)
    circle(surf, (245, 150, 250), (x - r * 0.35, y - r * 0.2), r * 0.25)
    if value is not None:
        text(surf, value, (x, y + 1), size or int(r * 1.3), anchor="center")


def crown(surf, center, w, color=GOLD, outline=True):
    x, y = center
    h = w * 0.7
    pts = [(x - w / 2, y + h / 2), (x - w / 2, y - h / 4), (x - w / 4, y + h / 8), (x, y - h / 2),
           (x + w / 4, y + h / 8), (x + w / 2, y - h / 4), (x + w / 2, y + h / 2)]
    if outline:
        poly(surf, shade(color, 0.45), [(px, py + 2) for px, py in pts])
    poly(surf, color, pts)


# ------------------------------------------------------------------ troops
def draw_unit(surf, name, pos, r, team, facing=0.0, t=0.0, flash=0.0, alpha=255, shadow=True,
              air_lift=0.0):
    """Draw one troop centered at pos with body radius r (pixels).

    facing: angle in radians (0 = right, screen coords); t: seconds for idle animation.
    """
    if alpha < 255:
        tmp = pygame.Surface((int(r * 6), int(r * 6)), pygame.SRCALPHA)
        draw_unit(tmp, name, (r * 3, r * 3), r, team, facing, t, flash, 255, shadow, air_lift)
        tmp.set_alpha(alpha)
        surf.blit(tmp, (pos[0] - r * 3, pos[1] - r * 3))
        return
    x, y = pos
    if shadow:
        sw = r * (2.2 if air_lift else 1.9)
        ellipse_alpha(surf, BLACK, (int(x - sw / 2), int(y + r * 0.35), int(sw), int(r * 0.8)), 70)
    y -= air_lift
    tc = TEAM[team]
    bob = math.sin(t * 8) * r * 0.06
    fn = _UNITS.get(name)
    if fn:
        fn(surf, x, y + bob, r, tc, facing, t)
    if flash > 0:
        s = pygame.Surface((int(r * 3), int(r * 3)), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 255, 255, int(180 * flash)), (int(r * 1.5), int(r * 1.5)), int(r * 1.05))
        surf.blit(s, (x - r * 1.5, y + bob - r * 1.5))


def _team_base(surf, x, y, r, tc):
    circle(surf, shade(tc, 0.55), (x, y + r * 0.12), r)
    circle(surf, tc, (x, y), r)


def _weapon(surf, x, y, r, facing, length, color, width, offset=0.55):
    sx, sy = rot(offset * r, 0, facing)
    ex, ey = rot((offset + length) * r, 0, facing)
    pygame.draw.line(surf, shade(color, 0.5), (x + sx, y + sy + 1), (x + ex, y + ey + 1), width + 2)
    pygame.draw.line(surf, color, (x + sx, y + sy), (x + ex, y + ey), width)


def _knight(surf, x, y, r, tc, facing, t):
    _weapon(surf, x, y, r, facing - 0.5, 1.0, (225, 230, 240), max(2, int(r * 0.22)))
    _team_base(surf, x, y, r, tc)
    circle(surf, (175, 180, 192), (x, y - r * 0.08), r * 0.74)              # helmet
    circle(surf, (205, 210, 220), (x - r * 0.2, y - r * 0.3), r * 0.25)
    pygame.draw.line(surf, (40, 40, 50), (x - r * 0.45, y + r * 0.05), (x + r * 0.45, y + r * 0.05),
                     max(2, int(r * 0.16)))                                  # visor
    poly(surf, (150, 95, 50), [(x - r * 0.35, y + r * 0.3), (x + r * 0.35, y + r * 0.3),
                               (x, y + r * 0.55)])                          # moustache


def _archers(surf, x, y, r, tc, facing, t):
    bx, by = rot(r * 0.85, 0, facing)
    pygame.draw.arc(surf, (130, 80, 40), (x + bx - r * 0.6, y + by - r * 0.6, r * 1.2, r * 1.2),
                    -facing - 1.3, -facing + 1.3, max(2, int(r * 0.18)))
    _team_base(surf, x, y, r, tc)
    circle(surf, (235, 95, 165), (x, y - r * 0.05), r * 0.78)               # pink hair
    circle(surf, SKIN, (x, y + r * 0.12), r * 0.48)                        # face
    circle(surf, (40, 30, 30), (x - r * 0.17, y + r * 0.08), r * 0.08)
    circle(surf, (40, 30, 30), (x + r * 0.17, y + r * 0.08), r * 0.08)
    circle(surf, (235, 95, 165), (x + r * 0.55, y - r * 0.55), r * 0.28)    # ponytail


def _giant(surf, x, y, r, tc, facing, t):
    for side in (-1, 1):
        fx, fy = rot(r * 0.35, side * r * 0.85, facing)
        circle(surf, shade(SKIN, 0.8), (x + fx, y + fy), r * 0.3)
    _team_base(surf, x, y, r, tc)
    circle(surf, (225, 170, 120), (x, y), r * 0.78)                         # bald head
    circle(surf, (240, 190, 140), (x - r * 0.2, y - r * 0.25), r * 0.3)
    pygame.draw.arc(surf, (90, 50, 25), (x - r * 0.4, y - r * 0.05, r * 0.8, r * 0.5), 3.4, 6.0,
                    max(2, int(r * 0.12)))                                   # frown
    for side in (-1, 1):
        circle(surf, (40, 30, 30), (x + side * r * 0.22, y - r * 0.1), r * 0.08)
    poly(surf, (230, 120, 40), [(x - r * 0.3, y - r * 0.7), (x + r * 0.3, y - r * 0.7),
                                (x, y - r * 0.95)])                          # tuft


def _goblins(surf, x, y, r, tc, facing, t):
    _weapon(surf, x, y, r, facing + 0.4, 0.6, (200, 200, 210), max(2, int(r * 0.2)))
    for side in (-1, 1):
        poly(surf, (90, 180, 60), [(x + side * r * 0.45, y - r * 0.2), (x + side * r * 1.2, y - r * 0.6),
                                   (x + side * r * 0.55, y + r * 0.2)])     # ears
    _team_base(surf, x, y, r, tc)
    circle(surf, (110, 205, 80), (x, y), r * 0.75)
    for side in (-1, 1):
        circle(surf, WHITE, (x + side * r * 0.25, y - r * 0.05), r * 0.2)
        circle(surf, BLACK, (x + side * r * 0.25, y - r * 0.02), r * 0.09)
    pygame.draw.line(surf, (40, 60, 30), (x - r * 0.25, y + r * 0.35), (x + r * 0.25, y + r * 0.35), 2)


def _musketeer(surf, x, y, r, tc, facing, t):
    _weapon(surf, x, y, r, facing, 1.25, (70, 50, 40), max(2, int(r * 0.2)), offset=0.2)
    _team_base(surf, x, y, r, tc)
    circle(surf, SKIN, (x, y + r * 0.1), r * 0.5)
    circle(surf, (110, 55, 160), (x, y - r * 0.15), r * 0.72)               # hat brim
    circle(surf, (140, 75, 195), (x, y - r * 0.2), r * 0.45)
    poly(surf, (255, 225, 120), [(x + r * 0.2, y - r * 0.4), (x + r * 0.9, y - r * 0.9),
                                 (x + r * 0.45, y - r * 0.2)])               # plume
    circle(surf, SKIN, (x, y + r * 0.3), r * 0.32)


def _minipekka(surf, x, y, r, tc, facing, t):
    _weapon(surf, x, y, r, facing - 0.3, 1.1, (200, 220, 255), max(3, int(r * 0.3)))
    _team_base(surf, x, y, r, tc)
    circle(surf, (45, 65, 120), (x, y), r * 0.78)
    for side in (-1, 1):
        poly(surf, (210, 215, 225), [(x + side * r * 0.35, y - r * 0.45), (x + side * r * 0.95, y - r * 1.0),
                                     (x + side * r * 0.6, y - r * 0.25)])   # horns
    pygame.draw.line(surf, (120, 240, 255), (x - r * 0.35, y), (x + r * 0.35, y), max(2, int(r * 0.18)))


def _minions(surf, x, y, r, tc, facing, t):
    flap = math.sin(t * 18) * 0.35
    for side in (-1, 1):
        poly(surf, (60, 80, 180), [(x + side * r * 0.4, y), (x + side * r * 1.5, y - r * (0.6 + flap)),
                                   (x + side * r * 1.2, y + r * 0.3)])
    _team_base(surf, x, y, r, tc)
    circle(surf, (90, 125, 240), (x, y), r * 0.75)
    for side in (-1, 1):
        poly(surf, (240, 240, 240), [(x + side * r * 0.2, y - r * 0.55), (x + side * r * 0.45, y - r * 1.0),
                                     (x + side * r * 0.5, y - r * 0.45)])   # horns
        circle(surf, (255, 240, 120), (x + side * r * 0.22, y - r * 0.1), r * 0.14)
    pygame.draw.line(surf, WHITE, (x - r * 0.25, y + r * 0.3), (x + r * 0.25, y + r * 0.3), 2)


def _fireball(surf, x, y, r, tc, facing, t):
    for k, col in enumerate([(200, 60, 20), (250, 130, 30), (255, 210, 80)]):
        circle(surf, col, (x, y), r * (1 - k * 0.28))
    circle(surf, (255, 250, 200), (x - r * 0.25, y - r * 0.25), r * 0.2)


_UNITS = {"Knight": _knight, "Archers": _archers, "Giant": _giant, "Goblins": _goblins,
          "Musketeer": _musketeer, "MiniPekka": _minipekka, "Minions": _minions,
          "Fireball": _fireball}


# ------------------------------------------------------------------ towers
def draw_tower(surf, center, half, team, is_king, active, t=0.0, flash=0.0):
    x, y = center
    h = half
    tc, td = TEAM[team], TEAM_DARK[team]
    ellipse_alpha(surf, BLACK, (int(x - h * 1.15), int(y + h * 0.55), int(h * 2.3), int(h * 0.9)), 70)
    # stone body
    body = pygame.Rect(0, 0, int(h * 2), int(h * 2))
    body.center = (int(x), int(y))
    pygame.draw.rect(surf, STONE_DARK, body.move(0, 4), border_radius=int(h * 0.35))
    pygame.draw.rect(surf, STONE, body, border_radius=int(h * 0.35))
    # brick lines
    for k in range(1, 4):
        yy = body.top + k * body.height // 4
        pygame.draw.line(surf, STONE_DARK, (body.left + 4, yy), (body.right - 4, yy), 1)
    # crenellations
    n = 4 if is_king else 3
    cw = body.width / (2 * n - 1)
    for side in (body.top - cw * 0.45, body.bottom - cw * 0.55):
        for k in range(n):
            pygame.draw.rect(surf, STONE_LIGHT, (body.left + k * 2 * cw, side, cw, cw), border_radius=3)
    # team platform
    circle(surf, td, (x, y + 2), h * 0.72)
    circle(surf, tc, (x, y), h * 0.72)
    circle(surf, shade(tc, 1.25), (x - h * 0.2, y - h * 0.22), h * 0.22)
    if is_king:
        circle(surf, SKIN, (x, y + h * 0.1), h * 0.38)                     # king's face
        circle(surf, (240, 240, 240), (x, y + h * 0.38), h * 0.28)         # beard
        if active:
            for side in (-1, 1):
                circle(surf, BLACK, (x + side * h * 0.14, y + h * 0.02), max(2, h * 0.06))
        else:
            for side in (-1, 1):
                pygame.draw.line(surf, BLACK, (x + side * h * 0.2, y + h * 0.03),
                                 (x + side * h * 0.08, y + h * 0.03), 2)
            zz = int(t * 1.5) % 3
            text(surf, "z" * (zz + 1), (x + h * 0.55, y - h * 0.9), int(h * 0.45), anchor="midleft")
        crown(surf, (x, y - h * 0.38), h * 0.7)
    else:
        circle(surf, (235, 95, 165), (x, y), h * 0.34)                      # princess hair
        circle(surf, SKIN, (x, y + h * 0.06), h * 0.24)
        crown(surf, (x, y - h * 0.32), h * 0.36, outline=False)
    if flash > 0:
        s = pygame.Surface(body.size, pygame.SRCALPHA)
        s.fill((255, 255, 255, int(110 * flash)))
        surf.blit(s, body.topleft)


def draw_rubble(surf, center, half, team):
    x, y = center
    rng = [(0.1, -0.3, 0.45), (-0.45, 0.2, 0.4), (0.45, 0.35, 0.38), (-0.1, 0.45, 0.3), (0.5, -0.4, 0.25)]
    ellipse_alpha(surf, BLACK, (int(x - half), int(y - half * 0.6), int(half * 2), int(half * 1.6)), 60)
    for dx, dy, rr in rng:
        circle(surf, STONE_DARK, (x + dx * half, y + dy * half + 2), rr * half)
        circle(surf, STONE, (x + dx * half, y + dy * half), rr * half)
    circle(surf, TEAM_DARK[team], (x, y), half * 0.2)


def hp_bar(surf, center_x, top, width, frac, team, hp=None, show_number=False):
    w, h = int(width), (14 if show_number else 7)
    x = int(center_x - w / 2)
    pygame.draw.rect(surf, (20, 20, 30), (x - 1, top - 1, w + 2, h + 2), border_radius=3)
    pygame.draw.rect(surf, (60, 60, 70), (x, top, w, h), border_radius=3)
    if frac > 0:
        pygame.draw.rect(surf, TEAM[team], (x, top, max(2, int(w * frac)), h), border_radius=3)
        pygame.draw.rect(surf, shade(TEAM[team], 1.3), (x + 1, top + 1, max(1, int(w * frac) - 2), 2),
                         border_radius=2)
    if show_number and hp is not None:
        text(surf, int(math.ceil(hp)), (center_x, top + h // 2), 11, outline=1, anchor="center")


# ------------------------------------------------------------------ cards
CARD_BG = {"Knight": (90, 120, 170), "Archers": (110, 150, 90), "Giant": (170, 120, 70),
           "Goblins": (80, 140, 70), "Musketeer": (120, 80, 160), "MiniPekka": (60, 80, 140),
           "Fireball": (180, 80, 40), "Minions": (70, 90, 170)}


def draw_card(surf, rect, name, cost, affordable=True, selected=False, t=0.0, small=False,
              elixir_frac=1.0):
    rect = pygame.Rect(rect)
    if selected:
        rect = rect.move(0, -10)
        glow = rect.inflate(8, 8)
        pygame.draw.rect(surf, (255, 255, 180), glow, border_radius=10)
    pygame.draw.rect(surf, (20, 20, 30), rect.move(0, 3), border_radius=8)
    pygame.draw.rect(surf, (230, 200, 120), rect, border_radius=8)           # gold frame
    inner = rect.inflate(-6, -6)
    bg = CARD_BG.get(name, (100, 100, 100))
    pygame.draw.rect(surf, bg, inner, border_radius=6)
    pygame.draw.rect(surf, shade(bg, 1.25), (inner.x, inner.y, inner.w, inner.h // 2), border_radius=6)
    r = inner.w * (0.28 if not small else 0.3)
    if name == "Goblins":
        for dx in (-0.6, 0.6, 0):
            draw_unit(surf, name, (inner.centerx + dx * r, inner.centery - r * 0.1 + (0.4 * r if dx == 0 else 0)),
                      r * 0.6, 0, -math.pi / 2, t, shadow=False)
    elif name == "Archers":
        for dx in (-0.5, 0.5):
            draw_unit(surf, name, (inner.centerx + dx * r, inner.centery), r * 0.72, 0, -math.pi / 2, t,
                      shadow=False)
    elif name == "Minions":
        for dx, dy in ((-0.6, 0.2), (0.6, 0.2), (0, -0.4)):
            draw_unit(surf, name, (inner.centerx + dx * r, inner.centery + dy * r), r * 0.55, 0,
                      -math.pi / 2, t, shadow=False)
    else:
        draw_unit(surf, name, (inner.centerx, inner.centery - 2), r, 0, -math.pi / 2, t, shadow=False)
    if not small:
        text(surf, name if name != "MiniPekka" else "Mini PEKKA", (inner.centerx, inner.bottom - 2),
             11, anchor="midbottom")
    if not affordable:
        # grey cover that drains from the top as elixir fills toward the card's cost
        cover = int(rect.h * (1 - elixir_frac))
        s = pygame.Surface((rect.w, max(1, cover)), pygame.SRCALPHA)
        s.fill((40, 40, 50, 170))
        surf.blit(s, rect.topleft)
    elixir_drop(surf, (rect.left + 9, rect.top + 11), 8 if not small else 6, cost, 13 if not small else 10)


def draw_elixir_bar(surf, rect, elixir, max_elixir=10):
    rect = pygame.Rect(rect)
    pygame.draw.rect(surf, (20, 15, 30), rect.inflate(4, 4), border_radius=8)
    pygame.draw.rect(surf, (55, 30, 70), rect, border_radius=6)
    seg = rect.w / max_elixir
    full = int(elixir)
    max_elixir = int(max_elixir)
    for k in range(max_elixir):
        r = pygame.Rect(rect.x + k * seg + 1, rect.y + 2, seg - 2, rect.h - 4)
        if k < full:
            pygame.draw.rect(surf, ELIXIR, r, border_radius=4)
            pygame.draw.rect(surf, (240, 140, 245), (r.x + 2, r.y + 1, r.w - 4, 3), border_radius=2)
        elif k == full:
            frac = elixir - full
            pygame.draw.rect(surf, ELIXIR_DARK, (r.x, r.y, int(r.w * frac), r.h), border_radius=4)
    elixir_drop(surf, (rect.left - 2, rect.centery - 1), 13, int(elixir), 17)
