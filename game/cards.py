"""Card and tower stats, loosely based on tournament-standard Clash Royale numbers.

Distances are in tiles, times in seconds, speeds in tiles/second.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Card:
    name: str
    cost: int
    kind: str = "troop"          # "troop" or "spell"
    count: int = 1               # troops spawned per card
    hp: float = 0.0
    damage: float = 0.0
    hit_speed: float = 1.0
    range: float = 1.0           # attack range, edge to edge
    speed: float = 1.0
    radius: float = 0.5
    is_air: bool = False
    hits_air: bool = False
    buildings_only: bool = False
    sight: float = 5.5
    # spells
    spell_radius: float = 0.0
    tower_damage: float = 0.0
    spell_delay: float = 0.0


CARDS = [
    Card("Knight", 3, hp=1452, damage=167, hit_speed=1.2, range=0.7, speed=1.0, radius=0.5),
    Card("Archers", 3, count=2, hp=252, damage=89, hit_speed=0.9, range=5.0, speed=1.0,
         radius=0.4, hits_air=True),
    Card("Giant", 5, hp=3275, damage=211, hit_speed=1.5, range=0.7, speed=0.75, radius=0.75,
         buildings_only=True, sight=7.5),
    Card("Goblins", 2, count=3, hp=167, damage=99, hit_speed=1.1, range=0.5, speed=2.0,
         radius=0.35),
    Card("Musketeer", 4, hp=598, damage=181, hit_speed=1.0, range=6.0, speed=1.0, radius=0.5,
         hits_air=True, sight=6.0),
    Card("MiniPekka", 4, hp=1129, damage=598, hit_speed=1.6, range=0.7, speed=1.5, radius=0.5),
    Card("Fireball", 4, kind="spell", damage=572, tower_damage=172, spell_radius=2.5,
         spell_delay=0.6),
    Card("Minions", 3, count=3, hp=190, damage=84, hit_speed=1.0, range=1.6, speed=1.5,
         radius=0.4, is_air=True, hits_air=True),
]
CARD_INDEX = {c.name: i for i, c in enumerate(CARDS)}
NUM_CARDS = len(CARDS)
TROOP_IDS = [i for i, c in enumerate(CARDS) if c.kind == "troop"]


@dataclass(frozen=True)
class TowerStats:
    hp: float
    damage: float
    hit_speed: float
    range: float
    radius: float


PRINCESS = TowerStats(hp=3052, damage=109, hit_speed=0.8, range=7.5, radius=1.5)
KING = TowerStats(hp=4824, damage=109, hit_speed=1.0, range=7.0, radius=2.0)

DEPLOY_TIME = 1.0
START_ELIXIR = 5.0
MAX_ELIXIR = 10.0
ELIXIR_PER_SEC = 1 / 2.8
