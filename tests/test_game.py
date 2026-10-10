import numpy as np
import pytest

from game.bots import RandomBot, ScriptedBot
from game.cards import CARD_INDEX, CARDS, MAX_ELIXIR
from game.env import (DECISION_TICKS, G, GW, NUM_ACTIONS, TROOP_ROWS, VecEnv, action_mask,
                      apply_action, decode_action, encode_action, observe)
from game.sim import HEIGHT, TICKS_PER_SEC, Match, Unit


def play_out(seed, b0, b1):
    m = Match(seed=seed)
    while not m.done:
        apply_action(m, 0, b0.act(m, 0))
        apply_action(m, 1, b1.act(m, 1))
        m.step(DECISION_TICKS)
    return m


def test_deterministic_with_seed():
    a = play_out(7, ScriptedBot(1), RandomBot(seed=2))
    b = play_out(7, ScriptedBot(1), RandomBot(seed=2))
    assert (a.winner, a.ticks, a.tower_damage_dealt) == (b.winner, b.ticks, b.tower_damage_dealt)


def test_elixir_regen_and_cap():
    m = Match(seed=0)
    m.step(int(2.8 * TICKS_PER_SEC))
    assert m.elixir[0] == pytest.approx(6.0, abs=0.05)
    m.step(60 * TICKS_PER_SEC)
    assert m.elixir[0] == MAX_ELIXIR


def test_card_cycle():
    m = Match(seed=0)
    hand, nxt = m.hand(0), m.next_card(0)
    m.elixir[0] = 10
    played = hand[1]
    assert m.play(0, 1, 9.0, 5.0)
    assert m.hand(0)[1] == nxt
    assert m.cycle[0][-1] == played
    assert m.elixir[0] == 10 - CARDS[played].cost


def test_cannot_deploy_troops_on_enemy_side_or_without_elixir():
    m = Match(seed=0)
    m.cycle[0][0] = CARD_INDEX["Knight"]
    assert not m.play(0, 0, 9.0, 20.0)
    m.elixir[0] = 2
    assert not m.play(0, 0, 9.0, 5.0)


def test_giant_only_targets_buildings():
    m = Match(seed=0)
    giant = Unit(0, CARD_INDEX["Giant"], 3.5, 14.0)
    knight = Unit(1, CARD_INDEX["Knight"], 3.5, 13.0)
    giant.deploy = knight.deploy = 0
    m.units += [giant, knight]
    assert m._find_target(giant).is_building
    assert m._find_target(knight) is giant


def test_princess_damage_and_king_activation():
    m = Match(seed=0)
    king, princess = m.towers[1][0], m.towers[1][1]
    assert not king.active
    m._damage(0, princess, princess.hp + 1)
    assert not princess.alive and king.active
    assert m.crowns(0) == 1


def test_king_destroyed_ends_match():
    m = Match(seed=0)
    m._damage(0, m.towers[1][0], 1e9)
    m.step()
    assert m.done and m.winner == 0 and m.crowns(0) == 3


def test_scripted_beats_random():
    wins = sum(play_out(s, ScriptedBot(s), RandomBot(seed=s)).winner == 0 for s in range(6))
    assert wins >= 5


def test_action_roundtrip_and_mask():
    for a in [1, 5, G, G + 1, NUM_ACTIONS - 1]:
        assert encode_action(*decode_action(a)) == a
    m = Match(seed=0)
    m.elixir[0] = 10
    mask = action_mask(m, 0)
    assert mask[0]
    for s, cid in enumerate(m.hand(0)):
        block = mask[1 + s * G:1 + (s + 1) * G]
        if CARDS[cid].kind == "spell":
            assert block.all()
        else:
            assert block[:TROOP_ROWS * GW].all() and not block[TROOP_ROWS * GW:].any()
    m.elixir[0] = 0
    assert action_mask(m, 0).sum() == 1


def test_mirrored_observation_is_symmetric():
    """A position and its mirror image look identical to the opposing players."""
    m = Match(seed=0)
    m.cycle[1] = list(m.cycle[0])
    m.units.append(Unit(0, CARD_INDEX["Knight"], 4.0, 10.0))
    m.units.append(Unit(1, CARD_INDEX["Knight"], 4.0, HEIGHT - 10.0))
    sp0, v0 = observe(m, 0)
    sp1, v1 = observe(m, 1)
    np.testing.assert_allclose(sp0, sp1)
    np.testing.assert_allclose(v0, v1)


def test_mirrored_actions_land_symmetrically():
    m = Match(seed=0)
    m.cycle[1] = list(m.cycle[0])
    slot = next(s for s, c in enumerate(m.hand(0)) if CARDS[c].kind == "troop")
    a = encode_action(slot, 2, 3)
    assert apply_action(m, 0, a) and apply_action(m, 1, a)
    u0 = [u for u in m.units if u.owner == 0][0]
    u1 = [u for u in m.units if u.owner == 1][0]
    assert u0.x == pytest.approx(u1.x) and u0.y == pytest.approx(HEIGHT - u1.y)


def test_vecenv_zero_sum_and_autoreset():
    env = VecEnv(4, seed=0)
    for _ in range(400):
        rew, done, winner = env.step(np.zeros(4, int), np.zeros(4, int))
        np.testing.assert_allclose(rew[:, 0], -rew[:, 1])
        if done.any():
            assert all(not m.done for m in env.matches)
            return
    pytest.fail("no match finished")
