"""The browser version (web/js/engine/game.js) must play exactly like the Python game."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from game.bots import RandomBot, ScriptedBot
from game.env import DECISION_TICKS, action_mask, apply_action, observe
from game.sim import Match

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def record(seed, tmp_path):
    m = Match(seed=seed)
    cycles = [list(c) for c in m.cycle]
    b0, b1 = ScriptedBot(seed), RandomBot(play_prob=0.4, seed=seed)
    actions, rec = [], {"checkpoints": [], "scripted": []}
    k = 0
    while not m.done:
        a0, a1 = b0.act(m, 0), b1.act(m, 1)
        rec["scripted"].append(a0)
        if k % 20 == 0:
            sp0, v0 = observe(m, 0)
            sp1, v1 = observe(m, 1)
            rec["checkpoints"].append({
                "tick": m.ticks, "elixir": list(m.elixir),
                "towers": [[t.hp, t.alive, t.active] for p in (0, 1) for t in m.towers[p]],
                "units": [[u.uid, u.card, u.owner, u.x, u.y, u.hp, u.deploy] for u in m.units],
                "obs": [sp0.ravel().tolist(), v0.tolist(), sp1.ravel().tolist(), v1.tolist()],
                "mask": [action_mask(m, 0).astype(int).tolist(), action_mask(m, 1).astype(int).tolist()],
            })
        actions.append([int(a0), int(a1)])
        apply_action(m, 0, a0)
        apply_action(m, 1, a1)
        m.step(DECISION_TICKS)
        k += 1
    rec["final"] = {"tick": m.ticks, "done": m.done, "winner": m.winner, "crowns": [m.crowns(0), m.crowns(1)]}
    path = tmp_path / f"scenario{seed}.json"
    path.write_text(json.dumps({"cycles": cycles, "actions": actions}))
    return path, rec


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_js_engine_matches_python(seed, tmp_path):
    path, py = record(seed, tmp_path)
    out = subprocess.run(["node", str(ROOT / "tests" / "js_parity.mjs"), str(path)],
                         capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    assert js["final"] == py["final"]
    assert js["scripted"] == py["scripted"]          # scripted bot decides identically
    assert len(js["checkpoints"]) == len(py["checkpoints"])
    for a, b in zip(js["checkpoints"], py["checkpoints"]):
        assert a["tick"] == b["tick"]
        np.testing.assert_allclose(a["elixir"], b["elixir"], atol=1e-9)
        assert [t[1:] for t in a["towers"]] == [t[1:] for t in b["towers"]]
        np.testing.assert_allclose([t[0] for t in a["towers"]], [t[0] for t in b["towers"]], atol=1e-6)
        assert [u[:3] for u in a["units"]] == [u[:3] for u in b["units"]]
        if a["units"]:
            np.testing.assert_allclose([u[3:] for u in a["units"]], [u[3:] for u in b["units"]], atol=1e-6)
        for x, y in zip(a["obs"], b["obs"]):
            np.testing.assert_allclose(x, y, atol=1e-5)
        assert a["mask"] == b["mask"]


MODEL = ROOT / "web" / "model"


@pytest.mark.skipif(not (MODEL / "meta.json").exists(), reason="no exported model (run export_web.py)")
def test_js_brain_matches_pytorch(tmp_path):
    import torch
    from fly.critic import Critic
    from fly.policies import load_checkpoint

    meta = json.loads((MODEL / "meta.json").read_text())
    policy, ck = load_checkpoint(meta["source"]["checkpoint"])
    critic = Critic()
    critic.load_state_dict(ck["critic"])
    m = Match(seed=5)
    b0, b1 = ScriptedBot(), RandomBot(play_prob=0.5, seed=5)
    for _ in range(40):                       # get some troops on the board
        apply_action(m, 0, b0.act(m, 0))
        apply_action(m, 1, b1.act(m, 1))
        m.step(DECISION_TICKS)
    sp, vec = observe(m, 1)
    mask = action_mask(m, 1)
    h0 = torch.rand(1, policy.n) * 0.3
    with torch.no_grad():
        logits, h1 = policy(torch.from_numpy(sp)[None], torch.from_numpy(vec)[None], torch.from_numpy(mask)[None], h0)
        probs = torch.softmax(logits[0], 0).numpy()
        v = float(critic(torch.from_numpy(sp)[None], torch.from_numpy(vec)[None]))
    sample = np.random.default_rng(0).choice(policy.n, 2000, replace=False)
    inp = tmp_path / "brain_in.json"
    inp.write_text(json.dumps({"sp": sp.ravel().tolist(), "vec": vec.tolist(), "mask": mask.astype(int).tolist(),
                               "h": h0[0].tolist(), "h_sample_idx": sample.tolist()}))
    out = json.loads(subprocess.run(["node", str(ROOT / "tests" / "js_brain.mjs"), str(MODEL), str(inp)],
                                    capture_output=True, text=True, check=True).stdout)
    print(f"JS brain forward: {out['ms']:.0f} ms")
    np.testing.assert_allclose(out["h_sample"], h1[0].numpy()[sample], atol=3e-3)   # weights are float16
    np.testing.assert_allclose(out["probs"], probs, atol=3e-3)
    assert int(np.argmax(out["probs"])) == int(np.argmax(probs))
    assert abs(out["win"] - min(1, max(0, (v + 1) / 2))) < 1e-4
