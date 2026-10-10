"""3D viewer: the match runs here in Python and streams to a three.js page in your browser.

    python play3d.py                                       # scripted bot vs random bot
    python play3d.py --p1 human --p2 scripted              # you vs the scripted bot
    python play3d.py --p1 runs/fly1/latest.pt --p2 human   # you vs the fly
    python play3d.py --p1 runs/fly1/latest.pt --p2 scripted

Players: human | random | scripted | <checkpoint.pt>. Opens http://127.0.0.1:8765.
The game simulation and the fly brain run in this process; the browser only draws,
interpolating between game ticks so motion stays smooth while the fly is thinking.
"""
import argparse
import base64
import json
import mimetypes
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from game.cards import CARDS
from game.env import DECISION_TICKS, GW, G, NUM_TROOP_TYPES, apply_action, decode_action
from game.players import Human, PolicyAgent, make_player
from game.sim import BRIDGES, DT, HEIGHT, RIVER_HI, RIVER_LO, WIDTH, Match

WEB = Path(__file__).resolve().parent / "web"
BRAIN_POINTS = 60_000
mimetypes.add_type("text/javascript", ".js")


def _num(v):
    return round(float(v), 3)


class Brain:
    """Which neurons to draw in the brain panel, their 3D positions and groups."""

    GROUPS = ["central brain", "eyes & optic lobes", "other senses", "descending & motor"]

    def __init__(self, agent: PolicyAgent, seed=0):
        from fly import connectome
        data = connectome.load(agent.cfg.get("subset", "full"), agent.cfg.get("threshold", 5))
        if "pos" not in data:
            raise SystemExit("connectome cache has no neuron positions: rerun `python -m fly.connectome`")
        n = data["n"]
        sc = np.array(data["super_class"])
        group = np.zeros(n, dtype=np.uint8)
        group[np.isin(sc, ["optic", "visual_projection", "visual_centrifugal"])] = 1
        group[data["eye_idx"].numpy()] = 1
        group[data["sense_idx"].numpy()] = 2
        group[data["read_idx"].numpy()] = 3
        must = np.unique(np.concatenate([data["read_idx"].numpy(), data["sense_idx"].numpy()]))
        rest = np.setdiff1d(np.arange(n), must)
        rng = np.random.default_rng(seed)
        k = max(0, min(len(rest), BRAIN_POINTS - len(must)))
        self.idx = np.sort(np.concatenate([must, rng.choice(rest, k, replace=False)]))
        pos = data["pos"].numpy()[self.idx].astype(np.float64) * np.array([4.0, 4.0, 40.0])  # voxels -> nm
        pos -= pos.mean(0)
        pos /= np.abs(pos).max()
        # three.js axes: x = lateral, y = up (FlyWire y points ventral), z = toward the viewer
        self.pos = np.stack([pos[:, 0], -pos[:, 1], -pos[:, 2]], 1).astype(np.float32)
        self.group = group[self.idx]
        self.read_idx = data["read_idx"].numpy()
        self.n = n
        self.n_edges = int(len(data["syn"]))
        self.n_syn = int(data["syn"].sum())

    def frame(self, agent: PolicyAgent, m, seat):
        h = agent.state[0].numpy()
        shown = np.clip(h[self.idx] / 0.4, 0, 1)
        out = {
            "act": base64.b64encode((shown * 255).astype(np.uint8).tobytes()).decode(),
            "active": int((h > 0.05).sum()),
            "active_out": int((h[self.read_idx] > 0.05).sum()),
            "mean": _num(h.mean()),
        }
        # what it saw (its own side at the bottom): its troops, enemy troops, its towers,
        # enemy towers, spells - one byte per cell each
        sp = agent.obs
        nt = NUM_TROOP_TYPES
        planes = np.stack([sp[:nt].sum(0) / 1.5, sp[nt:2 * nt].sum(0) / 1.5,
                           np.clip(sp[2 * nt], 0, 1), np.clip(sp[2 * nt + 1], 0, 1),
                           np.clip(sp[2 * nt + 2] + sp[2 * nt + 3], 0, 1)])
        out["vision"] = base64.b64encode((np.clip(planes, 0, 1) * 255).astype(np.uint8).tobytes()).decode()
        # where it wanted to play (summed over cards) and how much it liked each card
        probs = agent.probs
        cards = probs[1:].reshape(4, G)
        out["intent"] = [round(float(v), 4) for v in cards.sum(0)]
        out["slots"] = [round(float(v), 4) for v in cards.sum(1)]
        out["wait"] = round(float(probs[0]), 4)
        d = decode_action(agent.last_action)
        out["chosen"] = None if d is None else [d[0], d[2] * GW + d[1]]
        out["hand"] = list(m.hand(seat))
        out["elixir"] = _num(m.elixir[seat])
        return out


class Game:
    def __init__(self, specs, speed, threads):
        self.players = [make_player(specs[0], 1, threads), make_player(specs[1], 2, threads)]
        self.human = next((p for p, pl in enumerate(self.players) if isinstance(pl, Human)), None)
        self.view = 1 if self.human == 1 else 0
        self.speed = speed
        self.paused = False
        self.lock = threading.RLock()
        self.clients: list[queue.Queue] = []
        self.brains = {}
        for p, pl in enumerate(self.players):
            if isinstance(pl, PolicyAgent) and pl.is_fly:
                self.brains[p] = Brain(pl)
        self.match_id = 0
        self.decisions = 0
        self.new_match()

    # ------------------------------------------------------------------ state
    def new_match(self):
        with self.lock:
            self.match_id += 1
            self.m = Match(seed=int(time.time()) % 100_000 + self.match_id)
            self.m.events = []
            for pl in self.players:
                pl.reset()
            self.decisions = 0
            self.thinking = {}

    def init_info(self):
        info = {
            "arena": {"w": WIDTH, "h": HEIGHT, "river": [RIVER_LO, RIVER_HI], "bridges": list(BRIDGES),
                      "troop_limit": 14.0, "dt": DT},
            "cards": [{"name": c.name, "cost": c.cost, "kind": c.kind, "radius": c.radius,
                       "is_air": c.is_air, "count": c.count, "spell_radius": c.spell_radius,
                       "range": c.range} for c in CARDS],
            "players": [{"name": pl.name, "kind": getattr(pl, "kind", "bot")} for pl in self.players],
            "view": self.view,
            "human": self.human,
            "brains": {str(p): {"points": len(b.idx), "neurons": b.n, "connections": b.n_edges,
                                "synapses": b.n_syn, "groups": Brain.GROUPS}
                       for p, b in self.brains.items()},
        }
        return info

    def snapshot(self, events, brain_frames):
        m = self.m
        units = []
        for u in m.units:
            t = u.target if (u.target is not None and u.target.alive) else None
            units.append([u.uid, u.card, u.owner, _num(u.x), _num(u.y), round(u.hp), round(u.max_hp),
                          _num(max(0.0, u.deploy)),
                          None if t is None else _num(t.x), None if t is None else _num(t.y)])
        towers = [[t.owner, int(t.is_king), t.x, t.y, round(t.hp), round(t.max_hp), int(t.active),
                   int(t.alive)] for p in (0, 1) for t in m.towers[p]]
        ev = [[_num(e[0]), e[1]] + [(_num(v) if isinstance(v, float) else v) for v in e[2:]]
              for e in events]
        return {
            "match": self.match_id, "t": _num(m.time), "tick": m.ticks, "speed": self.speed,
            "paused": self.paused, "units": units, "towers": towers,
            "elixir": [_num(e) for e in m.elixir], "hand": [m.hand(0), m.hand(1)],
            "next": [m.next_card(0), m.next_card(1)], "crowns": [m.crowns(0), m.crowns(1)],
            "time_left": _num(m.time_left()), "double": m.double_elixir(), "done": m.done,
            "winner": m.winner, "events": ev, "brains": brain_frames,
            "thinking": self.thinking, "decisions": self.decisions,
        }

    def broadcast(self, msg):
        data = json.dumps(msg, separators=(",", ":"))
        for q in list(self.clients):
            try:
                q.put_nowait(data)
            except queue.Full:
                pass

    # ------------------------------------------------------------------ loop
    def tick(self):
        m = self.m
        brain_frames = {}
        if m.ticks % DECISION_TICKS == 0:
            self.decisions += 1
            actions = []
            for p, pl in enumerate(self.players):
                a = pl.act(m, p)
                if isinstance(pl, PolicyAgent):
                    self.thinking[str(p)] = [[desc, round(pr, 3)] for desc, pr in pl.top]
                    if p in self.brains:
                        brain_frames[str(p)] = self.brains[p].frame(pl, m, p)
                actions.append(a)
            for p, a in enumerate(actions):
                apply_action(m, p, a)
        m.step()
        events = list(m.events)
        m.events.clear()
        return self.snapshot(events, brain_frames)

    def run(self):
        next_t = time.perf_counter()
        last_idle = 0.0
        while True:
            with self.lock:
                idle = self.paused or self.m.done
                if not idle:
                    snap = self.tick()
            if idle:
                now = time.perf_counter()
                if now - last_idle > 0.25:
                    with self.lock:
                        self.broadcast(self.snapshot([], {}))
                    last_idle = now
                time.sleep(0.02)
                next_t = time.perf_counter()
                continue
            self.broadcast(snap)
            next_t += DT / self.speed
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                next_t = time.perf_counter()   # fell far behind (e.g. slow brain): don't fast-forward

    # ------------------------------------------------------------------ commands
    def human_play(self, slot, x, y):
        if self.human is None:
            return False
        with self.lock:
            return self.m.play(self.human, int(slot), float(x), float(y))

    def control(self, cmd, value=None):
        with self.lock:
            if cmd == "pause":
                self.paused = not self.paused
            elif cmd == "speed":
                self.speed = float(min(16.0, max(0.25, value)))
            elif cmd == "restart":
                self.new_match()
                self.paused = False


def make_handler(game: Game):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/init":
                return self._json(game.init_info())
            if path.startswith("/api/brain/"):
                b = game.brains.get(int(path.rsplit("/", 1)[1]))
                if b is None:
                    return self._send(404, b"", "text/plain")
                return self._send(200, b.pos.tobytes() + b.group.tobytes(), "application/octet-stream")
            if path == "/api/stream":
                return self._stream()
            return self._static(path)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/play":
                return self._json({"ok": game.human_play(body["slot"], body["x"], body["y"])})
            if self.path == "/api/control":
                game.control(body.get("cmd"), body.get("value"))
                return self._json({"ok": True})
            self._send(404, b"", "text/plain")

        def _static(self, path):
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            f = (WEB / rel).resolve()
            if WEB not in f.parents or not f.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
            self._send(200, f.read_bytes(), ctype)

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = queue.Queue(maxsize=600)
            game.clients.append(q)
            try:
                while True:
                    try:
                        data = q.get(timeout=10)
                        self.wfile.write(f"data: {data}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                game.clients.remove(q)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", default="scripted", help="seat 0")
    ap.add_argument("--p2", default="random", help="seat 1")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=2, help="CPU cores for the fly brain")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    game = Game([args.p1, args.p2], args.speed, args.threads)
    threading.Thread(target=game.run, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(game))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}"
    print(f"Fly Clash Royale 3D: {url}   (Ctrl-C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
