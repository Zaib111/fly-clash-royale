# Fly Clash Royale

A real fruit-fly brain learns to play Clash Royale, just from winning and losing against itself.

The "brain" is the [FlyWire](https://flywire.ai) adult *Drosophila* connectome: about 134k neurons and 2.7M connections, keeping connections with at least 5 synapses. The wiring and the sign of each connection (excitatory or inhibitory) come from the real fly and never change. The model is set up like the fruit-fly chess projects ([ChessFly](https://huggingface.co/mlabonne/chessfly), [fly-chess](https://huggingface.co/cesp99/fly-chess)), but it is trained by self-play reinforcement learning, not by copying a chess engine.

```
arena (9x16 grid) ─► ~7.5k photoreceptors ─┐
elixir/hand/clock ─► ~5k other sensory    ─┴─► real fly wiring, 4 steps / 0.5s ─► 1.4k descending + motor neurons ─► wait | play card X at cell Y
```

- **Neurons:** a simple rate model, `h ← (1-a)·h + a·tanh(relu(g_post·W(g_pre·h) + u + b))`. The state `h` carries over between decisions, so the fly has short-term memory.
- **What's trained:** per-neuron gains (or per-synapse gains with `--gains synapse`), per-neuron biases, the input encoder and the output decoder. The connections themselves and their signs stay fixed.
- **How it learns:** PPO self-play. Opponents are the current fly or frozen past snapshots, chosen more often when the fly tends to lose to them. A win scores +1 and a loss −1, plus a small tower-damage bonus that fades out early in training.
- **Bots:** the random and scripted bots (`game/bots.py`) are only used to *measure* progress, never as training targets.

## Setup

```bash
python3 -m venv .venv --system-site-packages && .venv/bin/pip install -r requirements.txt
.venv/bin/python data/fetch_connectome.py          # ~135 MB from GitHub
.venv/bin/python -m fly.connectome                 # full brain   (134k neurons)
.venv/bin/python -m fly.connectome --subset central  # optional: no optic lobes (49k neurons, ~2x faster)
```

## Train

Training runs on the CPU, since PyTorch has no sparse support on Apple GPUs. It will make your fans work. Use `--threads` to leave cores free.

```bash
# the fly (overnight). Resume later with the same --run and --resume
.venv/bin/python -m train.selfplay_ppo --policy fly --run fly1 --minutes 600 --threads 6

# baseline: ordinary MLP with the same inputs/outputs (fast, ~30 min)
.venv/bin/python -m train.selfplay_ppo --policy mlp --run mlp1 --minutes 30
```

**Training dashboard.** Live charts of win rate, Elo, play rate and match length, read from the log. It's safe to keep open while training runs:

```bash
.venv/bin/python dashboard.py            # http://127.0.0.1:8766
```

Progress also prints every iteration and is written to `runs/<name>/log.csv`. The columns are entropy, play rate, Elo vs past selves, and win rate against the random and scripted bots every 20 iterations. Checkpoints are `runs/<name>/latest.pt` plus `iterNNNNN.pt` every 50 iterations.

## Watch / play

**3D viewer (recommended).** It opens in your browser and the match runs here in Python. The page draws the arena in 3D, shows the fly hovering over its King tower, and displays its brain as a live point cloud of real neuron positions:

```bash
.venv/bin/python play3d.py --p1 runs/fly1/latest.pt --p2 human     # you vs the fly
.venv/bin/python play3d.py --p1 scripted --p2 runs/fly1/latest.pt  # watch the fly play the bot
.venv/bin/python play3d.py --p1 human --p2 scripted                # no fly needed
```

The side panel shows the fly's brain lighting up as a wave, from the eyes through the central brain to the motor neurons. It also shows what the fly sees (the 9×16 grid on its photoreceptors, with a glow where it wants to play) and its hand, with the odds of playing each card.

Controls: click a card (or press 1–4), then click your half of the arena. Drag to orbit, scroll to zoom, double-click to reset the view. Space pauses, +/- changes speed and R starts a rematch. three.js r160 is vendored in `web/vendor/` (MIT), so no internet connection is needed.

**2D viewer (pygame).**

```bash
.venv/bin/python play.py                                         # scripted vs random
.venv/bin/python play.py --p1 runs/fly1/latest.pt --p2 scripted  # fly vs bot, with live neuron panel
.venv/bin/python play.py --p1 runs/fly1/latest.pt --p2 human     # you vs the fly
```

Controls: click a card (or press 1–4), then click the arena. Space pauses, +/- changes speed, R starts a rematch and Esc quits.

## Tests

```bash
.venv/bin/python -m pytest -q tests
```

## License

- **Code:** MIT, see [LICENSE](LICENSE). The bundled three.js (`web/vendor/`) is also MIT.
- **FlyWire connectome data:** CC BY-NC 4.0, from Dorkenwald et al. 2024, Schlegel et al. 2024 and Shiu et al. 2024 (*Nature*). The data isn't included in this repo; `data/fetch_connectome.py` downloads it. Anything derived from it, including trained fly checkpoints, is **non-commercial only**.

This is an unofficial fan project and is not affiliated with, endorsed by, or connected to Supercell. "Clash Royale" is a trademark of Supercell. No Supercell art or assets are used: every visual is drawn from scratch in code.
