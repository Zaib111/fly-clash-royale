# Fly Clash Royale

A real fruit-fly brain learns to play a simplified Clash Royale, just from winning and losing against itself.

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

Progress prints every iteration and is written to `runs/<name>/log.csv`. The columns are entropy, play rate, Elo vs past selves, and win rate against the random and scripted bots every 20 iterations. Checkpoints are `runs/<name>/latest.pt` plus `iterNNNNN.pt` every 50 iterations.

## Watch / play

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

## License / data

FlyWire connectome data: CC BY-NC 4.0, from Dorkenwald et al. 2024, Schlegel et al. 2024 and Shiu et al. 2024 (*Nature*). Anything derived from it, including trained fly checkpoints, is **non-commercial only**.
