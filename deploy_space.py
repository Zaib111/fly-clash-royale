"""Publish the browser version to a Hugging Face Space (static site).

    hf auth login                                   # once, with a write token
    python export_web.py runs/fly1/latest.pt        # the fly to publish
    python deploy_space.py                          # -> https://huggingface.co/spaces/<you>/fly-clash-royale

Uploads web/ (viewer, JS engine, three.js, exported model) plus a Space card.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

CARD = """---
title: FlyRoyale
emoji: 🪰
colorFrom: gray
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-4.0
short_description: A real fruit-fly brain plays Clash Royale
---

# FlyRoyale

A simulation of an adult fruit fly's brain, wired exactly like the real one (134,181 neurons and
2.7M connections from the [FlyWire](https://flywire.ai) connectome), learned to play a Clash Royale-style
game purely by playing against itself (self-play reinforcement learning). Which neurons connect, and
whether each connection excites or inhibits, comes from the real fly and never changes; training only
tuned how strongly each neuron responds.

Everything runs in your browser: the game, the bots and the fly's brain ({iteration}).

- Code (MIT): https://github.com/Zaib111/fly-clash-royale
- Connectome data: FlyWire, CC BY-NC 4.0 (Dorkenwald et al. 2024, Schlegel et al. 2024, Shiu et al. 2024).
  The fly model here is derived from it, so this Space is non-commercial.
- Unofficial fan project, not affiliated with Supercell. No Supercell assets are used.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="fly-clash-royale", help="Space name (under your account)")
    args = ap.parse_args()
    if not (WEB / "model" / "meta.json").exists():
        raise SystemExit("no exported fly: run `python export_web.py runs/<run>/latest.pt` first")
    api = HfApi()
    user = api.whoami()["name"]
    repo = f"{user}/{args.space}"
    meta = json.loads((WEB / "model" / "meta.json").read_text())
    it = meta.get("source", {}).get("iteration")
    card = CARD.replace("{iteration}", f"trained for {it + 1} self-play iterations" if it is not None else "trained by self-play")
    api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(WEB), repo_id=repo, repo_type="space",
                      ignore_patterns=["dashboard.html", "**/.DS_Store"],
                      commit_message="Update the fly and the viewer")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="space",
                    commit_message="Update Space card")
    print(f"published: https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    main()
