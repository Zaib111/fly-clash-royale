"""Pack a trained fly into files the browser version can load (web/model/).

    python export_web.py runs/fly1/latest.pt

Writes web/model/meta.json and web/model/fly.bin.gz: the frozen wiring with the trained
gains folded into each connection's weight (float16), the input/output layers, the value
network, and the points for the brain view. Derived from FlyWire data: CC BY-NC 4.0.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from fly import connectome
from fly.critic import Critic
from fly.policies import load_checkpoint
from fly.viz import GROUPS, brain_points

OUT = Path(__file__).resolve().parent / "web" / "model"


class Packer:
    def __init__(self):
        self.parts, self.meta, self.offset = [], {}, 0

    def add(self, name, arr, dtype):
        a = np.ascontiguousarray(np.asarray(arr).astype(dtype))
        pad = (-self.offset) % 8
        if pad:
            self.parts.append(b"\0" * pad)
            self.offset += pad
        self.meta[name] = {"dtype": np.dtype(dtype).name, "shape": list(a.shape), "offset": self.offset,
                           "length": int(a.size)}
        self.parts.append(a.tobytes())
        self.offset += a.nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    policy, ck = load_checkpoint(args.checkpoint)
    cfg = ck["cfg"]
    if cfg["policy"] != "fly":
        raise SystemExit("only fly checkpoints can be exported")
    if "critic" not in ck:
        raise SystemExit("checkpoint has no value network; use runs/<name>/latest.pt")
    data = connectome.load(cfg.get("subset", "full"), cfg.get("threshold", 5))
    sd = {k: v.detach().float() for k, v in policy.state_dict().items()}
    n = data["n"]
    pre, post = data["pre"].numpy(), data["post"].numpy()

    # fold every trained gain into the per-connection weight: W[post, pre] * g_post * g_pre
    w = policy.w0.float().numpy().copy()
    scale = float(sd["log_scale"].exp())
    if policy.gains == "neuron":
        w *= sd["log_gain_post"].exp().numpy()[post] * scale * sd["log_gain_pre"].exp().numpy()[pre]
    else:
        w *= sd["log_gain"].exp().numpy() * scale

    crow = data["crow"].numpy()
    col_delta = pre.astype(np.int64).copy()          # delta-encode columns within each row (compresses well)
    starts = crow[:-1][np.diff(crow) > 0]
    d = np.diff(col_delta, prepend=0)
    d[starts] = col_delta[starts]
    critic = Critic()
    critic.load_state_dict(ck["critic"])
    csd = {k: v.detach().float().numpy() for k, v in critic.state_dict().items()}
    idx, pos, group = brain_points(data)

    p = Packer()
    p.add("crow", crow, np.int32)
    p.add("col_delta", d, np.uint32)
    p.add("w", w, np.float16)
    p.add("bias", sd["bias"], np.float32)
    p.add("eye_idx", data["eye_idx"], np.uint32)
    p.add("eye_cell", data["eye_cell"], np.uint16)
    p.add("eye_w", sd["eye_w"], np.float32)
    p.add("eye_b", sd["eye_b"], np.float32)
    p.add("sense_idx", data["sense_idx"], np.uint32)
    p.add("sense_w", sd["sense.weight"], np.float32)
    p.add("sense_b", sd["sense.bias"], np.float32)
    p.add("read_idx", data["read_idx"], np.uint32)
    p.add("dec1_w", sd["decoder.0.weight"], np.float32)
    p.add("dec1_b", sd["decoder.0.bias"], np.float32)
    p.add("dec2_w", sd["decoder.2.weight"], np.float32)
    p.add("dec2_b", sd["decoder.2.bias"], np.float32)
    for k, v in csd.items():
        p.add("critic." + k, v, np.float32)
    p.add("viz_idx", idx, np.uint32)
    p.add("viz_pos", pos, np.float32)
    p.add("viz_group", group, np.uint8)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = b"".join(p.parts)
    (out / "fly.bin.gz").write_bytes(gzip.compress(raw, compresslevel=9))
    meta = {
        "n": n, "steps": policy.steps, "leak": policy.leak, "edges": int(len(w)),
        "synapses": int(data["syn"].sum()), "groups": GROUPS,
        "source": {"checkpoint": str(args.checkpoint), "iteration": ck.get("iter")},
        "license": "Derived from FlyWire (CC BY-NC 4.0): Dorkenwald et al. 2024, Schlegel et al. 2024, Shiu et al. 2024",
        "arrays": p.meta,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}/fly.bin.gz: {len(raw) / 1e6:.1f} MB raw -> {(out / 'fly.bin.gz').stat().st_size / 1e6:.1f} MB gzipped")


if __name__ == "__main__":
    main()
