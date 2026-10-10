"""Build a compact graph of the FlyWire connectome for the fly brain model.

    python -m fly.connectome [--threshold 5] [--subset full|central]

Produces data/connectome_<subset>_t<threshold>.pt with:
  n, pre, post, syn, sign        edges sorted by (post, pre)  (CSR row = postsynaptic neuron)
  crow                           CSR row pointer for W[post, pre]
  t_perm, t_crow                 edge order + row pointer for W^T (row = presynaptic neuron)
  eye_idx, eye_cell              "photoreceptor" input neurons and the arena cell each one sees
  sense_idx                      other sensory neurons (receive the non-spatial observation vector)
  read_idx                       descending + motor neurons (the fly's "decision" output)
  super_class, root_id           per-neuron labels

FlyWire data is CC BY-NC 4.0 (non-commercial use only).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw"


def cache_path(subset="full", threshold=5):
    return DATA / f"connectome_{subset}_t{threshold}.pt"


def _retinotopic_cells(pos, gw, gh):
    """Map 3D neuron positions onto a gw x gh grid by PCA + rank normalisation, so the
    neurons' spatial layout becomes the arena layout and every cell gets neurons."""
    pos = pos - pos.mean(0)
    _, _, vt = np.linalg.svd(pos, full_matrices=False)
    proj = pos @ vt[:2].T
    ranks = proj.argsort(0).argsort(0) / len(proj)
    cx = np.minimum((ranks[:, 0] * gw).astype(np.int64), gw - 1)
    cy = np.minimum((ranks[:, 1] * gh).astype(np.int64), gh - 1)
    return cy * gw + cx


def build(threshold=5, subset="full", gw=9, gh=16):
    comp = pd.read_csv(RAW / "Completeness_783.csv", index_col=0)
    root_ids = comp.index.values
    ann = pd.read_csv(RAW / "neuron_annotations.tsv", sep="\t", low_memory=False,
                      usecols=["root_id", "super_class", "cell_class", "cell_type", "side",
                               "pos_x", "pos_y", "pos_z"])
    ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(root_ids)
    ann["super_class"] = ann["super_class"].fillna("unknown")

    con = pd.read_parquet(RAW / "Connectivity_783.parquet",
                          columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity",
                                   "Excitatory"])
    con = con[con.Connectivity >= threshold]

    is_photo = ann.cell_type.isin(["R1-6", "R7", "R8"]).values
    sc = ann.super_class.values
    keep = np.ones(len(ann), dtype=bool)
    if subset == "central":
        # drop the optic lobes; visual projection neurons become the "eyes"
        keep &= (sc != "optic") & ~is_photo
    elif subset != "full":
        raise ValueError(subset)

    # keep neurons that still have at least one edge inside the kept set
    pre0 = con.Presynaptic_Index.values
    post0 = con.Postsynaptic_Index.values
    ok = keep[pre0] & keep[post0]
    pre0, post0 = pre0[ok], post0[ok]
    syn0 = con.Connectivity.values[ok].astype(np.float32)
    sign0 = con.Excitatory.values[ok].astype(np.int8)
    used = np.zeros(len(ann), dtype=bool)
    used[pre0] = True
    used[post0] = True
    old_ids = np.flatnonzero(used)
    remap = -np.ones(len(ann), dtype=np.int64)
    remap[old_ids] = np.arange(len(old_ids))
    n = len(old_ids)
    pre, post = remap[pre0], remap[post0]

    order = np.lexsort((pre, post))
    pre, post, syn, sign = pre[order], post[order], syn0[order], sign0[order]
    crow = np.zeros(n + 1, dtype=np.int64)
    crow[1:] = np.cumsum(np.bincount(post, minlength=n))
    t_perm = np.lexsort((post, pre))
    t_crow = np.zeros(n + 1, dtype=np.int64)
    t_crow[1:] = np.cumsum(np.bincount(pre, minlength=n))

    a = ann.iloc[old_ids]
    sc = a.super_class.values
    cell_class = a.cell_class.fillna("").values
    if subset == "full":
        eye_mask = a.cell_type.isin(["R1-6", "R7", "R8"]).values
    else:
        eye_mask = sc == "visual_projection"
    sense_mask = ((sc == "sensory") | (sc == "sensory_ascending")) & (cell_class != "visual") & ~eye_mask
    read_mask = (sc == "descending") | (sc == "motor")

    # each eye (left/right) sees the whole arena, laid out retinotopically
    eye_idx, eye_cell = [], []
    sides = a.side.fillna("center").values
    pos = a[["pos_x", "pos_y", "pos_z"]].values.astype(np.float64)
    for side in np.unique(sides[eye_mask]):
        idx = np.flatnonzero(eye_mask & (sides == side))
        if len(idx) < gw * gh:
            continue
        p = pos[idx]
        good = np.isfinite(p).all(1)
        idx, p = idx[good], p[good]
        eye_idx.append(idx)
        eye_cell.append(_retinotopic_cells(p, gw, gh))
    eye_idx = np.concatenate(eye_idx)
    eye_cell = np.concatenate(eye_cell)

    out = dict(
        n=n, threshold=threshold, subset=subset,
        pre=torch.from_numpy(pre), post=torch.from_numpy(post),
        syn=torch.from_numpy(syn), sign=torch.from_numpy(sign),
        crow=torch.from_numpy(crow),
        t_perm=torch.from_numpy(t_perm), t_crow=torch.from_numpy(t_crow),
        eye_idx=torch.from_numpy(eye_idx), eye_cell=torch.from_numpy(eye_cell),
        sense_idx=torch.from_numpy(np.flatnonzero(sense_mask)),
        read_idx=torch.from_numpy(np.flatnonzero(read_mask)),
        super_class=list(sc), root_id=torch.from_numpy(a.index.values.astype(np.int64)),
    )
    return out


def load(subset="full", threshold=5):
    path = cache_path(subset, threshold)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: run `python data/fetch_connectome.py` then "
            f"`python -m fly.connectome --subset {subset} --threshold {threshold}`")
    return torch.load(path, weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=5, help="min synapses per connection")
    ap.add_argument("--subset", default="full", choices=["full", "central"])
    args = ap.parse_args()
    g = build(args.threshold, args.subset)
    path = cache_path(args.subset, args.threshold)
    torch.save(g, path)
    print(f"{path.name}: {g['n']:,} neurons, {len(g['syn']):,} connections "
          f"({int(g['syn'].sum()):,} synapses), "
          f"{(g['sign'] > 0).float().mean():.0%} excitatory")
    print(f"  eyes {len(g['eye_idx']):,} | other senses {len(g['sense_idx']):,} | "
          f"descending+motor readout {len(g['read_idx']):,}")


if __name__ == "__main__":
    main()
