"""Which neurons the brain view draws, where, and in which colour group."""
import numpy as np

GROUPS = ["central brain", "eyes & optic lobes", "other senses", "descending & motor"]


def brain_points(data, n_points=60_000, seed=0):
    """Returns (idx, pos, group): a subset of neurons (always including every sensory and
    descending/motor neuron), their positions scaled to about [-1, 1] in three.js axes
    (x lateral, y up, z toward the viewer), and a group id per point (index into GROUPS)."""
    n = data["n"]
    sc = np.array(data["super_class"])
    group = np.zeros(n, dtype=np.uint8)
    group[np.isin(sc, ["optic", "visual_projection", "visual_centrifugal"])] = 1
    group[data["eye_idx"].numpy()] = 1
    group[data["sense_idx"].numpy()] = 2
    group[data["read_idx"].numpy()] = 3
    must = np.unique(np.concatenate([data["read_idx"].numpy(), data["sense_idx"].numpy()]))
    rest = np.setdiff1d(np.arange(n), must)
    k = max(0, min(len(rest), n_points - len(must)))
    idx = np.sort(np.concatenate([must, np.random.default_rng(seed).choice(rest, k, replace=False)]))
    pos = data["pos"].numpy()[idx].astype(np.float64) * np.array([4.0, 4.0, 40.0])   # voxels -> nm
    pos -= pos.mean(0)
    pos /= np.abs(pos).max()
    pos = np.stack([pos[:, 0], -pos[:, 1], -pos[:, 2]], 1).astype(np.float32)        # FlyWire y points ventral
    return idx, pos, group[idx]


def activity_bytes(h_shown):
    """Activity for the brain view: scaled to this moment's most active neurons, 0-255."""
    scale = max(float(np.percentile(h_shown, 99.5)), 0.05)
    return (np.clip(h_shown / scale, 0, 1) ** 0.8 * 255).astype(np.uint8)
