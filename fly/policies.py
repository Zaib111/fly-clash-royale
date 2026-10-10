"""Build / save / load policies (fly brain or MLP baseline)."""
import torch

from . import connectome
from .brain import FlyBrain
from .mlp_baseline import MLPPolicy

# buffers rebuilt from the connectome file, so they're not stored in checkpoints
_DERIVED = {"w0", "eye_idx", "eye_cell", "sense_idx", "read_idx"}


def make_policy(cfg: dict):
    if cfg["policy"] == "mlp":
        return MLPPolicy()
    data = connectome.load(cfg.get("subset", "full"), cfg.get("threshold", 5))
    return FlyBrain(data, steps=cfg.get("steps", 4), gains=cfg.get("gains", "neuron"),
                    log_scale=cfg.get("log_scale", 1.5))


def params_of(policy):
    """CPU copy of the learnable state (what differs between snapshots)."""
    return {k: v.detach().clone() for k, v in policy.state_dict().items() if k not in _DERIVED}


def load_params(policy, params):
    policy.load_state_dict(params, strict=False)


def save_checkpoint(path, policy, cfg, extra=None):
    torch.save({"cfg": cfg, "params": params_of(policy), **(extra or {})}, path)


def load_checkpoint(path):
    ck = torch.load(path, weights_only=False)
    policy = make_policy(ck["cfg"])
    load_params(policy, ck["params"])
    policy.eval()
    return policy, ck
