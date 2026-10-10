"""Brain tests run on a tiny synthetic connectome, so they need no download and little CPU."""
import torch

from fly.brain import FlyBrain
from game.env import NUM_ACTIONS, VecEnv


def tiny_connectome(n=300, e=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    pre = torch.randint(0, n, (e,), generator=g)
    post = torch.randint(0, n, (e,), generator=g)
    key = torch.unique(post * n + pre)
    post, pre = key // n, key % n
    crow = torch.zeros(n + 1, dtype=torch.long)
    crow[1:] = torch.bincount(post, minlength=n).cumsum(0)
    t_perm = torch.argsort(pre * n + post)
    t_crow = torch.zeros(n + 1, dtype=torch.long)
    t_crow[1:] = torch.bincount(pre, minlength=n).cumsum(0)
    sign = torch.where(torch.rand(len(pre), generator=g) < 0.6, 1, -1).to(torch.int8)
    return dict(n=n, pre=pre, post=post, syn=torch.randint(5, 20, (len(pre),), generator=g).float(),
                sign=sign, crow=crow, t_perm=t_perm, t_crow=t_crow,
                eye_idx=torch.arange(0, 160), eye_cell=torch.arange(160) % 144,
                sense_idx=torch.arange(160, 200), read_idx=torch.arange(250, 300))


def obs(b=3):
    env = VecEnv(b, seed=0)
    sp, vec, mask = env.obs()
    return (torch.from_numpy(sp[:, 0].copy()), torch.from_numpy(vec[:, 0].copy()),
            torch.from_numpy(mask[:, 0].copy()))


def test_forward_shapes_and_mask():
    brain = FlyBrain(tiny_connectome())
    sp, vec, mask = obs()
    logits, h = brain(sp, vec, mask, brain.initial_state(3))
    assert logits.shape == (3, NUM_ACTIONS) and h.shape == (3, 300)
    assert (logits[~mask] < -1e30).all()


def test_sparse_matmul_matches_dense():
    data = tiny_connectome()
    brain = FlyBrain(data)
    dense = torch.zeros(300, 300)
    dense[data["post"], data["pre"]] = brain.w0
    x = torch.rand(300, 4, requires_grad=True)
    y = brain.graph.mm(brain.w0, x)
    torch.testing.assert_close(y, dense @ x)
    y.sum().backward()
    torch.testing.assert_close(x.grad, dense.t() @ torch.ones(300, 4))


def test_no_grad_fast_path_matches_autograd_path():
    brain = FlyBrain(tiny_connectome())
    sp, vec, mask = obs()
    h0 = torch.rand(3, 300)
    with torch.no_grad():
        fast, hf = brain(sp, vec, mask, h0)
    slow, hs = brain(sp, vec, mask, h0)
    torch.testing.assert_close(hf, hs.detach())


def test_synapse_gains_get_gradients():
    brain = FlyBrain(tiny_connectome(), gains="synapse")
    sp, vec, mask = obs()
    logits, _ = brain(sp, vec, mask, torch.rand(3, 300))
    logits.masked_fill(~mask, 0).sum().backward()
    assert brain.log_gain.grad is not None and brain.log_gain.grad.abs().sum() > 0


def test_signs_never_flip_after_training_step():
    for gains in ("neuron", "synapse"):
        brain = FlyBrain(tiny_connectome(), gains=gains)
        opt = torch.optim.SGD(brain.parameters(), lr=100.0)
        sp, vec, mask = obs()
        logits, _ = brain(sp, vec, mask, torch.rand(3, 300))
        (-logits.masked_fill(~mask, 0).sum()).backward()
        opt.step()
        signs_before = torch.sign(brain.w0)
        w = brain.edge_weights().detach()
        assert torch.equal(torch.sign(w), signs_before)
        if gains == "neuron":
            assert (brain.log_gain_pre.exp() > 0).all() and (brain.log_gain_post.exp() > 0).all()
