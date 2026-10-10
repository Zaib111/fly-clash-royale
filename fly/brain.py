"""The fly brain as a policy network.

Rate model on the frozen FlyWire wiring (state h is one firing rate per neuron):

    h <- (1 - a) h + a * f( g_post * (W (g_pre * h)) + u + bias ),   f(x) = tanh(relu(x))

W[post, pre] = sign * synapses / (total input synapses of post). Topology and
signs (excitatory / inhibitory, from neurotransmitter predictions) never change.
Trainable: gains (per neuron, or per synapse with gains="synapse"), per-neuron
bias, the encoder that turns the game into input currents u on sensory neurons,
and the decoder that reads descending/motor neurons into an action.

h persists between decisions within a match, so the fly carries short-term
memory, and signals that take more than K steps to cross the brain still arrive.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn

from game.env import C, G, NUM_ACTIONS, V

_POOL = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4)))


def set_threads(n):
    """Limit CPU cores used by the brain (sparse worker threads + torch intra-op)."""
    global _POOL
    _POOL = ThreadPoolExecutor(max_workers=max(1, n))
    torch.set_num_threads(max(1, n))


class _Blocks:
    """CSR matrix split into row blocks with ~equal nnz, multiplied in parallel threads
    (torch releases the GIL inside sparse kernels)."""

    def __init__(self, crow, col, n, nblocks):
        nnz = int(crow[-1])
        bounds = [0]
        for k in range(1, nblocks):
            bounds.append(int(torch.searchsorted(crow, torch.tensor(nnz * k // nblocks))))
        bounds.append(n)
        self.n = n
        self.blocks = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b > a:
                self.blocks.append((a, b, int(crow[a]), int(crow[b]), crow[a:b + 1] - crow[a], col))

    def mm(self, vals, x):
        """(W with `vals`) @ x for x of shape (n, B)."""
        out = torch.empty(self.n, x.shape[1], dtype=x.dtype)
        vals, x = vals.detach(), x.detach()   # grad mode is thread-local; _SpMM handles grads

        def run(blk):
            a, b, s, e, crow, col = blk
            m = torch.sparse_csr_tensor(crow, col[s:e], vals[s:e], (b - a, self.n))
            out[a:b] = m @ x

        list(_POOL.map(run, self.blocks))
        return out


class _SpMM(torch.autograd.Function):
    """y = W x with gradients for x and (optionally) for the edge values of W."""

    @staticmethod
    def forward(ctx, vals, x, graph):
        ctx.graph = graph
        ctx.save_for_backward(vals, x)
        return graph.fwd.mm(vals, x)

    @staticmethod
    def backward(ctx, g):
        vals, x = ctx.saved_tensors
        graph = ctx.graph
        gx = graph.bwd.mm(vals[graph.t_perm], g.contiguous()) if ctx.needs_input_grad[1] else None
        gv = None
        if ctx.needs_input_grad[0]:
            gv = torch.empty_like(vals)
            chunk = 400_000
            for s in range(0, len(vals), chunk):
                e = slice(s, s + chunk)
                gv[e] = (g[graph.post[e]] * x[graph.pre[e]]).sum(1)
        return gv, gx, None


class Graph:
    def __init__(self, data, nblocks=None):
        n = data["n"]
        nblocks = nblocks or max(1, (os.cpu_count() or 4))
        self.n = n
        self.pre = data["pre"]
        self.post = data["post"]
        self.t_perm = data["t_perm"]
        self.fwd = _Blocks(data["crow"], data["pre"], n, nblocks)
        self.bwd = _Blocks(data["t_crow"], data["post"][data["t_perm"]], n, nblocks)

    def mm(self, vals, x):
        return _SpMM.apply(vals, x, self)


class FlyBrain(nn.Module):
    def __init__(self, data, steps=4, gains="neuron", leak=0.5, hidden=256, log_scale=1.0):
        super().__init__()
        self.graph = Graph(data)
        n = data["n"]
        self.n, self.steps, self.leak, self.gains = n, steps, leak, gains

        syn, sign, post = data["syn"], data["sign"].float(), data["post"]
        in_tot = torch.zeros(n).index_add_(0, post, syn)
        self.register_buffer("w0", sign * syn / in_tot[post].clamp(min=1.0))
        if gains == "synapse":
            self.log_gain = nn.Parameter(torch.zeros(len(syn)))
        elif gains == "neuron":
            self.log_gain_pre = nn.Parameter(torch.zeros(n))
            self.log_gain_post = nn.Parameter(torch.zeros(n))
        else:
            raise ValueError(gains)
        self.log_scale = nn.Parameter(torch.tensor(float(log_scale)))
        self.bias = nn.Parameter(torch.zeros(n))

        self.register_buffer("eye_idx", data["eye_idx"])
        self.register_buffer("eye_cell", data["eye_cell"])
        self.register_buffer("sense_idx", data["sense_idx"])
        self.register_buffer("read_idx", data["read_idx"])
        ne, ns, nr = len(data["eye_idx"]), len(data["sense_idx"]), len(data["read_idx"])
        # each photoreceptor sees the C feature planes at its own arena cell
        self.eye_w = nn.Parameter(torch.randn(ne, C) / C ** 0.5)
        self.eye_b = nn.Parameter(torch.zeros(ne))
        self.sense = nn.Linear(V, ns)
        self.decoder = nn.Sequential(nn.Linear(nr, hidden), nn.ReLU(), nn.Linear(hidden, NUM_ACTIONS))
        with torch.no_grad():
            self.decoder[-1].bias[0] = 4.0   # start out mostly waiting rather than spamming cards

    # -------------------------------------------------------------------- helpers
    def initial_state(self, batch):
        return torch.zeros(batch, self.n)

    def edge_weights(self):
        # the global scale is applied per neuron in think(), so in "neuron" mode the edge
        # values need no gradient (per-edge gradients are the expensive part)
        if self.gains == "synapse":
            return self.w0 * self.log_gain.exp()
        return self.w0

    def encode(self, spatial, vector):
        """Observation -> input current per neuron, shape (n, B)."""
        b = spatial.shape[0]
        flat = spatial.reshape(b, C, G)                       # (B, C, G)
        seen = flat[:, :, self.eye_cell]                      # (B, C, ne)
        eye_u = torch.einsum("bce,ec->eb", seen, self.eye_w) + self.eye_b[:, None]
        u = torch.zeros(self.n, b)
        u = u.index_add(0, self.eye_idx, eye_u)
        u = u.index_add(0, self.sense_idx, self.sense(vector).t())
        return u

    def think(self, h, u):
        """Run the recurrent dynamics for `steps` steps. h: (n, B)."""
        vals = self.edge_weights()
        a = self.leak
        ub = u + self.bias[:, None]
        neuron = self.gains == "neuron"
        g_post = self.log_scale.exp()
        if neuron:
            g_pre = self.log_gain_pre.exp()[:, None]
            g_post = g_post * self.log_gain_post.exp()[:, None]
        if not torch.is_grad_enabled():
            # in-place fast path for rollouts
            h = h.clone()
            for _ in range(self.steps):
                rec = self.graph.fwd.mm(vals, h * g_pre if neuron else h)
                rec.mul_(g_post)
                rec.add_(ub).relu_().tanh_()
                h.mul_(1 - a).add_(rec, alpha=a)
            return h
        for _ in range(self.steps):
            rec = self.graph.mm(vals, h * g_pre if neuron else h) * g_post
            h = (1 - a) * h + a * torch.tanh(torch.relu(rec + ub))
        return h

    def forward(self, spatial, vector, mask, state):
        """Returns (masked logits (B, A), new state (B, n))."""
        u = self.encode(spatial, vector)
        h = self.think(state.t().contiguous(), u)
        readout = h[self.read_idx].t()                       # (B, nr)
        logits = self.decoder(readout)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, h.t()
