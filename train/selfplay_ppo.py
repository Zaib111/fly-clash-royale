"""Self-play PPO: the policy learns Clash Royale purely from wins and losses.

    python -m train.selfplay_ppo --policy fly --minutes 600 --run fly1
    python -m train.selfplay_ppo --policy mlp --minutes 30  --run mlp1

Each match pits the learner (seat 0) against either its current self (both seats
learn) or a frozen past snapshot from the opponent pool, picked with priority
toward snapshots it struggles against. The bots in game/bots.py are only used
for evaluation.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fly.policies import load_params, make_policy, params_of, save_checkpoint
from game.bots import RandomBot, ScriptedBot
from game.env import C, GH, GW, NUM_ACTIONS, SHAPING, V, VecEnv

ROOT = Path(__file__).resolve().parent.parent
SELF = -1


class Critic(nn.Module):
    """Value function on the raw observation (only used for training, not for playing)."""

    def __init__(self, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(32 * 8 * 5 + V, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, spatial, vector):
        x = self.conv(spatial).flatten(1)
        return self.head(torch.cat([x, vector], 1)).squeeze(1)


class Pool:
    """Past snapshots + Elo + per-opponent learner win rate (for prioritised sampling)."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.params, self.elo, self.winrate, self.ids = {}, {}, {}, []
        self.current_elo = 1000.0
        self._next = 0

    def add(self, params):
        i = self._next
        self._next += 1
        self.params[i] = params
        self.elo[i] = self.current_elo
        self.winrate[i] = 0.5
        self.ids.append(i)
        if len(self.ids) > self.max_size:
            old = self.ids.pop(0)
            del self.params[old], self.winrate[old]
        return i

    def sample(self, rng, k):
        ids = self.ids
        w = np.array([(1.0 - self.winrate[i]) ** 2 + 0.05 for i in ids])
        k = min(k, len(ids))
        picks = rng.choice(len(ids), size=k, replace=False, p=w / w.sum())
        return [ids[j] for j in picks]

    def record(self, opp, score):
        """score: 1 learner win, 0.5 draw, 0 loss."""
        if opp not in self.winrate:
            return
        self.winrate[opp] = 0.95 * self.winrate[opp] + 0.05 * score
        ea = 1 / (1 + 10 ** ((self.elo[opp] - self.current_elo) / 400))
        self.current_elo += 16 * (score - ea)
        self.elo[opp] -= 16 * (score - ea)


def _t(x):
    return torch.from_numpy(np.array(x))   # copy: env buffers are overwritten by env.step


@torch.no_grad()
def act(policy, spatial, vector, mask, state, greedy=False):
    logits, state = policy(spatial, vector, mask, state)
    if greedy:
        a = logits.argmax(1)
    else:
        a = torch.distributions.Categorical(logits=logits).sample()
    logp = torch.log_softmax(logits, 1).gather(1, a[:, None]).squeeze(1)
    return a, logp, state


@torch.no_grad()
def evaluate(policy, bot_cls, n, seed):
    """Learner (seat 0) vs a scripted bot for n full matches. Returns (win, draw, loss)."""
    env = VecEnv(n, seed=seed)
    bots = [bot_cls(seed=seed + i) for i in range(n)]
    state = policy.initial_state(n)
    finished = np.zeros(n, dtype=bool)
    res = [0, 0, 0]
    while not finished.all():
        sp, vec, mask = env.obs()
        a0, _, state = act(policy, _t(sp[:, 0]), _t(vec[:, 0]), _t(mask[:, 0]), state)
        a1 = [bots[i].act(m, 1) for i, m in enumerate(env.matches)]
        _, done, winner = env.step(a0.numpy(), a1, shaping=0.0)
        for i in np.flatnonzero(done & ~finished):
            res[0 if winner[i] == 0 else 1 if winner[i] == -1 else 2] += 1
            finished[i] = True
        state = state * torch.from_numpy(~done).float()[:, None]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["fly", "mlp"], default="fly")
    ap.add_argument("--run", default=None)
    ap.add_argument("--subset", choices=["full", "central"], default="full")
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--gains", choices=["neuron", "synapse"], default="neuron")
    ap.add_argument("--steps", type=int, default=4, help="brain steps per decision")
    ap.add_argument("--envs", type=int, default=None)
    ap.add_argument("--rollout", type=int, default=None, help="decisions per iteration")
    ap.add_argument("--chunk", type=int, default=8, help="BPTT length (decisions)")
    ap.add_argument("--minibatch", type=int, default=None, help="sequences per minibatch")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--p-self", type=float, default=0.2)
    ap.add_argument("--active-opponents", type=int, default=3)
    ap.add_argument("--pool-size", type=int, default=30)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--shaping-anneal", type=int, default=300, help="iterations to fade shaping")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--iters", type=int, default=10 ** 9)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU cores to use (default: all); lower it to keep your machine usable")
    args = ap.parse_args()
    if args.threads:
        from fly.brain import set_threads
        set_threads(args.threads)

    fly = args.policy == "fly"
    args.envs = args.envs or (32 if fly else 64)
    args.rollout = args.rollout or (64 if fly else 128)
    args.minibatch = args.minibatch or (16 if fly else 64)
    if not fly:
        args.chunk = 16
    run = ROOT / "runs" / (args.run or f"{args.policy}-{time.strftime('%m%d-%H%M')}")
    run.mkdir(parents=True, exist_ok=True)
    cfg = {k: getattr(args, k) for k in ("policy", "subset", "threshold", "gains", "steps")}
    (run / "args.json").write_text(json.dumps(vars(args), indent=2))

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    policy = make_policy(cfg)
    opp_net = make_policy(cfg)
    critic = Critic()
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=args.lr)
    pool = Pool(args.pool_size)
    it0 = 0
    if args.resume and (run / "latest.pt").exists():
        ck = torch.load(run / "latest.pt", weights_only=False)
        load_params(policy, ck["params"])
        critic.load_state_dict(ck["critic"])
        opt.load_state_dict(ck["opt"])
        pool = ck["pool"]
        it0 = ck["iter"] + 1
        print(f"resumed from iteration {it0}")
    else:
        pool.add(params_of(policy))
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"{args.policy} policy: {n_params:,} trainable params -> {run}")

    n, T, S = args.envs, args.rollout, 2 * args.envs
    env = VecEnv(n, seed=args.seed * 100_000)
    opp = np.full(n, SELF)
    active = pool.sample(rng, args.active_opponents)

    def new_opponent():
        if rng.random() < args.p_self or not active:
            return SELF
        return int(rng.choice(active))

    for i in range(n):
        opp[i] = new_opponent()
    state = policy.initial_state(S)

    log_f = open(run / "log.csv", "a", newline="")
    log = csv.writer(log_f)
    if it0 == 0:
        log.writerow(["iter", "minutes", "samples", "sps", "pi_loss", "v_loss", "entropy", "kl",
                      "play_rate", "match_len", "elo", "pool_winrate", "eval_random_win",
                      "eval_scripted_win", "eval_scripted_draw"])
    t_start = time.time()
    total_samples = 0
    match_lens, results = [], []
    step_in_match = np.zeros(n, dtype=np.int64)

    for it in range(it0, args.iters):
        if (time.time() - t_start) / 60 > args.minutes:
            break
        shaping = SHAPING * max(0.0, 1 - it / args.shaping_anneal)
        active = pool.sample(rng, args.active_opponents)
        for i in range(n):
            if opp[i] != SELF and opp[i] not in active:
                opp[i] = new_opponent()
                state[2 * i + 1] = 0

        # ---------------------------------------------------------------- rollout
        b_sp = torch.zeros(T, S, C, GH, GW)
        b_vec = torch.zeros(T, S, V)
        b_mask = torch.zeros(T, S, NUM_ACTIONS, dtype=torch.bool)
        b_act = torch.zeros(T, S, dtype=torch.long)
        b_logp = torch.zeros(T, S)
        b_rew = torch.zeros(T, S)
        b_done = torch.zeros(T, S)
        b_valid = torch.zeros(T, S, dtype=torch.bool)
        n_chunks = math.ceil(T / args.chunk)
        b_h0 = torch.zeros((n_chunks,) + tuple(state.shape))
        t_roll = time.time()
        for t in range(T):
            if t % args.chunk == 0:
                b_h0[t // args.chunk] = state
            sp, vec, mask = env.obs()
            sp, vec, mask = (_t(sp.reshape(S, C, GH, GW)), _t(vec.reshape(S, V)),
                             _t(mask.reshape(S, NUM_ACTIONS)))
            learner = np.zeros(S, dtype=bool)
            learner[0::2] = True
            learner[1::2] = opp == SELF
            actions = torch.zeros(S, dtype=torch.long)
            li = torch.from_numpy(np.flatnonzero(learner))
            a, lp, st = act(policy, sp[li], vec[li], mask[li], state[li])
            actions[li], state[li] = a, st
            b_logp[t, li] = lp
            for o in set(opp[opp != SELF].tolist()):
                oi = torch.from_numpy(2 * np.flatnonzero(opp == o) + 1)
                load_params(opp_net, pool.params[o])
                a, _, st = act(opp_net, sp[oi], vec[oi], mask[oi], state[oi])
                actions[oi], state[oi] = a, st
            rew, done, winner = env.step(actions[0::2].numpy(), actions[1::2].numpy(), shaping)
            b_sp[t], b_vec[t], b_mask[t], b_act[t] = sp, vec, mask, actions
            b_rew[t] = torch.from_numpy(rew.reshape(S))
            b_done[t] = torch.from_numpy(np.repeat(done, 2)).float()
            b_valid[t] = torch.from_numpy(learner)
            step_in_match += 1
            for i in np.flatnonzero(done):
                score = 1.0 if winner[i] == 0 else 0.5 if winner[i] == -1 else 0.0
                if opp[i] != SELF:
                    pool.record(int(opp[i]), score)
                    results.append(score)
                match_lens.append(step_in_match[i] / 2)   # seconds
                step_in_match[i] = 0
                opp[i] = new_opponent()
                state[2 * i:2 * i + 2] = 0
        roll_time = time.time() - t_roll

        # ---------------------------------------------------------------- advantages
        with torch.no_grad():
            sp, vec, _ = env.obs()
            v_last = critic(_t(sp.reshape(S, C, GH, GW)), _t(vec.reshape(S, V)))
            values = torch.cat([critic(b_sp[t], b_vec[t])[None] for t in range(T)])
        adv = torch.zeros(T, S)
        gae = torch.zeros(S)
        for t in reversed(range(T)):
            nxt = v_last if t == T - 1 else values[t + 1]
            nonterm = 1 - b_done[t]
            delta = b_rew[t] + args.gamma * nxt * nonterm - values[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[t] = gae
        ret = adv + values
        v_adv = adv[b_valid]
        adv = (adv - v_adv.mean()) / (v_adv.std() + 1e-8)

        # ---------------------------------------------------------------- PPO update
        t_upd = time.time()
        seqs = [(c, s) for c in range(n_chunks) for s in range(S)
                if b_valid[c * args.chunk:(c + 1) * args.chunk, s].any()]
        stats = []
        policy.train()
        for _ in range(args.epochs):
            random.shuffle(seqs)
            for k in range(0, len(seqs), args.minibatch):
                mb = seqs[k:k + args.minibatch]
                cs = torch.tensor([c for c, _ in mb])
                ss = torch.tensor([s for _, s in mb])
                h = b_h0[cs, ss]
                pl, vl, ent, kl, cnt = 0.0, 0.0, 0.0, 0.0, 0
                for j in range(args.chunk):
                    ts = cs * args.chunk + j
                    ok = ts < T
                    if not ok.any():
                        break
                    ts_ok, ss_ok = ts[ok], ss[ok]
                    logits, h_new = policy(b_sp[ts_ok, ss_ok], b_vec[ts_ok, ss_ok],
                                           b_mask[ts_ok, ss_ok], h[ok])
                    valid = b_valid[ts_ok, ss_ok]
                    logp_all = torch.log_softmax(logits, 1)
                    logp = logp_all.gather(1, b_act[ts_ok, ss_ok][:, None]).squeeze(1)
                    ratio = (logp - b_logp[ts_ok, ss_ok]).exp()
                    a_ = adv[ts_ok, ss_ok]
                    surr = torch.min(ratio * a_, ratio.clamp(1 - args.clip, 1 + args.clip) * a_)
                    e = -(logp_all.exp() * logp_all).sum(1)
                    v = critic(b_sp[ts_ok, ss_ok], b_vec[ts_ok, ss_ok])
                    vf = (v - ret[ts_ok, ss_ok]) ** 2
                    w = valid.float()
                    pl = pl - (surr * w).sum()
                    ent = ent + (e * w).sum()
                    vl = vl + (vf * w).sum()
                    with torch.no_grad():
                        kl += ((b_logp[ts_ok, ss_ok] - logp) * w).sum().item()
                    cnt += int(w.sum())
                    keep = (1 - b_done[ts_ok, ss_ok])[:, None]
                    h = h.clone()
                    h[ok] = h_new * keep
                if cnt == 0:
                    continue
                loss = (pl - args.ent * ent + 0.5 * vl) / cnt
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), 0.5)
                opt.step()
                stats.append(((pl / cnt).item(), (vl / cnt).item(), (ent / cnt).item(), kl / cnt))
        policy.eval()
        upd_time = time.time() - t_upd

        # ---------------------------------------------------------------- bookkeeping
        samples = int(b_valid.sum())
        total_samples += samples
        if (it + 1) % args.snapshot_every == 0:
            pool.add(params_of(policy))
        ev_r = ev_s = ev_sd = ""
        if args.eval_every and (it + 1) % args.eval_every == 0:
            wr = evaluate(policy, lambda seed: RandomBot(seed=seed), args.eval_n, 10_000 + it)
            ws = evaluate(policy, ScriptedBot, args.eval_n, 20_000 + it)
            ev_r, ev_s, ev_sd = wr[0] / args.eval_n, ws[0] / args.eval_n, ws[1] / args.eval_n
            print(f"  eval vs random W/D/L {wr} | vs scripted W/D/L {ws}")
        st = np.mean(stats, 0) if stats else [0, 0, 0, 0]
        play_rate = (b_act[b_valid] != 0).float().mean().item()
        mins = (time.time() - t_start) / 60
        mlen = np.mean(match_lens[-50:]) if match_lens else 0
        pwr = np.mean(results[-100:]) if results else 0.5
        log.writerow([it, f"{mins:.2f}", total_samples, f"{samples / (roll_time + upd_time):.1f}",
                      f"{st[0]:.4f}", f"{st[1]:.4f}", f"{st[2]:.3f}", f"{st[3]:.4f}",
                      f"{play_rate:.3f}", f"{mlen:.1f}", f"{pool.current_elo:.0f}", f"{pwr:.3f}",
                      ev_r, ev_s, ev_sd])
        log_f.flush()
        print(f"it {it:4d} | {mins:6.1f} min | {samples} samples ({roll_time:.0f}s roll, "
              f"{upd_time:.0f}s update) | ent {st[2]:.2f} kl {st[3]:.4f} | play {play_rate:.2f} | "
              f"elo {pool.current_elo:.0f} | vs pool {pwr:.2f} | match {mlen:.0f}s", flush=True)
        save_checkpoint(run / "latest.pt", policy, cfg,
                        {"critic": critic.state_dict(), "opt": opt.state_dict(), "pool": pool,
                         "iter": it})
        if (it + 1) % 50 == 0:
            save_checkpoint(run / f"iter{it + 1:05d}.pt", policy, cfg)
    log_f.close()
    print("done")


if __name__ == "__main__":
    main()
