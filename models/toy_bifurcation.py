"""Toy bifurcation with known fall probability, using the same model code as the Go2 study.

A context scalar c in [0, 1] sets the fall probability p(c) = 0.01 * 50**c (log-uniform
from 1% to 50%). Given c, a trajectory of H steps over two channels (tilt, height) either
- recovers: a near-miss tilt bump of random size (peak 0.05-0.7, below the fall line 1.0),
  random timing and width, or
- falls: the tilt rises sigmoidally to ~1.6 (lying on the side) from a random onset.
Fall label for any trajectory: max tilt > 1.0. Because p(c) is known exactly, the
question is direct: does a model's sampled fall rate track p(c), and does the error grow
as falls get rarer?

Models (C2FWorldModel, toy channels packed into the first two state dims, c in par[0]):
flat CFM, flat diffusion, continuous NCM (levels 4..H), continuous NCM (levels 1..H).

    python toy_bifurcation.py --steps 8000
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from c2f import C2FWorldModel, HIST, MAP_DIM, PAR_DIM, PLAN_DIM, SCAN_DIM, STATE_DIM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = 32


def p_fall(c):
    return 0.01 * 50.0 ** c


def make(n, rng):
    c = rng.uniform(0, 1, n)
    fall = rng.random(n) < p_fall(c)
    t = np.arange(H)[None]
    tau = rng.uniform(6, 20, n)[:, None]
    k = rng.uniform(0.4, 0.8, n)[:, None]
    amp = rng.uniform(0.05, 0.7, n)[:, None]
    w = rng.uniform(2, 5, n)[:, None]
    tilt_f = 1.6 / (1 + np.exp(-k * (t - tau)))
    tilt_r = amp * np.exp(-((t - tau) / w) ** 2)
    tilt = np.where(fall[:, None], tilt_f, tilt_r) + 0.01 * rng.standard_normal((n, H))
    height = 0.3 - 0.12 * tilt + 0.005 * rng.standard_normal((n, H))
    return c.astype(np.float32), np.stack([tilt, height], -1).astype(np.float32), fall


class Toy:
    """Normalization + packing into the C2FWorldModel input format."""

    def __init__(self, x):
        self.mu, self.sd = x.reshape(-1, 2).mean(0), x.reshape(-1, 2).std(0)

    def pack(self, c, x=None):
        n = len(c)
        b = dict(s0=torch.zeros(n, STATE_DIM), hist=torch.zeros(n, HIST, STATE_DIM),
                 scan=torch.zeros(n, SCAN_DIM), tmap=torch.zeros(n, MAP_DIM),
                 par=torch.zeros(n, PAR_DIM), plan=torch.zeros(n, H, PLAN_DIM))
        b["par"][:, 0] = torch.from_numpy(2 * c - 1)
        if x is not None:
            xx = torch.zeros(n, H, STATE_DIM)
            xx[..., :2] = torch.from_numpy((x - self.mu) / self.sd)
            b["x"] = xx
        return b

    def tilt(self, y):
        return y[..., 0] * float(self.sd[0]) + float(self.mu[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--samples", type=int, default=1024)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    to = lambda b: {k: v.to(dev) for k, v in b.items()}
    rng = np.random.default_rng(0)
    c_tr, x_tr, f_tr = make(args.n_train, rng)
    toy = Toy(x_tr)
    print(f"train fall rate {f_tr.mean():.3f}", flush=True)
    specs = {"cfm": dict(levels=[H]), "diff": dict(levels=[H], mode="diff"),
             "ncm4": dict(levels=[4, 8, 16, H]), "ncm1": dict(levels=[1, 2, 4, 8, 16, H])}
    grid = np.linspace(0.0, 1.0, 11)
    results = dict(grid=grid.tolist(), p_true=p_fall(grid).tolist())
    for name, kw in specs.items():
        torch.manual_seed(0)
        m = C2FWorldModel(H, d=128, depth=3, **kw).to(dev)
        m.fit_level_std(to(toy.pack(c_tr[:8192], x_tr[:8192]))["x"])
        opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=args.steps, pct_start=0.05)
        t0 = time.time()
        for step in range(1, args.steps + 1):
            i = rng.integers(0, args.n_train, 256)
            loss, _ = m.loss(to(toy.pack(c_tr[i], x_tr[i])))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sch.step()
            if step % 2000 == 0:
                print(f"  {name} step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        m.eval()
        pred = []
        for c in grid:
            y = m.sample(to(toy.pack(np.full(args.samples, c, np.float32))), nfe=args.nfe)
            pred.append(float((toy.tilt(y).max(1).values > 1.0).float().mean()))
        results[name] = pred
        print(name, " ".join(f"{p:.3f}/{q:.3f}" for p, q in zip(pred, results["p_true"])), flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(results, open(os.path.join(ROOT, "results", "toy_bifurcation.json"), "w"), indent=1)
    print("predicted / true fall probability by c:")
    print("c      true   " + "  ".join(f"{n:>6s}" for n in specs))
    for j, c in enumerate(grid):
        print(f"{c:.1f}  {results['p_true'][j]:.3f}  " + "  ".join(f"{results[n][j]:.3f}" for n in specs))


if __name__ == "__main__":
    main()
