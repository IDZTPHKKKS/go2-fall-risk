"""Information diagnostic: is a state's fate (fall within the horizon) predictable from the
world models' inputs?

A plain classifier with the world models' own encoders and transformer (same inputs: state,
history, scan, terrain map, parameters, planned commands; same width and depth) is trained
with log-loss on the fall label of each training window, then evaluated against the true
per-anchor fall probability of a branch set, next to the generative models' sampled
probabilities (results/branch_eval_v3_final_*.json).

- classifier good where the world models are not: the information is in the inputs and
  generative training loses it;
- classifier as bad: the inputs do not determine the outcome (hidden state).

Windows of one episode share its fall, and the terrain map and parameters identify the
episode, so the classifier can memorise training episodes; the checkpoint with the lowest
log-loss on held-out (val split) episodes is kept.

    python fall_classifier.py --data '../data/v3/train_*.h5' --branches '../data/final/branch_*.h5'
"""
import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from c2f import C2FWorldModel, sinusoid
from data import BranchSet, Corpus, TerrainCache, fall_label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FallClassifier(nn.Module):
    """World-model encoders + transformer; a class token over the prefix and plan tokens."""

    def __init__(self, horizon, d=256, depth=4):
        super().__init__()
        self.wm = C2FWorldModel(horizon, [horizon], d=d, depth=depth)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.head = nn.Linear(d, 1)

    def forward(self, b):
        wm, plan = self.wm, b["plan"]
        pos = (torch.arange(plan.shape[1], device=plan.device) + 0.5) / plan.shape[1]
        tok = wm.in_plan(plan) + wm.level_emb.weight[1] + sinusoid(pos * 50, wm.d)[None]
        tok = torch.cat([self.cls.expand(len(tok), -1, -1), tok], 1)
        h = wm._run(wm._prefix(b), [tok], raw=True)[0]
        return self.head(h[:, 0]).squeeze(-1)


def auc(score, p):
    us, inv = np.unique(score, return_inverse=True)
    w1, w0 = np.bincount(inv, weights=p), np.bincount(inv, weights=1 - p)
    below = np.concatenate([[0], np.cumsum(w0)[:-1]])
    return float((w1 * (below + 0.5 * w0)).sum() / (w1.sum() * w0.sum()))


def summary(p, pt):
    fk = (pt > 0.05) & (pt < 0.95)
    bins = {}
    for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.95), (0.95, 1.01)):
        m = (pt >= lo) & (pt < hi)
        if m.any():
            bins[f"{lo}-{min(hi, 1)}"] = [int(m.sum()), round(float(pt[m].mean()), 3),
                                          round(float(p[m].mean()), 3)]
    doomed = pt >= 0.95
    return dict(ratio_fork=float(p[fk].mean() / pt[fk].mean()),
                brier_fork=float(((p - pt) ** 2)[fk].mean()), brier_all=float(((p - pt) ** 2).mean()),
                auc=auc(p, pt), doomed_below_0_4=int((p[doomed] < 0.4).sum()), n_doomed=int(doomed.sum()),
                bins_n_true_pred=bins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data/v3/train_*.h5"))
    ap.add_argument("--branches", default=os.path.join(ROOT, "data/final/branch_*.h5"))
    ap.add_argument("--terrain-cache", default=os.path.join(ROOT, "data/v2/terrain_cache.h5"))
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="v3_fallclf")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    terrain = TerrainCache(args.terrain_cache) if os.path.exists(args.terrain_cache) else None
    if terrain is None:
        print("WARNING: no terrain cache, terrain map input is zero", flush=True)
    train = Corpus(args.data, "train", args.horizon, terrain=terrain)
    mean = torch.from_numpy(train.stats["s_mean"]).to(dev)
    std = torch.from_numpy(train.stats["s_std"]).to(dev)
    val = Corpus(args.data, "val", args.horizon, stats=train.stats, terrain=terrain)
    vrng = np.random.default_rng(12345)
    vb = [val.sample(vrng, args.batch) for _ in range(args.val_batches)]   # fixed val windows
    model = FallClassifier(args.horizon).to(dev)

    def val_loss():
        model.eval()
        tot = n = 0.0
        with torch.no_grad():
            for cb in vb:
                b = {k: v.to(dev) for k, v in cb.items()}
                y = fall_label(b["x"] * std + mean).float()
                tot += F.binary_cross_entropy_with_logits(model(b), y, reduction="sum").item()
                n += len(y)
        model.train()
        return tot / n

    yv = np.concatenate([fall_label(cb["x"].to(dev) * std + mean).float().cpu().numpy() for cb in vb])
    q = yv.mean()
    base = float(-(q * np.log(q) + (1 - q) * np.log(1 - q)))
    print(f"val: {len(val.meta)} episodes, {len(yv)} fixed windows, fall rate {q:.3f}, "
          f"constant-predictor log-loss {base:.4f}", flush=True)
    best = (float("inf"), 0, None)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                              pct_start=max(0.05, 2.5 / args.steps))
    t0, run, rate = time.time(), 0.0, 0.0
    for step in range(1, args.steps + 1):
        b = {k: v.to(dev) for k, v in train.sample(rng, args.batch).items()}
        y = fall_label(b["x"] * std + mean).float()
        loss = F.binary_cross_entropy_with_logits(model(b), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        run, rate = run + loss.item(), rate + y.mean().item()
        if step % 1000 == 0 or step == args.steps:
            k = 1000 if step % 1000 == 0 else step % 1000
            print(f"step {step}: log-loss {run / k:.4f}, fall rate in batches {rate / k:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            run = rate = 0.0
        if step % args.val_every == 0 or step == args.steps:
            v = val_loss()
            if v < best[0]:
                best = (v, step, copy.deepcopy(model.state_dict()))
            print(f"step {step}: val log-loss {v:.4f} (best {best[0]:.4f} at {best[1]})", flush=True)
    model.load_state_dict(best[2])
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), args=vars(args), stats=train.stats,
                    best_step=best[1], val_logloss=best[0]),
               os.path.join(ROOT, "runs", f"{args.name}.pt"))

    model.eval()
    bs = BranchSet(args.branches, args.horizon, train.stats, terrain)
    names, p, pt = [], [], []
    with torch.no_grad():
        for i, it in enumerate(bs.items):
            b = {k: v.to(dev) for k, v in bs.inputs(i, 1).items()}
            p.append(torch.sigmoid(model(b)).item())
            f = torch.from_numpy(it["futures"]).to(dev)
            pt.append(fall_label(f * std + mean).float().mean().item())
            names.append(it["name"])
    p, pt = np.array(p), np.array(pt)
    out = dict(classifier=summary(p, pt), n_anchors=len(p), best_step=best[1],
               val_logloss=best[0], val_constant_logloss=base)
    # the generative models on the same anchors (seed-averaged), if their evaluations exist
    gen = {}
    for path in glob.glob(os.path.join(ROOT, "results", "branch_eval_v3_final_*.json")):
        for k, v in json.load(open(path)).items():
            gen.setdefault(k.split("@")[0], {r["name"]: r["p_samp"] for r in v["rows"]})
    for fam in ("v3_cfm", "v3_diff", "v3_ncm4", "v2_cfm"):
        runs = [r for r in gen if r == fam or r.startswith(fam + "_s")]
        if runs and all(n in gen[r] for r in runs for n in names):
            out[fam] = summary(np.mean([[gen[r][n] for n in names] for r in runs], 0), pt)
            out[fam]["runs"] = runs
    out["per_anchor"] = dict(zip(names, p.round(4).tolist()))
    path = os.path.join(ROOT, "results", f"{args.name}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_anchor"}, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
