"""Train a world model (C-NCM or a baseline) on the Go2 MuJoCo dataset.

Examples:
    python3 train.py --name cfm   --levels H
    python3 train.py --name reg   --levels H --mode reg
    python3 train.py --name ncm   --levels pyramid
    python3 train.py --name ncm_full --levels pyramid --no-residual   # HiFlow-style
    python3 train.py --name ncm_nocorr --levels pyramid --corrupt 0
    python3 train.py --name diff  --levels H --mode diff
    python3 train.py --name concept --levels pyramid --concept-k 256   # concept-first
--duty < 1 pauses between steps so the Mac's GPU/CPU stay cool.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from c2f import C2FWorldModel
from data import Corpus, TerrainCache, fall_label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_levels(spec, H):
    if spec == "H":
        return [H]
    if spec == "pyramid":
        ls, l = [], 1
        while l < H:
            ls.append(l)
            l *= 2
        return ls + [H]
    return [int(v) for v in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data/pilot/*.h5"))
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--levels", default="pyramid")
    ap.add_argument("--mode", default="flow", choices=["flow", "reg", "diff"])
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--corrupt", type=float, default=0.5)
    ap.add_argument("--concept-k", type=int, default=0)
    ap.add_argument("--corrupt-concept", type=float, default=0.1)
    ap.add_argument("--norm", default="zscore", choices=["zscore", "robust"])
    ap.add_argument("--terrain-cache", default=os.path.join(ROOT, "data/v2/terrain_cache.h5"))
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duty", type=float, default=0.6)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    H = args.horizon
    levels = parse_levels(args.levels, H)

    terrain = TerrainCache(args.terrain_cache) if os.path.exists(args.terrain_cache) else None
    train = Corpus(args.data, "train", H, terrain=terrain, norm=args.norm)
    if terrain is None:
        print("WARNING: no terrain cache, terrain map input is zero", flush=True)
    print(f"{args.name}: {len(train.meta)} train episodes, {len(train.starts)} windows, "
          f"levels {levels}, device {dev}", flush=True)
    model = C2FWorldModel(H, levels, d=args.d, depth=args.depth, residual=not args.no_residual,
                          corrupt=args.corrupt, alpha=args.alpha, mode=args.mode,
                          concept_k=args.concept_k, corrupt_concept=args.corrupt_concept).to(dev)
    if args.concept_k:
        xs = train.sample(rng, 50000)["x"].to(dev)
        raw = xs * torch.from_numpy(train.stats["s_std"]).to(dev) + torch.from_numpy(train.stats["s_mean"]).to(dev)
        cnt = model.fit_concepts(xs, fall_label(raw))
        print(f"concepts: {int((cnt > 0).sum())}/{args.concept_k} used, "
              f"fall rate per concept max {model.code_fall.max():.2f}", flush=True)
    model.fit_level_std(train.sample(rng, 4096)["x"].to(dev))
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                                pct_start=max(0.05, 2.5 / args.steps))
    t0, hist = time.time(), []
    for step in range(1, args.steps + 1):
        ts = time.time()
        b = {k: v.to(dev) for k, v in train.sample(rng, args.batch).items()}
        loss, per_level = model.loss(b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss))
        if args.duty < 1:
            if dev == "mps":
                torch.mps.synchronize()
            time.sleep((time.time() - ts) * (1 / args.duty - 1))
        if step % min(1000, max(args.steps // 3, 1)) == 0:
            print(f"  step {step}: loss {np.mean(hist[-1000:]):.4f} per-level "
                  f"{[round(v, 3) for v in per_level.tolist()]} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), args=vars(args), levels=levels,
                    stats={k: np.asarray(v).tolist() for k, v in train.stats.items()}),
               os.path.join(ROOT, "runs", f"{args.name}.pt"))
    json.dump(dict(vars(args), params=n_par, levels=levels, final_loss=float(np.mean(hist[-1000:])),
                   seconds=time.time() - t0),
              open(os.path.join(ROOT, "runs", f"{args.name}.json"), "w"), indent=1)
    print(f"saved runs/{args.name}.pt ({n_par / 1e6:.2f}M params)", flush=True)


if __name__ == "__main__":
    main()
