"""Evaluate world models against branching ground truth (v2 branch files).

For every anchor, the model draws S futures and is compared with the N simulated futures
from the identical simulator state:

- fall probability: model (fraction of samples that fall; concept-first models also give
  the exact sum_k p(k) P(fall | k)) vs truth (fraction of branches that fall);
  Brier score and cross-entropy against the true probability, per anchor
- energy distance between the sample set and the branch set (normalized states)
- valley mass ("in-between futures"): true outcomes at forks are bimodal (upright vs
  fallen) with a sparsely populated valley between. Two physically defined valleys:
    * teetering at the horizon end: final body tilt in [0.9, 1.5] rad (past the 1 rad
      fall line, short of lying on the side at ~pi/2)
    * near-threshold severity: s in [0.9, 1.6), with s = max over time of
      max(tilt / 1 rad, (0.30 m - height) / 0.18 m) (s >= 1 is the generator's fall test)
  Reported per model as pooled mass over fork anchors, next to the true pooled mass
  from the branches; the ratio model/true measures over-population of the valley.
- gap mass: share of samples inside the empty interval (if any) between the largest
  true recovery severity and the smallest true fall severity at an anchor (often tiny,
  so secondary); truth baseline from a split-half of the branches (gap_truth).
- nearest-branch distance of samples (precision) vs of held-out branches (reference)
- best-of-S error (coverage) and mean error to the branch mean

Usage:
    python evaluate.py --runs cfm diff concept --branches '../data/v2/branch_*.h5' --nfe 1 4 16
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from c2f import C2FWorldModel
from data import BranchSet, TerrainCache, fall_label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, dev):
    ck = torch.load(os.path.join(ROOT, "runs", f"{name}.pt"), map_location=dev, weights_only=False)
    a = ck["args"]
    m = C2FWorldModel(a["horizon"], ck["levels"], d=a["d"], depth=a["depth"],
                      residual=not a["no_residual"], corrupt=a["corrupt"], alpha=a["alpha"],
                      mode=a["mode"], concept_k=a.get("concept_k", 0),
                      corrupt_concept=a.get("corrupt_concept", 0.1)).to(dev)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), {k: np.asarray(v, np.float32) for k, v in ck["stats"].items()}


def traj_dist(a, b):
    """RMS distance between trajectory sets a (n, H, D), b (m, H, D) -> (n, m)."""
    return torch.cdist(a.flatten(1), b.flatten(1)) / np.sqrt(a.shape[1] * a.shape[2])


def severity(x_raw):
    """(..., H, 46) raw states -> (...,) fall severity; >= 1 means a fall (see fall_label)."""
    tilt = torch.arccos((-x_raw[..., 3]).clamp(-1, 1))
    phi = torch.maximum(tilt / 1.0, (0.30 - x_raw[..., 0]) / 0.18)
    return phi.max(-1).values


def final_tilt(x_raw):
    return torch.arccos((-x_raw[..., -1, 3]).clamp(-1, 1))


VALLEY_TILT, VALLEY_SEV = (0.9, 1.5), (0.9, 1.6)


def in_range(v, r):
    return ((v >= r[0]) & (v <= r[1])).float().mean().item()


def gap_interval(s_true):
    """Empty interval between recoveries and falls among true severities, or None."""
    rec, fall = s_true[s_true < 1], s_true[s_true >= 1]
    if len(rec) == 0 or len(fall) == 0:
        return None
    lo, hi = rec.max().item(), fall.min().item()
    return (lo, hi) if hi > lo else None


def energy_distance(x, y):
    return (2 * traj_dist(x, y).mean() - traj_dist(x, x).mean() - traj_dist(y, y).mean()).item()


@torch.no_grad()
def evaluate(model, stats, bs, n_samples, nfe, eta, dev, ref=None, seed=0):
    """ref: (mean, std) of a fixed reference normalization; all distance metrics are
    computed in it, so models trained with different normalizations stay comparable."""
    torch.manual_seed(seed)
    mean = torch.from_numpy(stats["s_mean"]).to(dev)
    std = torch.from_numpy(stats["s_std"]).to(dev)
    ref = ref or stats
    r_mean = torch.from_numpy(np.asarray(ref["s_mean"], np.float32)).to(dev)
    r_std = torch.from_numpy(np.asarray(ref["s_std"], np.float32)).to(dev)
    to_ref = lambda z: (z * std + mean - r_mean) / r_std
    rows = []
    g = torch.Generator().manual_seed(seed)
    t_sample = 0.0
    for i, it in enumerate(bs.items):
        b = {k: v.to(dev) for k, v in bs.inputs(i, n_samples).items()}
        ts = time.time()
        y = model.sample(b, nfe=nfe, eta=eta)
        if dev == "cuda":
            torch.cuda.synchronize()
        t_sample += time.time() - ts
        f = torch.from_numpy(it["futures"]).to(dev)
        p_true = fall_label(f * std + mean).float().mean().item()
        y_r, f_r = to_ref(y), to_ref(f)
        p_samp = fall_label(y * std + mean).float().mean().item()
        if model.K:
            probs = model.concept_probs({k: v[:1] for k, v in b.items()})[0]
            p_exact = float((probs * model.code_fall).sum())
        else:
            p_exact = p_samp
        d_mt = traj_dist(y_r, f_r)
        d_tt = traj_dist(f_r, f_r) + torch.eye(len(f), device=dev) * 1e9
        s_true = severity(f * std + mean).cpu()
        s_samp = severity(y * std + mean).cpu()
        ft_true, ft_samp = final_tilt(f * std + mean).cpu(), final_tilt(y * std + mean).cpu()
        gap = gap_interval(s_true)
        gap_mass = gap_truth = None
        if gap is not None:
            gap_mass = ((s_samp > gap[0]) & (s_samp < gap[1])).float().mean().item()
            perm = torch.randperm(len(s_true), generator=g)
            half_a, half_b = s_true[perm[:len(perm) // 2]], s_true[perm[len(perm) // 2:]]
            gap_a = gap_interval(half_a)
            if gap_a is not None:
                gap_truth = ((half_b > gap_a[0]) & (half_b < gap_a[1])).float().mean().item()
        rows.append(dict(name=it["name"], terrain=it["terrain"], push_level=it["push_level"],
                         p_true=p_true, p_samp=p_samp, p_exact=p_exact,
                         energy=energy_distance(y_r, f_r),
                         gap_mass=gap_mass, gap_truth=gap_truth,
                         valley_tilt=in_range(ft_samp, VALLEY_TILT),
                         valley_tilt_true=in_range(ft_true, VALLEY_TILT),
                         valley_sev=in_range(s_samp, VALLEY_SEV),
                         valley_sev_true=in_range(s_true, VALLEY_SEV),
                         gap=None if gap is None else list(gap),
                         sev_q_true=np.percentile(s_true.numpy(), [50, 90]).tolist(),
                         sev_q_samp=np.percentile(s_samp.numpy(), [50, 90]).tolist(),
                         nn_dist=d_mt.min(1).values.mean().item(),     # each sample's closest branch
                         nn_true=d_tt.min(1).values.mean().item(),     # held-out branch reference
                         best_of=d_mt.min(0).values.mean().item(),     # each branch's closest sample
                         mean_err=traj_dist(y_r.mean(0, keepdim=True), f_r.mean(0, keepdim=True)).item()))
    return rows, t_sample / len(bs.items)


def summarize(rows):
    pt = np.array([r["p_true"] for r in rows])
    out = {}
    for key in ("p_samp", "p_exact"):
        p = np.clip(np.array([r[key] for r in rows]), 1e-3, 1 - 1e-3)
        out[f"brier_{key}"] = float(np.mean((p - pt) ** 2))
        out[f"xent_{key}"] = float(np.mean(-(pt * np.log(p) + (1 - pt) * np.log(1 - p))))
    bif = (pt > 0.05) & (pt < 0.95)                     # anchors with a genuine fork
    for key in ("energy", "nn_dist", "nn_true", "best_of", "mean_err"):
        v = np.array([r[key] for r in rows])
        out[key] = float(v.mean())
        out[key + "_fork"] = float(v[bif].mean()) if bif.any() else None
    gm = [(r["gap_mass"], r["gap_truth"]) for r in rows
          if r["gap_mass"] is not None and r["gap_truth"] is not None]
    out["gap_mass"] = float(np.mean([a for a, _ in gm])) if gm else None
    out["gap_truth"] = float(np.mean([b for _, b in gm])) if gm else None
    fork = (pt > 0) & (pt < 1)
    for key in ("valley_tilt", "valley_sev"):
        m = np.array([r[key] for r in rows])[fork]
        t = np.array([r[key + "_true"] for r in rows])[fork]
        out[key], out[key + "_true"] = float(m.mean()), float(t.mean())
        out[key + "_ratio"] = float(m.mean() / max(t.mean(), 1e-9))
    out["n_anchors"], out["n_fork"], out["n_gap"] = len(rows), int(bif.sum()), len(gm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--branches", default=os.path.join(ROOT, "data/v2/branch_*.h5"))
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--nfe", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--max-anchors", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    bs, bs_key = None, None
    # reference normalization for distances: the z-score CFM run if present, else the first run
    ref_run = "v2_cfm" if os.path.exists(os.path.join(ROOT, "runs", "v2_cfm.pt")) else args.runs[0]
    ref = load(ref_run, dev)[1] if os.path.exists(os.path.join(ROOT, "runs", f"{ref_run}.pt")) else None
    print("distance reference normalization:", ref_run, flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", f"branch_eval{args.tag}.json")
    for name in args.runs:
        if not os.path.exists(os.path.join(ROOT, "runs", f"{name}.pt")):
            print(f"SKIP {name}: no checkpoint", flush=True)
            continue
        model, stats = load(name, dev)
        key = stats["s_std"].tobytes() + stats["s_mean"].tobytes()
        if bs is None or key != bs_key:          # models with different normalization
            bs_key = key
            tc = os.path.join(ROOT, "data/v2/terrain_cache.h5")
            bs = BranchSet(args.branches, model.H, stats,
                           TerrainCache(tc) if os.path.exists(tc) else None)
            if args.max_anchors:
                bs.items = bs.items[:args.max_anchors]
            print(f"{len(bs.items)} anchors", flush=True)
        for nfe in (args.nfe if model.mode != "reg" else [1]):
            rows, sec = evaluate(model, stats, bs, args.samples, nfe, args.eta, dev, ref=ref)
            s = summarize(rows)
            s.update(nfe=nfe, eta=args.eta, sec_per_anchor=sec)
            results[f"{name}@nfe{nfe}"] = dict(summary=s, rows=rows)
            print(name, nfe, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in s.items()}), flush=True)
            json.dump(results, open(out, "w"), indent=1)       # keep partial results
    print("saved", out)


if __name__ == "__main__":
    main()
