"""Data for the paper figures that needs the cluster (branch files, training data, models).

Writes paper/data/fig_data.npz:
- names, p_true, fall_idx: per anchor of the final branch set, the first fall index of each
  of its 128 true futures (-1: no fall);
- ex_<i>_<src>_tilt / _height: tilt and height traces of the true futures and of 128
  samples from v3_cfm_s0, v3_diff_s0, v3_ncm4_s0 at the example anchors;
- win_<v2|v3>: histogram over training windows (start state not fallen) of the first fall
  index in the window, counted over all valid window starts, as the models saw them.

    python fig_data.py --examples ep_971670/branch_256 ep_970380/branch_212
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "models"))
from data import BranchSet, TerrainCache          # noqa: E402
from evaluate import load                          # noqa: E402

H = 100


def fall_flags(x):
    """(..., 46) raw states -> (...,) bool, the test of data.fall_label per step."""
    tilt = np.arccos(np.clip(-x[..., 3], -1, 1))
    return (x[..., 0] < 0.12) | (tilt > 1.0)


def first_fall(flags):
    return np.where(flags.any(-1), flags.argmax(-1), -1)


def window_hist(pattern):
    """Over all windows the Corpus would make (start s in [0, n - H), future rows s+1..s+H)
    whose start state has not fallen: counts of the first fall index (0..H-1), and of
    windows without a fall."""
    counts, no_fall = np.zeros(H, np.int64), 0
    for path in sorted(glob.glob(pattern)):
        with h5py.File(path, "r") as f:
            for g in f["episodes"].values():
                if g.attrs["split"] != "train":
                    continue
                s = g["state"][()]
                n = len(s)
                if n <= H:
                    continue
                fl = fall_flags(s)
                starts = np.arange(n - H)
                starts = starts[~fl[starts]]
                if not fl.any():
                    no_fall += len(starts)
                    continue
                # first fall row at or after each start + 1
                nxt = np.full(n + 1, n, np.int64)
                for r in range(n - 1, -1, -1):
                    nxt[r] = r if fl[r] else nxt[r + 1]
                j = nxt[starts + 1] - (starts + 1)
                inside = j < H
                counts += np.bincount(j[inside], minlength=H)
                no_fall += int((~inside).sum())
    return counts, no_fall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", default=os.path.join(ROOT, "data/final/branch_*.h5"))
    ap.add_argument("--terrain-cache", default=os.path.join(ROOT, "data/v2/terrain_cache.h5"))
    ap.add_argument("--examples", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["v3_cfm_s0", "v3_diff_s0", "v3_ncm4_s0"])
    ap.add_argument("--nfe", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(ROOT, "paper/data/fig_data.npz"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}

    for v in ("v2", "v3"):
        c, nf = window_hist(os.path.join(ROOT, f"data/{v}/train_*.h5"))
        out[f"win_{v}"], out[f"win_{v}_nofall"] = c, nf
        print(v, "windows with a fall:", c.sum(), "first fall index min:",
              int(np.nonzero(c)[0].min()), "without:", nf, flush=True)

    models = {m: load(m, dev) for m in args.models}
    stats = models[args.models[0]][1]
    terrain = TerrainCache(args.terrain_cache)
    bs = BranchSet(args.branches, H, stats, terrain)
    mean, std = stats["s_mean"], stats["s_std"]
    names = [it["name"] for it in bs.items]
    fi = []
    for it in bs.items:
        fi.append(first_fall(fall_flags(it["futures"] * std + mean)))
    fi = np.stack(fi).astype(np.int16)
    out.update(names=np.array(names), fall_idx=fi, p_true=(fi >= 0).mean(1))

    tr = lambda x: (np.arccos(np.clip(-x[..., 3], -1, 1)).astype(np.float16),
                    x[..., 0].astype(np.float16))
    for e, name in enumerate(args.examples):
        i = names.index(name)
        raw = bs.items[i]["futures"] * std + mean
        out[f"ex_{e}_true_tilt"], out[f"ex_{e}_true_height"] = tr(raw)
        out[f"ex_{e}_name"] = np.array(name)
        for m, (model, st) in models.items():
            torch.manual_seed(0)
            b = {k: v.to(dev) for k, v in bs.inputs(i, raw.shape[0]).items()}
            with torch.no_grad():
                y = model.sample(b, nfe=args.nfe).cpu().numpy()
            y = y * st["s_std"] + st["s_mean"]
            key = m.split("_")[1]
            out[f"ex_{e}_{key}_tilt"], out[f"ex_{e}_{key}_height"] = tr(y)
            print(name, m, "p_true", (first_fall(fall_flags(raw)) >= 0).mean(),
                  "p_samp", (first_fall(fall_flags(y)) >= 0).mean(), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
