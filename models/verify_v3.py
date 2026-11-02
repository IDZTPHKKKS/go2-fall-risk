"""Checks on the v3 data before any training (run by sim/gen_v3.sbatch; exit 1 = fail).
Fatal: episode counts, NaN, and every fallen episode logged >= 100 steps after its fall
(unless it hit the 500-step limit). Informational: whether the v3 episodes reproduce the
v2 ones over their common length (same seeds and code; only the post-fall tail differs)."""
import glob
import sys

import h5py
import numpy as np

from data import TerrainCache

bad, n_ep, same, n_cmp, max_d, fall_moved = [], 0, 0, 0, 0.0, 0
for p in sorted(glob.glob("../data/v3/train_*.h5")):
    with h5py.File(p, "r") as f, h5py.File(p.replace("/v3/", "/v2/"), "r") as g:
        if f.attrs["post_fall_steps"] != 100:
            bad.append((p, "post_fall_steps"))
        for name, e in f["episodes"].items():
            n_ep += 1
            s = e["state"][()]
            n, fs = len(s), int(e.attrs["fall_step"])
            if not np.isfinite(s).all():
                bad.append((name, "non-finite state"))
            if e.attrs["fell"] and n < 500 and n - fs < 100:
                bad.append((name, f"tail {n - fs}"))
            if name in g["episodes"]:
                o = g["episodes"][name]
                m = min(n, len(o["state"]))
                d = float(np.abs(s[:m] - o["state"][:m]).max())
                n_cmp += 1
                same += d == 0.0
                max_d = max(max_d, d)
                fall_moved += int(o.attrs["fall_step"]) != fs
eps, anchors, missing = 0, 0, []
tc = TerrainCache("../data/v2/terrain_cache.h5")
for p in sorted(glob.glob("../data/final/branch_*.h5")):
    with h5py.File(p, "r") as f:
        for name, e in f["episodes"].items():
            eps += 1
            anchors += sum(k.startswith("branch_") for k in e)
            try:
                tc.get(int(e.attrs["seed"]))
            except KeyError:
                missing.append(name)
            for b in (k for k in e if k.startswith("branch_")):
                if e[b]["state"].shape[1:] != (101, 46) or not np.isfinite(e[b]["state"][()]).all():
                    bad.append((f"{name}/{b}", "branch shape/NaN"))
print(f"v3 train: {n_ep} episodes; reproduce v2 exactly: {same}/{n_cmp} "
      f"(max state diff {max_d:.3g}, fall step changed in {fall_moved})")
print(f"final branch set: {eps} episodes, {anchors} anchors, terrain missing for {len(missing)}")
if n_ep != 6016:
    bad.append(("train", f"{n_ep} episodes, expected 6016"))
if eps != 320 or missing:
    bad.append(("final", f"{eps} episodes (expected 320), {len(missing)} without terrain"))
print("FAIL", bad[:20]) if bad else print("OK")
sys.exit(1 if bad else 0)
