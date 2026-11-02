"""Pre-registered statistics for the confirmatory test (see ../PREREG.md).

Reads results/branch_eval_confirm.json (evaluate.py on the fresh branch set) and reports
H1-H3 plus the controls. Per-anchor quantities are averaged over training seeds, then
bootstrapped over fork anchors.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(ROOT, "results", "branch_eval_confirm.json")))
NFE = 64
SEEDS = {"cfm": ["v2_cfm", "v2_cfm_s1", "v2_cfm_s2"],
         "diff": ["v2_diff", "v2_diff_s1", "v2_diff_s2"],
         "ncm4": ["v2_ncm4", "v2_ncm4_s1", "v2_ncm4_s2"]}
CONTROLS = ["v2_cfm_long", "v2_ncm4_nocorr"]


def rows(name):
    return R[f"{name}@nfe{NFE}"]["rows"]


pt = np.array([r["p_true"] for r in rows("v2_cfm")])
fork = np.where((pt > 0.05) & (pt < 0.95))[0]
P = {m: np.mean([[r["p_samp"] for r in rows(n)] for n in names], 0) for m, names in SEEDS.items()}
E = {m: np.mean([[r["energy"] for r in rows(n)] for n in names], 0) for m, names in SEEDS.items()}
for c in CONTROLS:
    if f"{c}@nfe{NFE}" in R:
        P[c] = np.array([r["p_samp"] for r in rows(c)])
        E[c] = np.array([r["energy"] for r in rows(c)])
rng = np.random.default_rng(0)
B = rng.integers(0, len(fork), (10000, len(fork)))


def boot(stat):
    v = np.array([stat(fork[b]) for b in B])
    return float(np.mean(v)), [float(x) for x in np.percentile(v, [2.5, 97.5])]


out = dict(n_anchors=len(pt), n_fork=len(fork), true_mean_p_fork=float(pt[fork].mean()))
for m in P:
    out[f"ratio_{m}"] = boot(lambda i: P[m][i].mean() / pt[i].mean())
    out[f"brier_fork_{m}"] = boot(lambda i: np.mean((P[m][i] - pt[i]) ** 2))
    out[f"energy_{m}"] = float(E[m].mean())
out["ratio_diff_ncm4_minus_cfm"] = boot(lambda i: P["ncm4"][i].mean() / pt[i].mean()
                                       - P["cfm"][i].mean() / pt[i].mean())
out["brier_ncm4_minus_cfm"] = boot(lambda i: np.mean((P["ncm4"][i] - pt[i]) ** 2)
                                   - np.mean((P["cfm"][i] - pt[i]) ** 2))
out["H1_cfm_and_diff_underestimate"] = (out["ratio_cfm"][1][1] < 0.6
                                        and out["ratio_diff"][1][1] < 0.6)
out["H2_ncm4_less_biased"] = out["ratio_diff_ncm4_minus_cfm"][1][0] > 0
out["H3_ncm4_lower_brier"] = out["brier_ncm4_minus_cfm"][1][1] < 0
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(ROOT, "results", "confirm_stats.json"), "w"), indent=1)
