"""Pre-registered statistics for the v3 rerun (PREREG_v3.md).

Inputs: results/branch_eval_v3_final_*.json (final test set) and
results/branch_eval_v3_calib_*.json (calibration set = the v2 confirm set).
Per model family, the per-anchor predicted fall probability is averaged over the
available seeds; CIs bootstrap over anchors (paired differences on the same anchors).
Recalibration: isotonic map (seed-averaged prediction -> true probability) fit on the
calibration set, applied unchanged to the final set.

    python v3_stats.py            # writes results/v3_stats.json
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILIES = {"cfm": [f"v3_cfm_s{s}" for s in range(3)],
            "diff": [f"v3_diff_s{s}" for s in range(3)],
            "ncm4": [f"v3_ncm4_s{s}" for s in range(3)],
            "v2_cfm": ["v2_cfm"], "v2_diff": ["v2_diff"], "v2_ncm4": ["v2_ncm4"]}
B = 10000


def load(kind):
    runs = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", f"branch_eval_v3_{kind}_*.json"))):
        for k, v in json.load(open(p)).items():
            if k.endswith("@nfe64"):
                runs[k[:-6]] = {r["name"]: r for r in v["rows"]}
    return runs


def family(runs, names, order):
    have = [n for n in names if n in runs]
    if not have:
        return None, have
    p = np.mean([[runs[n][a]["p_samp"] for a in order] for n in have], 0)
    return p, have


def isotonic(x, y):
    """Least-squares non-decreasing fit (pool adjacent violators), linear between
    the distinct x values and clipped outside them."""
    ux, inv = np.unique(x, return_inverse=True)
    w = np.bincount(inv).astype(float)
    v = np.bincount(inv, weights=y) / w
    blocks = []                                   # [value, weight, n_points]
    for vi, wi in zip(v, w):
        blocks.append([vi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    fit = np.concatenate([[b[0]] * b[2] for b in blocks])
    return lambda q: np.interp(q, ux, fit)


def auc(score, p):
    """Probability that an anchor's fall outcome is ranked above a non-fall outcome,
    with each anchor weighted p (fall) and 1 - p (no fall); ties count 1/2."""
    us, inv = np.unique(score, return_inverse=True)
    w1, w0 = np.bincount(inv, weights=p), np.bincount(inv, weights=1 - p)
    below = np.concatenate([[0], np.cumsum(w0)[:-1]])
    return float((w1 * (below + 0.5 * w0)).sum() / (w1.sum() * w0.sum()))


def ci(stat, n, rng):
    i = rng.integers(0, n, (B, n))
    return np.percentile(stat(i), [2.5, 97.5]).tolist()


def main():
    rng = np.random.default_rng(0)
    final, calib = load("final"), load("calib")
    ref = next(iter(final.values()))
    order = sorted(ref)
    pt = np.array([ref[a]["p_true"] for a in order])
    c_ref = next(iter(calib.values()))
    c_order = sorted(c_ref)
    c_pt = np.array([c_ref[a]["p_true"] for a in c_order])
    fk = (pt > 0.05) & (pt < 0.95)
    out = dict(n_anchors=len(order), n_fork=int(fk.sum()), true_mean_p_fork=float(pt[fk].mean()),
               n_calib=len(c_order))
    P, R = {}, {}
    for g, names in FAMILIES.items():
        p, have = family(final, names, order)
        if p is None:
            continue
        P[g] = p
        x = dict(seeds=have, ratio=[float(p[fk].mean() / pt[fk].mean())],
                 brier_fork=[float(((p - pt) ** 2)[fk].mean())],
                 brier_all=float(((p - pt) ** 2).mean()), auc=auc(p, pt),
                 energy=float(np.mean([np.mean([final[n][a]["energy"] for a in order]) for n in have])),
                 energy_fork=float(np.mean([np.mean([final[n][a]["energy"] for a in np.array(order)[fk]])
                                            for n in have])))
        pf, tf = p[fk], pt[fk]
        x["ratio"].append(ci(lambda i: pf[i].mean(1) / tf[i].mean(1), len(pf), rng))
        e = (pf - tf) ** 2
        x["brier_fork"].append(ci(lambda i: e[i].mean(1), len(e), rng))
        cp, c_have = family(calib, names, c_order)
        if cp is not None:
            f = isotonic(cp, c_pt)
            R[g] = f(p)
            x["recal_seeds"] = c_have
            x["recal_ratio"] = float(R[g][fk].mean() / pt[fk].mean())
            x["recal_brier_fork"] = float(((R[g] - pt) ** 2)[fk].mean())
            x["recal_brier_all"] = float(((R[g] - pt) ** 2).mean())
        bins = {}
        for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.95), (0.95, 1.01)):
            m = (pt >= lo) & (pt < hi)
            if m.any():
                bins[f"{lo}-{min(hi, 1)}"] = dict(n=int(m.sum()), true=float(pt[m].mean()),
                                                  pred=float(p[m].mean()),
                                                  recal=float(R[g][m].mean()) if g in R else None)
        x["bins"] = bins
        out[g] = x

    def paired(a, b, scope):
        m = fk if scope == "fork" else np.ones_like(fk)
        d = ((a - pt) ** 2 - (b - pt) ** 2)[m]
        return [float(d.mean()), ci(lambda i: d[i].mean(1), len(d), rng)]

    if "ncm4" in P and "cfm" in P:
        a, c, tf = P["ncm4"][fk], P["cfm"][fk], pt[fk]
        out["ratio_ncm4_minus_cfm"] = [float(a.mean() / tf.mean() - c.mean() / tf.mean()),
                                       ci(lambda i: (a[i].mean(1) - c[i].mean(1)) / tf[i].mean(1),
                                          len(tf), rng)]
        out["brier_fork_ncm4_minus_cfm"] = paired(P["ncm4"], P["cfm"], "fork")
        out["brier_all_ncm4_minus_cfm"] = paired(P["ncm4"], P["cfm"], "all")
        if "cfm" in R:
            out["brier_fork_ncm4_minus_recal_cfm"] = paired(P["ncm4"], R["cfm"], "fork")
            out["brier_all_ncm4_minus_recal_cfm"] = paired(P["ncm4"], R["cfm"], "all")
        if "cfm" in R and "ncm4" in R:
            out["brier_fork_recal_ncm4_minus_recal_cfm"] = paired(R["ncm4"], R["cfm"], "fork")
    if "cfm" in out and "diff" in out:
        out["H1_cfm_diff_underestimate"] = out["cfm"]["ratio"][1][1] < 0.8 and out["diff"]["ratio"][1][1] < 0.8
    if "ratio_ncm4_minus_cfm" in out:
        out["H2_ncm4_less_biased"] = out["ratio_ncm4_minus_cfm"][1][0] > 0
        out["H3_ncm4_lower_brier"] = out["brier_fork_ncm4_minus_cfm"][1][1] < 0
    if "brier_fork_ncm4_minus_recal_cfm" in out:
        out["H4_ncm4_beats_recal_cfm"] = out["brier_fork_ncm4_minus_recal_cfm"][1][1] < 0
    complete = all(len(out.get(g, {}).get("seeds", [])) == 3 for g in ("cfm", "diff", "ncm4")) and \
        all(len(out.get(g, {}).get("recal_seeds", [])) == 3 for g in ("cfm", "diff", "ncm4"))
    out["complete"] = complete
    path = os.path.join(ROOT, "results", "v3_stats.json" if complete else "v3_stats_partial.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
