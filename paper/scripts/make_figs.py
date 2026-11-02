"""Paper figures and the main table, from the result files in paper/data.

    python make_figs.py            # writes paper/figs/*.pdf and paper/table_main.tex
"""
import glob
import json
import os
import sys

import h5py
import matplotlib
import matplotlib.ticker
import matplotlib.lines
import numpy as np

matplotlib.use("pdf")
import matplotlib.pyplot as plt                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "paper/data")
FIGS = os.path.join(ROOT, "paper/figs")
COL, FULL = 3.45, 7.16                               # IEEE column / page width [in]
DT = 0.02                                            # control step [s]
FAM = {"cfm": ("CFM", "#2a78d6", "o"), "diff": ("Diffusion", "#eb6834", "s"),
       "ncm4": ("C-NCM (ours)", "#1baf7a", "D"), "clf": ("Classifier", "#52514e", "^")}
INK, MUTED, FALL, SAFE = "#0b0b0b", "#52514e", "#d03b3b", "#7d8ca3"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- per-anchor predictions
def load_preds():
    runs = {}
    for p in sorted(glob.glob(os.path.join(DATA, "branch_eval_v3_final_*.json"))):
        for k, v in json.load(open(p)).items():
            runs[k.split("@")[0]] = {r["name"]: r for r in v["rows"]}
    ref = runs["v3_cfm_s0"]
    names = sorted(ref)
    pt = np.array([ref[n]["p_true"] for n in names])
    P = {}
    for fam in ("cfm", "diff", "ncm4"):
        rs = [r for r in runs if r.startswith(f"v3_{fam}_s")]
        assert len(rs) == 3, rs
        P[fam] = np.mean([[runs[r][n]["p_samp"] for n in names] for r in rs], 0)
        P["v2_" + fam] = np.array([runs["v2_" + fam][n]["p_samp"] for n in names])
    clf = json.load(open(os.path.join(DATA, "v3_fallclf.json")))["per_anchor"]
    P["clf"] = np.array([clf[n] for n in names])
    meta = {n: (ref[n]["terrain"], ref[n]["push_level"]) for n in names}
    return names, pt, P, meta


def boot_ratio(p, pt, mask, B=10000, seed=0):
    pf, tf = p[mask], pt[mask]
    i = np.random.default_rng(seed).integers(0, len(pf), (B, len(pf)))
    return pf.mean() / tf.mean(), np.percentile(pf[i].mean(1) / tf[i].mean(1), [2.5, 97.5])


def auc(score, p):
    us, inv = np.unique(score, return_inverse=True)
    w1, w0 = np.bincount(inv, weights=p), np.bincount(inv, weights=1 - p)
    below = np.concatenate([[0], np.cumsum(w0)[:-1]])
    return float((w1 * (below + 0.5 * w0)).sum() / (w1.sum() * w0.sum()))


# ---------------------------------------------------------------- figure 1: rendered fork
def fig_render():
    z = np.load(os.path.join(DATA, "render_fork.npz"))
    fr, pushed, steps = z["frames"], z["pushed"], z["frame_steps"]
    r0, r1, c0, c1 = 70, 430, 130, 590                 # crop around the robot
    fig, ax = plt.subplots(2, len(steps), figsize=(FULL, 2.35),
                           gridspec_kw=dict(wspace=0.03, hspace=0.06))
    label = ["recovers", "falls"]
    for r in range(2):
        for c, k in enumerate(steps):
            a = ax[r, c]
            a.imshow(fr[r, c, r0:r1, c0:c1])
            a.set_xticks([]), a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
            if pushed[r, c]:
                a.text(0.04, 0.93, "push", transform=a.transAxes, color="white", fontsize=7,
                       va="top", bbox=dict(boxstyle="round,pad=0.2", fc=FALL, ec="none"))
            if r == 0:
                a.set_title(f"$t$ = {k * DT:.1f} s", pad=2)
        ax[r, 0].set_ylabel(f"future {'AB'[r]}:\n{label[r]}", fontsize=8)
    save(fig, "fig1_fork")
    print("  render: fall fraction", float(z["fall_fraction"]), "of", int(z["n_branches"]),
          z["terrain"], z["push_level"])


# ---------------------------------------------------------------- figure 2: levels
def fig_levels():
    sys.path.insert(0, os.path.join(ROOT, "models"))
    import torch
    from c2f import down, up
    path = sorted(glob.glob(os.path.join(ROOT, "data/v2/train_*.h5")))[0]
    with h5py.File(path) as f:
        g = next(g for g in f["episodes"].values()
                 if g.attrs["fell"] and 160 < g.attrs["fall_step"] <= len(g["state"]) - 31)
        s = g["state"][()]
        k = int(g.attrs["fall_step"]) - 70
    x = torch.from_numpy(s[k + 1:k + 101].astype(np.float32))[None]   # (1, 100, 46)
    levels = [4, 8, 16, 32, 64, 100]
    recon, recs = torch.zeros_like(x), []
    for l in levels:
        recon = recon + up(down(x - recon, l), 100)
        recs.append(recon[0].numpy())
    tilt = lambda a: np.degrees(np.arccos(np.clip(-a[:, 3], -1, 1)))
    t = np.arange(1, 101) * DT
    show = [0, 1, 2, 5]
    fig, ax = plt.subplots(1, len(show), figsize=(FULL, 1.45), sharey=True,
                           gridspec_kw=dict(wspace=0.08))
    for a, i in zip(ax, show):
        a.plot(t, tilt(s[k + 1:k + 101]), color="#c3c2b7", lw=2.2, label="true")
        a.plot(t, tilt(recs[i]), color=FAM["ncm4"][1], lw=1.2, label="levels so far")
        y = tilt(recs[i])
        if levels[i] < 100:
            tc = (np.arange(levels[i]) + 0.5) * 100 / levels[i] * DT
            a.plot(tc, np.interp(tc, t, y), "o", ms=2.5, color=FAM["ncm4"][1])
        a.set_title(f"after level {i + 1} ({levels[i]} steps)", pad=2)
        a.set_xlabel("time ahead [s]")
        a.grid(True, axis="y")
    ax[0].set_ylabel("body tilt [deg]")
    ax[0].legend(loc="lower right")
    save(fig, "fig2_levels")


# ---------------------------------------------------------------- figure 3: example fork
GROUPS = [("early", "falls within 1 s", "#d03b3b"), ("late", "falls after 1 s", "#ec835a"),
          ("rec", "recovers", "#7d8ca3")]


def outcome_groups(tl, h):
    """Group futures by outcome: first fall (tilt > 1 rad or height < 0.12 m) within the
    first 50 steps (1 s), later, or never."""
    bad = (tl > np.degrees(1.0)) | (h < 0.12)
    first = np.where(bad.any(1), bad.argmax(1), -1)
    return {"early": (first >= 0) & (first < 50), "late": first >= 50, "rec": first < 0}


def fig_example(fd, e=0):
    t = np.arange(1, 101) * DT
    srcs = [("true", "Simulator"), ("cfm", "CFM"), ("diff", "Diffusion"), ("ncm4", "C-NCM (ours)")]
    fig = plt.figure(figsize=(FULL, 2.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.1, 1.0], hspace=0.62, wspace=0.12)
    shares = []
    for j, (key, title) in enumerate(srcs):
        a = fig.add_subplot(gs[0, j], sharey=fig.axes[0] if j else None)
        tl = np.degrees(fd[f"ex_{e}_{key}_tilt"].astype(np.float32))
        h = fd[f"ex_{e}_{key}_height"].astype(np.float32)
        g = outcome_groups(tl, h)
        shares.append([int(g[k].sum()) for k, _, _ in GROUPS])
        for k, label, col in GROUPS[::-1]:
            m = g[k]
            if m.sum() < 3:
                continue
            lo, med, hi = np.percentile(tl[m], [25, 50, 75], axis=0)
            a.fill_between(t, np.minimum(lo, 120), np.minimum(hi, 120), color=col, alpha=0.22, lw=0)
            a.plot(t, np.minimum(med, 120), color=col, lw=1.4)
        a.axhline(np.degrees(1.0), color=INK, lw=0.6, ls="--")
        nf = int((~g["rec"]).sum())
        a.set_title(f"{title}: {nf}/{len(tl)} fall", pad=3)
        a.set_xlabel("time ahead [s]")
        a.set_ylim(0, 120)
        a.set_xlim(0, 2)
        a.set_xticks([0, 0.5, 1, 1.5, 2])
        a.set_xticklabels(["0", "", "1", "", "2"])
        a.grid(True, axis="y")
        if j:
            plt.setp(a.get_yticklabels(), visible=False)
    ax0 = fig.axes[0]
    ax0.set_ylabel("body tilt [deg]")
    ax0.text(0.05, np.degrees(1.0) + 3, "fall threshold", fontsize=6.5, color=INK)
    handles = [matplotlib.lines.Line2D([], [], color=c, lw=1.4) for _, _, c in GROUPS]
    fig.axes[2].legend(handles, [l for _, l, _ in GROUPS], loc="upper left", fontsize=6.5,
                       handlelength=1.4, borderaxespad=0.3)
    # share of futures in each group
    bx = fig.add_subplot(gs[1, :])
    names = [tt for _, tt in srcs]
    ys = np.arange(len(names))[::-1]
    for y, row in zip(ys, shares):
        left = 0
        for (k, label, col), n in zip(GROUPS, row):
            if n:
                bx.barh(y, n, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.2)
                if n >= 7:
                    bx.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=6.5,
                            color="white" if k != "rec" else INK)
            left += n
    bx.set_yticks(ys)
    bx.set_yticklabels(names)
    bx.set_xlim(0, 128)
    bx.set_xticks([0, 32, 64, 96, 128])
    bx.set_xlabel("number of the 128 futures in each group")
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)
    save(fig, "fig3_example")
    print("  example", str(fd[f"ex_{e}_name"]), "groups (early, late, recover):", dict(zip(names, shares)))


# ---------------------------------------------------------------- figure 4: truncation
def fig_truncation(fd):
    fi = fd["fall_idx"]
    falls = fi[fi >= 0]
    H = 100
    true_d = np.bincount(falls, minlength=H)[:H] / len(falls)
    v2 = fd["win_v2"] / fd["win_v2"].sum()
    v3 = fd["win_v3"] / fd["win_v3"].sum()
    t = (np.arange(H) + 1) * DT
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    ax.axvline(1.0, color="#c3c2b7", lw=0.6)
    ax.plot(t, np.cumsum(true_d) * 100, color=INK, lw=1.1, label="true futures (test set)")
    ax.plot(t, np.cumsum(v3) * 100, color=FAM["ncm4"][1], lw=1.1, label="training windows, fixed")
    ax.plot(t, np.cumsum(v2) * 100, color=FALL, lw=1.1, label="training windows, truncated")
    ax.set_xlabel("time of the first fall inside the 2 s window [s]")
    ax.set_ylabel("falls that happened by then [%]")
    ax.set_xlim(0, 2.0), ax.set_ylim(0, 102)
    ax.grid(True)
    ax.legend(loc="upper left")
    save(fig, "fig4_truncation")
    print(f"  true falls in first 50 steps: {(falls < 50).mean():.3f}; v2 windows: "
          f"{v2[:50].sum():.4f}, v3 windows: {v3[:50].sum():.3f}")


# ---------------------------------------------------------------- figure 5: risk ratio
def fig_ratio(pt, P, st):
    fk = (pt > 0.05) & (pt < 0.95)
    rows = []
    for fam in ("cfm", "diff", "ncm4"):
        r, ci = boot_ratio(P["v2_" + fam], pt, fk)
        rows.append(("truncated", fam, r, ci))
    for fam in ("cfm", "diff", "ncm4"):
        x = st[fam]["ratio"]
        rows.append(("fixed", fam, x[0], x[1]))
    r, ci = boot_ratio(P["clf"], pt, fk)
    rows.append(("fixed", "clf", r, ci))
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    ys = [6.6, 5.6, 4.6, 2.9, 1.9, 0.9, -0.1]
    for y, (grp, fam, r, ci) in zip(ys, rows):
        name, col, mk = FAM[fam]
        ax.plot(ci, [y, y], color=col, lw=1.4)
        ax.plot(r, y, mk, color=col, ms=4.5, mec="white", mew=0.6)
        ax.text(ci[1] + 0.03, y, f"{r:.2f}", va="center", fontsize=7, color=INK)
    ax.axvline(1.0, color=INK, lw=0.7, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([FAM[f][0] for _, f, _, _ in rows])
    ax.text(0.02, 7.35, "trained on truncated data (1 seed)", fontsize=7, color=MUTED,
            transform=ax.get_yaxis_transform())
    ax.text(0.02, 3.65, "trained on fixed data (3 seeds; classifier 1)", fontsize=7, color=MUTED,
            transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1.45)
    ax.set_ylim(-0.7, 7.8)
    ax.set_xlabel("predicted / true fall probability at forks (1 = unbiased)")
    ax.grid(True, axis="x")
    save(fig, "fig5_ratio")
    return rows


# ---------------------------------------------------------------- figure 6: calibration
def fig_calibration(pt, P):
    edges = [0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.25, 0.4, 0.7, 0.95, 1.01]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    lim = (0.004, 0.996)
    ax.plot(lim, lim, color=INK, lw=0.7, ls="--")
    for fam in ("cfm", "diff", "ncm4", "clf"):
        name, col, mk = FAM[fam]
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pt >= lo) & (pt < hi)
            if m.sum() >= 5:
                xs.append(np.clip(pt[m].mean(), *lim))
                ys.append(np.clip(P[fam][m].mean(), *lim))
        ax.plot(xs, ys, "-", color=col, lw=0.9, marker=mk, ms=3.5, mec="white", mew=0.5,
                label=name)
    ax.set_xscale("logit")
    ax.set_yscale("logit")
    ax.set_xlim(*lim), ax.set_ylim(*lim)
    ticks = [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{v:g}" for v in ticks]))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("true fall probability (bin mean)")
    ax.set_ylabel("predicted fall probability (bin mean)")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.text(0.3, 0.02, "below the line = risk underestimated", fontsize=6.5, color=MUTED,
            transform=ax.transAxes)
    save(fig, "fig6_calibration")


# ---------------------------------------------------------------- figure 7: certain falls
def fig_doomed(fd, names, pt, P, meta):
    idx = {n: i for i, n in enumerate(str(x) for x in fd["names"])}
    fi = fd["fall_idx"]
    d = [i for i, n in enumerate(names) if pt[i] >= 0.95]
    tmed = np.array([np.median(fi[idx[names[i]]][fi[idx[names[i]]] >= 0]) for i in d]) * DT
    none = np.array([meta[names[i]][1] == "none" for i in d])
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for fam, off in (("cfm", -0.012), ("ncm4", 0.012)):
        name, col, mk = FAM[fam]
        p = P[fam][d]
        ax.scatter(tmed[~none] + off, p[~none], marker=mk, s=14, color=col, lw=0.4,
                   edgecolor="white", label=f"{name}, pushes")
        ax.scatter(tmed[none] + off, p[none], marker=mk, s=14, facecolor="white",
                   edgecolor=col, lw=0.8, label=f"{name}, no pushes")
    ax.axhline(0.5, color="#c3c2b7", lw=0.6)
    ax.set_xlabel("median time of the true fall [s]")
    ax.set_ylabel("predicted fall probability")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(0, 2.0)
    ax.grid(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=0.8,
              handletextpad=0.2)
    save(fig, "fig7_certain")
    print(f"  certain-fall anchors: {len(d)}, push-free {none.sum()}")


# ---------------------------------------------------------------- figure 8: sampling steps
def fig_steps():
    """Risk ratio at forks against network passes per sample (seed-0 models)."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(DATA, "branch_eval_v3_nfe_*.json"))) + \
            sorted(glob.glob(os.path.join(DATA, "branch_eval_v3_final_*.json"))):
        for k, v in json.load(open(path)).items():
            name, nfe = k.split("@nfe")
            if name in ("v3_cfm_s0", "v3_diff_s0", "v3_ncm4_s0"):
                rows[(name.split("_")[1], int(nfe))] = v["rows"]
    pt = None
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    out = {}
    for fam in ("cfm", "diff", "ncm4"):
        name, col, mk = FAM[fam]
        pts = sorted(n for f, n in rows if f == fam)
        xs, ys, lo, hi = [], [], [], []
        for n in pts:
            r = sorted(rows[(fam, n)], key=lambda a: a["name"])
            p = np.array([a["p_samp"] for a in r])
            t = np.array([a["p_true"] for a in r])
            fk = (t > 0.05) & (t < 0.95)
            rat, ci = boot_ratio(p, t, fk)
            passes = n * (6 if fam == "ncm4" else 1)
            xs.append(passes), ys.append(rat), lo.append(ci[0]), hi.append(ci[1])
            out[(fam, n)] = (passes, rat, ci)
        ax.errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], color=col, lw=1.0,
                    marker=mk, ms=3.5, mec="white", mew=0.5, capsize=2, elinewidth=0.7,
                    label=name)
    ax.axhline(1.0, color=INK, lw=0.7, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128, 256, 512])
    ax.set_xticklabels(["16", "32", "64", "128", "256", "512"])
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("network passes per sample")
    ax.set_ylabel("risk ratio at forks")
    ax.grid(True)
    ax.legend(loc="lower right")
    save(fig, "fig8_steps")
    for k, v in sorted(out.items()):
        print(f"  steps {k[0]:5s} {k[1]:4d}: passes {v[0]:4d} ratio {v[1]:.3f} "
              f"[{v[2][0]:.3f}, {v[2][1]:.3f}]")


# ---------------------------------------------------------------- table
def table(pt, P, st, rows):
    fk = (pt > 0.05) & (pt < 0.95)
    clf = dict(ratio=boot_ratio(P["clf"], pt, fk),
               bf=float(((P["clf"] - pt) ** 2)[fk].mean()),
               ba=float(((P["clf"] - pt) ** 2).mean()), auc=auc(P["clf"], pt))
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & Ratio at forks & Brier (forks) & Brier (all) & AUC & Energy \\",
             r"\midrule"]
    for fam in ("cfm", "diff", "ncm4"):
        x = st[fam]
        name = FAM[fam][0]
        lines.append(f"{name} & {x['ratio'][0]:.2f} [{x['ratio'][1][0]:.2f}, {x['ratio'][1][1]:.2f}]"
                     f" & {x['brier_fork'][0]:.4f} & {x['brier_all']:.4f} & {x['auc']:.3f}"
                     f" & {x['energy']:.3f} \\\\")
    r, ci = clf["ratio"]
    lines.append(f"Classifier & {r:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] & {clf['bf']:.4f} & "
                 f"{clf['ba']:.4f} & {clf['auc']:.3f} & -- \\\\")
    lines.append(r"\midrule")
    for fam in ("cfm", "diff", "ncm4"):
        x = st[fam]
        lines.append(f"{FAM[fam][0]} + isotonic & {x['recal_ratio']:.2f} & {x['recal_brier_fork']:.4f}"
                     f" & {x['recal_brier_all']:.4f} & -- & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(ROOT, "paper/table_main.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote table_main.tex")
    return clf


def main():
    os.makedirs(FIGS, exist_ok=True)
    names, pt, P, meta = load_preds()
    st = json.load(open(os.path.join(DATA, "v3_stats.json")))
    assert st["complete"]
    fig_render()
    fig_levels()
    rows = fig_ratio(pt, P, st)
    for g, f, r, ci in rows:
        print(f"  ratio {g:9s} {f:5s} {r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    fig_calibration(pt, P)
    fig_steps()
    clf = table(pt, P, st, rows)
    print("  classifier", clf)
    fd_path = os.path.join(DATA, "fig_data.npz")
    if os.path.exists(fd_path):
        fd = np.load(fd_path)
        assert np.allclose(fd["p_true"][np.argsort([str(x) for x in fd["names"]])], pt, atol=1e-6)
        fig_example(fd)
        fig_truncation(fd)
        fig_doomed(fd, names, pt, P, meta)
    else:
        print("no fig_data.npz yet: skipped figures 3, 4, 7")


if __name__ == "__main__":
    main()
