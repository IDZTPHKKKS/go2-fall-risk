"""Windows from the Go2 MuJoCo dataset for world-model training.

A window at step t:
    prefix : state s_t (46), the previous HIST states (incl. s_t), height scan at t (49),
             a coarse 4 m x 4 m terrain map around the robot (MAP_DIM; covers everywhere
             the robot can reach within the horizon, so unseen terrain is not a source of
             uncertainty), known physics/disturbance parameters (PAR_DIM)
    plan   : per-step command features for t .. t+H-1 (cmd 3, gait one-hot 5, phase sin/cos 2)
    target : states s_{t+1} .. s_{t+H} (H x 46)
The world model is a planner-level model: it knows the commands that will be sent and
the push intensity of the episode, but not when or where the pushes will land, so the
future is genuinely uncertain. In v2 data that uncertainty is exactly the spread of the
branch rollouts (see BranchSet).
"""
import glob
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))
import terrain as T  # noqa: E402

EP_LEN = 500   # episode length limit [steps] (sim/generate.py)

STATE_DIM, SCAN_DIM, PLAN_DIM = 46, 49, 10
HIST = 25                                     # 0.5 s of past states
PUSH_LEVELS = ("none", "light", "medium", "strong")
PAR_DIM = 7 + len(PUSH_LEVELS)
MAP_N, MAP_STEP = 17, 0.25                    # 17 x 17 grid, +/- 2 m, heading frame
MAP_DIM = MAP_N * MAP_N
_G = (np.arange(MAP_N) - MAP_N // 2) * MAP_STEP
MAP_GRID = np.stack(np.meshgrid(_G, _G), -1).reshape(-1, 2)


class TerrainCache:
    """Heightfields regenerated from the episode seeds (the generator builds terrain
    with terrain.make(kind, default_rng(seed))), cached in one HDF5 file."""

    def __init__(self, path):
        self.path, self.z = path, {}

    def build(self, patterns):
        need = {}
        for pat in patterns:
            for p in sorted(glob.glob(pat)):
                with h5py.File(p, "r") as f:
                    for g in f["episodes"].values():
                        need[int(g.attrs["seed"])] = str(g.attrs["terrain"])
        with h5py.File(self.path, "a") as f:
            todo = [s for s in need if str(s) not in f]
            for s in todo:
                z, _ = T.make(need[s], np.random.default_rng(s))
                f.create_dataset(str(s), data=z.astype(np.float16), compression="gzip")
        return len(need), len(todo)

    def get(self, seed):
        if seed not in self.z:
            with h5py.File(self.path, "r") as f:
                self.z[seed] = f[str(seed)][()]          # float16, 0.3 MB per terrain
        return self.z[seed]


def terrain_map(z, qpos):
    """(n, 19) qpos rows -> (n, MAP_DIM) terrain height minus base height on MAP_GRID,
    in the heading frame (same convention as the generator's 7 x 7 scan)."""
    q = qpos[:, 3:7]
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                     1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    gx, gy = MAP_GRID[None, :, 0], MAP_GRID[None, :, 1]
    x = qpos[:, :1] + c * gx - s * gy
    y = qpos[:, 1:2] + s * gx + c * gy
    return (T.heights(z, x, y) - qpos[:, 2:3]).astype(np.float32)


def plan_features(cmd, gait, phase):
    oh = np.eye(5, dtype=np.float32)[gait]
    ph = np.stack([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], -1)
    return np.concatenate([cmd, oh, ph], -1).astype(np.float32)


def param_features(a):
    """Episode attributes -> (PAR_DIM,): friction, payload, payload_pos (3), motor strength,
    damping scale, push level one-hot. Roughly scaled to unit range."""
    pl = str(a["push_level"])
    return np.concatenate([[a["friction"], a["payload"] / 3.0],
                           np.asarray(a["payload_pos"]) * 10.0,
                           [(a["motor_strength"] - 1.0) * 10.0, a["damping_scale"]],
                           np.eye(len(PUSH_LEVELS))[PUSH_LEVELS.index(pl)]]).astype(np.float32)


def history_rows(rows, ep_start):
    """(n,) window rows -> (n, HIST) rows of the past states, clamped at episode start."""
    h = rows[:, None] - np.arange(HIST - 1, -1, -1)[None]
    return np.maximum(h, ep_start[:, None])


class Corpus:
    """All episodes of the given split(s), concatenated in RAM."""

    def __init__(self, pattern, split, horizon, stats=None, terrain=None, norm="zscore"):
        self.H = horizon
        self.terrain = terrain
        S, C, P, R, E, Q, I, starts, meta = [], [], [], [], [], [], [], [], []
        off, short_tail = 0, []
        for path in sorted(glob.glob(pattern)):
            with h5py.File(path, "r") as f:
                for name, g in f["episodes"].items():
                    if split != "all" and g.attrs["split"] != split:
                        continue
                    s = g["state"][()]
                    n = len(s)
                    if g.attrs["fell"] and n - int(g.attrs["fall_step"]) < horizon:
                        short_tail.append((name, n))
                    if n <= horizon:
                        continue
                    S.append(s)
                    C.append(g["scan"][()].astype(np.float32))
                    P.append(plan_features(g["cmd"][()], g["gait"][()], g["phase"][()]))
                    R.append(np.repeat(param_features(g.attrs)[None], n, 0))
                    E.append(np.full(n, off))
                    Q.append(g["qpos"][()].astype(np.float32))
                    I.append(np.full(n, len(meta)))
                    # valid window starts: need H future states
                    starts.append(off + np.arange(n - horizon))
                    meta.append(dict(name=name, offset=off, n=n, fell=bool(g.attrs["fell"]),
                                     seed=int(g.attrs["seed"]),
                                     fall_step=int(g.attrs["fall_step"]),
                                     terrain=str(g.attrs["terrain"]),
                                     push=g["push"][()][:, :2].any(1)))
                    off += n
        # Episodes that stopped fewer than H steps after a fall (other than at the episode
        # length limit) lose every window that starts shortly before the fall, so no window
        # would show an early fall: a survivorship bias in the learned fall probability.
        short_tail = [name for name, n in short_tail if n < EP_LEN]
        if short_tail:
            raise ValueError(f"{len(short_tail)} episodes (e.g. {short_tail[0]}) end fewer than "
                             f"{horizon} steps after a fall; regenerate with "
                             f"generate2.py --post-fall >= {horizon}")
        self.state = np.concatenate(S).astype(np.float32)
        self.scan = np.concatenate(C)
        self.plan = np.concatenate(P)
        self.par = np.concatenate(R)
        self.ep_start = np.concatenate(E)
        self.qpos = np.concatenate(Q)
        self.ep_id = np.concatenate(I)
        self.starts = np.concatenate(starts)
        self.meta = meta
        if stats is None:
            s_std = self.state.std(0) + 1e-4
            if norm == "robust":
                # outlier-friendly scale: rare extreme states (falls) land within ~+/-3
                q = np.percentile(self.state, [0.1, 99.9], axis=0)
                s_std = np.maximum(s_std, (q[1] - q[0]) / 6.0)
            stats = dict(s_mean=self.state.mean(0), s_std=s_std,
                         c_mean=self.scan.mean(0), c_std=self.scan.std(0) + 1e-4)
        self.stats = {k: np.asarray(v, np.float32) for k, v in stats.items()}
        if "m_mean" not in self.stats:               # terrain-map scale from a row sample
            rows = np.random.default_rng(0).choice(len(self.state), min(20000, len(self.state)),
                                                   replace=False)
            m = self.maps(rows, raw=True)
            self.stats["m_mean"] = np.float32(m.mean())
            self.stats["m_std"] = np.float32(m.std() + 1e-4)
        self.state_n = self.norm_state(self.state)
        self.scan_n = self.norm_scan(self.scan)

    def maps(self, idx, raw=False):
        out = np.zeros((len(idx), MAP_DIM), np.float32)
        if self.terrain is not None:
            for e in np.unique(self.ep_id[idx]):
                sel = self.ep_id[idx] == e
                out[sel] = terrain_map(self.terrain.get(self.meta[e]["seed"]),
                                       self.qpos[idx[sel]])
        return out if raw else (out - self.stats["m_mean"]) / self.stats["m_std"]

    def norm_state(self, s):
        return (s - self.stats["s_mean"]) / self.stats["s_std"]

    def norm_scan(self, c):
        return (c - self.stats["c_mean"]) / self.stats["c_std"]

    def batch(self, idx):
        """idx: array of window start rows -> dict of tensors."""
        H = self.H
        steps = idx[:, None] + np.arange(H)[None]
        return dict(s0=torch.from_numpy(self.state_n[idx]),
                    hist=torch.from_numpy(self.state_n[history_rows(idx, self.ep_start[idx])]),
                    scan=torch.from_numpy(self.scan_n[idx]),
                    tmap=torch.from_numpy(self.maps(idx)),
                    par=torch.from_numpy(self.par[idx]),
                    plan=torch.from_numpy(self.plan[steps]),
                    x=torch.from_numpy(self.state_n[steps + 1]))

    def sample(self, rng, n):
        return self.batch(self.starts[rng.integers(0, len(self.starts), n)])


class BranchSet:
    """Branching ground truth (v2 files written with --anchors): for each anchor, the
    model inputs plus N simulated futures from the identical simulator state."""

    def __init__(self, pattern, horizon, stats, terrain=None):
        self.H = horizon
        self.items = []
        st = {k: np.asarray(v, np.float32) for k, v in stats.items()}
        for path in sorted(glob.glob(pattern)):
            with h5py.File(path, "r") as f:
                for name, g in f["episodes"].items():
                    s_ep = g["state"][()]
                    q_ep = g["qpos"][()].astype(np.float32)
                    par = param_features(g.attrs)
                    z = terrain.get(int(g.attrs["seed"])) if terrain is not None else None
                    for bname in (k for k in g if k.startswith("branch_")):
                        b = g[bname]
                        k = int(b.attrs["anchor_step"])
                        hist = s_ep[np.maximum(np.arange(k - HIST + 1, k + 1), 0)]
                        bs = b["state"][()]
                        assert np.allclose(bs[0, 0], s_ep[k], atol=1e-5)
                        self.items.append(dict(
                            name=f"{name}/{bname}", terrain=str(g.attrs["terrain"]),
                            push_level=str(g.attrs["push_level"]),
                            s0=(s_ep[k] - st["s_mean"]) / st["s_std"],
                            hist=(hist - st["s_mean"]) / st["s_std"],
                            scan=(b["scan"][()].astype(np.float32) - st["c_mean"]) / st["c_std"],
                            par=par,
                            tmap=np.zeros(MAP_DIM, np.float32) if z is None else
                            (terrain_map(z, q_ep[k:k + 1])[0] - st["m_mean"]) / st["m_std"],
                            plan=plan_features(b["cmd"][:horizon], b["gait"][:horizon],
                                               b["phase"][:horizon]),
                            futures=(bs[:, 1:horizon + 1] - st["s_mean"]) / st["s_std"],
                            fell=b["fell"][()],
                            fall_step=np.where(b["fall_step"][()] >= 0, b["fall_step"][()] - k, -1)))

    def inputs(self, i, k):
        """Model inputs for anchor i, repeated k times (for k samples)."""
        it = self.items[i]
        rep = lambda a: torch.from_numpy(np.repeat(np.asarray(a, np.float32)[None], k, 0))
        return dict(s0=rep(it["s0"]), hist=rep(it["hist"]), scan=rep(it["scan"]),
                    tmap=rep(it["tmap"]), par=rep(it["par"]), plan=rep(it["plan"]))


def fall_label(x_raw):
    """(..., H, 46) raw states -> (...,) bool: the generator's fall test (|roll| or |pitch|
    > 1 rad, or height above terrain < 0.12 m), with roll/pitch approximated by the body
    tilt angle from the gravity direction in the body frame."""
    x_raw = torch.as_tensor(x_raw)
    tilt = torch.arccos((-x_raw[..., 3]).clamp(-1, 1))
    return ((x_raw[..., 0] < 0.12) | (tilt > 1.0)).any(-1)


if __name__ == "__main__":
    # build the terrain cache: python data.py CACHE.h5 PATTERN [PATTERN ...]
    print("episodes, newly built:", TerrainCache(sys.argv[1]).build(sys.argv[2:]))
