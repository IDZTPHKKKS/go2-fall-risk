"""Episode generator v2 for the Go2 MuJoCo dataset, with branching ground truth.

Changes from v1 (generate.py):
- The logged state is observed before this step's push is applied (v1 logged it after),
  so a window's first state never contains the push that starts its future.
- Pushes follow a Poisson process (constant hazard after the first second), so the
  distribution of future pushes does not depend on the hidden time since the last one.
- No pace on uneven terrain and walking yaw rate within +/-0.5 rad/s (the v1 pilot's
  unprovoked falls came mostly from these).
- Commands and pushes use separate random streams, so a fork can keep the commands
  and redraw only the pushes.
- Branching: at chosen anchor steps the full simulator + controller state is copied and
  rolled out N times with fresh push draws. The spread of those futures is the true
  conditional distribution a world model should learn.
- Schema 2.1: logging continues post_fall steps after a fall (default 100; schema 2.0
  and v1 used 50). It must be at least the world-model horizon H: a training window
  from step t needs states up to t + H, so with a shorter tail every window that starts
  within H - post_fall steps before a fall is missing, and no training window ever shows
  an early fall (a survivorship bias in the learned fall probability).

Usage:
    python generate2.py --start 0 --count 100 --out ../data/v2/train_000.h5
    python generate2.py --start 900000 --count 10 --out ../data/v2/branch_000.h5 \
        --anchors 4 --branches 128 --horizon 50
"""
import argparse
import copy
import os
import sys
import time

import h5py
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import go2_ctrl as G
import terrain as T
from generate import (CTRL_DT, DOCS, EP_LEN, GAIT_IDS, PUSH_LEVELS, SCAN, SCENE,
                      SUBSTEPS, Episode, sample_config, write_episode)
from generate import sample_segment as sample_segment_v1

SCHEMA_VERSION = "2.1"
POST_FALL = 100                   # steps logged after a fall; must be >= the model horizon
PUSH_START = 1.0                  # [s] no pushes while standing still at the start
PUSH_RATE = 0.4                   # pushes per second (mean interval 2.5 s, as in v1)
LOG_KEYS = ("qpos", "qvel", "state", "q_des", "dq_des", "kp", "kd", "tau_ff", "tau", "cmd",
            "gait", "phase", "stance_sched", "contact", "contact_force", "feet", "push",
            "scan", "ground_z")


def sample_segment(rng, kind):
    """v1 command sampling minus the combinations behind most unprovoked falls in the
    v1 pilot: pace on uneven ground, and fast turning while walking sideways."""
    gait, cmd, dur = sample_segment_v1(rng, kind)
    if gait == "pace" and kind != "flat":
        gait = "trot"
    if gait == "walk":
        cmd[2] = np.clip(cmd[2], -0.5, 0.5)
    return gait, cmd, dur


class Rollout:
    """One simulated robot that can be stepped and forked."""

    def __init__(self, ep, cfg, post_fall=POST_FALL):
        self.ep, self.cfg, m = ep, cfg, ep.m
        self.post_fall = post_fall
        z, self.tparams = T.make(cfg["terrain"], np.random.default_rng(cfg["seed"]))
        T.apply(m, z)
        self.ground = T.Ground(z)
        ep.randomize(cfg)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mju_axisAngle2Quat(d.qpos[3:7], np.array([0, 0, 1.0]), cfg["heading"])
        mujoco.mj_forward(m, d)
        c = G.Go2Controller(m, d, ctrl_dt=CTRL_DT)
        c.ground = self.ground
        if cfg["terrain"] == "stairs":
            c.clearance = abs(self.tparams["step_h"]) + 0.03
        elif cfg["terrain"] == "rough":
            c.clearance = 0.7 * self.tparams["amp"]
        self.d, self.c = d, c
        self.rng_cmd = np.random.default_rng(cfg["seed"] + 7919)
        self.rng_push = np.random.default_rng(cfg["seed"] + 104729)
        self.push_max = PUSH_LEVELS[cfg["push_level"]]
        self.k, self.seg_end, self.gait, self.cmd = 0, 0.0, "stand", np.zeros(3)
        self.fell_at, self.truncated = None, False
        self.log = {key: [] for key in LOG_KEYS}

    def fork(self, push_seed):
        """Exact copy of the simulator and controller state; same future commands,
        new future pushes, empty log."""
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        memo = {id(self.ep.m): self.ep.m, id(self.d): new.d, id(self.ground): self.ground}
        new.c = copy.deepcopy(self.c, memo)
        new.rng_cmd = copy.deepcopy(self.rng_cmd)
        new.rng_push = np.random.default_rng(push_seed)
        new.cmd = self.cmd.copy()
        new.log = {key: [] for key in LOG_KEYS}
        return new

    def step(self):
        """One 50 Hz control step. Returns False when the episode should stop."""
        m, d, c, ep, log = self.ep.m, self.d, self.c, self.ep, self.log
        t = self.k * CTRL_DT
        if t >= self.seg_end - 1e-9:
            if self.k == 0:                       # first second: stand still
                self.gait, self.cmd, dur = "stand", np.zeros(3), 1.0
            else:
                self.gait, self.cmd, dur = sample_segment(self.rng_cmd, self.cfg["terrain"])
            c.set_gait(self.gait)
            c.cmd = self.cmd.copy()
            self.seg_end = t + dur
        qpos, qvel = d.qpos.copy(), d.qvel.copy()
        s = c.state()
        R = s["R"]
        gz = self.ground(*s["p"][:2])
        yaw = s["rpy"][2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        scan_w = s["p"][:2] + SCAN @ np.array([[cy, sy], [-sy, cy]])
        scan = np.array([self.ground(x, y) for x, y in scan_w]) - s["p"][2]
        feat = np.concatenate([[s["p"][2] - gz], R.T @ np.array([0, 0, -1.0]), R.T @ s["v"],
                               s["w_body"], s["q"], s["dq"],
                               (R.T @ (s["feet"] - s["p"]).T).T.ravel()])

        push = np.zeros(6)
        u = self.rng_push.random(4)               # always 4 draws per step
        if (self.push_max > 0 and t >= PUSH_START and self.fell_at is None
                and u[0] < PUSH_RATE * CTRL_DT):
            ang = (2 * u[1] - 1) * np.pi
            mag = (0.2 + 0.8 * u[2]) * self.push_max
            push[:2] = mag * np.array([np.cos(ang), np.sin(ang)])
            push[5] = (2 * u[3] - 1) * self.push_max / 1.8
            d.qvel[:2] += push[:2]
            d.qvel[5] += push[5]

        lc = c.step()
        tau = np.zeros(12)
        for _ in range(SUBSTEPS):
            tau += c.apply()
            mujoco.mj_step(m, d)
        flag, force = ep.contacts(d)

        log["qpos"].append(qpos)
        log["qvel"].append(qvel)
        log["state"].append(feat)
        for key in ("q_des", "dq_des", "kp", "kd", "tau_ff"):
            log[key].append(lc[key])
        log["tau"].append(tau / SUBSTEPS * m.actuator_gainprm[:, 0])
        log["cmd"].append(self.cmd)
        log["gait"].append(GAIT_IDS[self.gait])
        log["phase"].append(c.phase)
        log["stance_sched"].append(lc["stance"])
        log["contact"].append(flag)
        log["contact_force"].append(force)
        log["feet"].append(s["feet"].ravel())
        log["push"].append(push)
        log["scan"].append(scan)
        log["ground_z"].append(gz)

        self.k += 1
        s2 = c.state()
        h = s2["p"][2] - self.ground(*s2["p"][:2])
        if self.fell_at is None and (abs(s2["rpy"][0]) > 1.0 or abs(s2["rpy"][1]) > 1.0
                                     or h < 0.12):
            self.fell_at = self.k
        if self.fell_at is not None and self.k - self.fell_at >= self.post_fall:
            return False
        if np.linalg.norm(s2["p"][:2]) > 7.0:
            self.truncated = True
            return False
        return self.k < EP_LEN

    def output(self):
        out = {key: np.asarray(v) for key, v in self.log.items()}
        info = dict(fell=self.fell_at is not None,
                    fall_step=-1 if self.fell_at is None else self.fell_at,
                    truncated=self.truncated, terrain_params=self.tparams)
        return out, info


def pick_anchors(rng, n):
    """Anchor steps: after the stand phase, leaving room for the horizon."""
    lo = int((PUSH_START + 0.5) / CTRL_DT)
    return sorted(rng.choice(np.arange(lo, EP_LEN - 120), n, replace=False).tolist())


def branch(main, n, horizon, seed):
    """n futures of horizon steps from main's current state. Logs horizon+1 states
    (the first equals the anchor state). Branches keep running after a fall."""
    S, P, fell, fstep = [], [], [], []
    for b in range(n):
        r = main.fork(push_seed=seed * 1000 + b)
        r.fell_at = None
        for _ in range(horizon + 1):
            r.step()
        S.append(np.asarray(r.log["state"]))
        P.append(np.asarray(r.log["push"]))
        fell.append(r.fell_at is not None)
        fstep.append(-1 if r.fell_at is None else r.fell_at)
        if b == 0:                                # identical in every branch
            plan = dict(cmd=np.asarray(r.log["cmd"], np.float32),
                        gait=np.asarray(r.log["gait"], np.int8),
                        phase=np.asarray(r.log["phase"], np.float32),
                        scan=np.asarray(r.log["scan"][0], np.float16))
    return dict(state=np.stack(S).astype(np.float32), push=np.stack(P).astype(np.float32),
                fell=np.array(fell), fall_step=np.array(fstep, np.int16), **plan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", type=int, default=0, help="branch points per episode")
    ap.add_argument("--branches", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--stride", type=int, default=1, help="seed step (10 keeps every seed in "
                    "the test split, since split = seed %% 10)")
    ap.add_argument("--post-fall", type=int, default=POST_FALL,
                    help="steps logged after a fall (>= the world-model horizon)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model = mujoco.MjModel.from_xml_path(SCENE)
    ep = Episode(model)
    t0 = time.time()
    with h5py.File(args.out, "a") as f:
        if len(f.get("episodes", {})) and f.attrs.get("post_fall_steps") != args.post_fall:
            sys.exit(f"{args.out} holds episodes with a different post-fall length")
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["post_fall_steps"] = args.post_fall
        f.attrs["ctrl_hz"] = 50
        f.attrs["physics_hz"] = 500
        f.attrs["push_rate_hz"] = PUSH_RATE
        f.attrs["push_start_s"] = PUSH_START
        for k, v in DOCS.items():
            f.attrs["doc_" + k] = v
        eg = f.require_group("episodes")
        for seed in range(args.start, args.start + args.count * args.stride, args.stride):
            name = f"ep_{seed:06d}"
            if name in eg:
                continue
            cfg = sample_config(seed)
            r = Rollout(ep, cfg, args.post_fall)
            anchors = pick_anchors(np.random.default_rng(seed + 31337), args.anchors) \
                if args.anchors else []
            br = {}
            while True:
                if r.k in anchors and r.fell_at is None:
                    br[r.k] = branch(r, args.branches, args.horizon, seed * 100 + len(br))
                if not r.step():
                    break
            out, info = r.output()
            grp = eg.create_group(name)
            write_episode(grp, out, info, cfg)
            for k, b in br.items():
                bg = grp.create_group(f"branch_{k:03d}")
                bg.attrs["anchor_step"] = k
                for key, v in b.items():
                    bg.create_dataset(key, data=v, compression="gzip", compression_opts=4,
                                      shuffle=True)
            f.flush()
            fr = [f"{b['fell'].mean():.2f}" for b in br.values()]
            print(f"{name} {cfg['terrain']:6s} push={cfg['push_level']:6s} "
                  f"steps={len(out['state'])} fell={info['fell']} branch_fall={fr} "
                  f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
