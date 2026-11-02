"""Episode generator for the Go2 MuJoCo dataset.

One episode = up to 10 s at 50 Hz on a procedural terrain, with randomized
physics, piecewise-constant velocity/gait commands and random base pushes.
After a fall, logging continues for 1 s and the episode ends.

Usage (low CPU, efficiency cores):
    taskpolicy -b python3 generate.py --start 0 --count 100 --out ../data/shards/shard_000.h5
"""
import argparse
import os
import sys
import time

import h5py
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import go2_ctrl as G
import terrain as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "assets/menagerie/unitree_go2/scene_terrain.xml")
CTRL_DT, SUBSTEPS = 0.02, 10
EP_LEN = 500                                  # 10 s at 50 Hz
POST_FALL = 50                                # keep logging 1 s after a fall
GAIT_IDS = {"stand": 0, "walk": 1, "trot": 2, "bound": 3, "pace": 4}
GAIT_P = {"stand": 0.10, "walk": 0.20, "trot": 0.35, "bound": 0.15, "pace": 0.20}
VX = {"stand": (0, 0), "walk": (-0.3, 0.5), "trot": (-0.5, 1.0), "bound": (0.0, 0.8),
      "pace": (-0.3, 0.7)}
PUSH_LEVELS = {"none": 0.0, "light": 0.6, "medium": 1.2, "strong": 1.8}   # max |dv_xy| [m/s]
SCAN = np.stack(np.meshgrid(np.arange(-3, 4) * 0.15, np.arange(-3, 4) * 0.15), -1).reshape(-1, 2)
SCHEMA_VERSION = "1.0"


def sample_config(seed):
    rng = np.random.default_rng(seed)
    kind = T.KINDS[rng.integers(len(T.KINDS))]
    push = list(PUSH_LEVELS)[rng.integers(len(PUSH_LEVELS))]
    return dict(seed=int(seed), terrain=kind, push_level=push,
                friction=float(rng.uniform(0.4, 1.2)),
                payload=float(rng.uniform(0.0, 3.0)),
                payload_pos=rng.uniform([-0.08, -0.04, 0.0], [0.08, 0.04, 0.06]).round(4).tolist(),
                motor_strength=float(rng.uniform(0.9, 1.1)),
                damping_scale=float(rng.uniform(0.7, 1.3)),
                heading=float(rng.uniform(-np.pi, np.pi)))


def sample_segment(rng, kind):
    gaits = list(GAIT_P)
    gait = gaits[rng.choice(len(gaits), p=[GAIT_P[g] for g in gaits])]
    lo, hi = VX[gait]
    vx = rng.uniform(lo, hi)
    vy = 0.0 if gait in ("stand", "bound") else rng.uniform(-0.3, 0.3)
    wz = 0.0 if gait == "stand" else rng.uniform(-0.8, 0.8) * (0.5 if gait == "bound" else 1)
    if rng.random() < 0.3:
        vy = wz = 0.0
    if kind in ("slope", "stairs"):
        vx = float(np.clip(vx, -0.4, 0.6))
    return gait, np.array([vx, vy, wz]), float(rng.uniform(1.5, 4.0))


class Episode:
    def __init__(self, model):
        self.m = model
        self.defaults = dict(friction=model.geom_friction.copy(), mass=model.body_mass.copy(),
                             ipos=model.body_ipos.copy(), damping=model.dof_damping.copy(),
                             gain=model.actuator_gainprm.copy())
        self.foot_gid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in G.LEGS]
        self.floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def randomize(self, cfg):
        m, d0 = self.m, self.defaults
        m.geom_friction[:] = d0["friction"]
        for g in self.foot_gid:
            m.geom_friction[g, 0] = cfg["friction"]
        m.body_mass[:] = d0["mass"]
        m.body_ipos[:] = d0["ipos"]
        m.body_mass[1] += cfg["payload"]
        # a point payload at payload_pos (trunk frame, relative to the trunk CoM)
        # shifts the trunk CoM by its mass share
        share = cfg["payload"] / m.body_mass[1]
        m.body_ipos[1] = d0["ipos"][1] + share * np.array(cfg["payload_pos"])
        m.dof_damping[:] = d0["damping"]
        m.dof_damping[6:] *= cfg["damping_scale"]
        m.actuator_gainprm[:] = d0["gain"]
        m.actuator_gainprm[:, 0] *= cfg["motor_strength"]

    def contacts(self, d):
        flag, force = np.zeros(4, bool), np.zeros(4)
        f6 = np.zeros(6)
        for k in range(d.ncon):
            c = d.contact[k]
            for i, g in enumerate(self.foot_gid):
                if (c.geom1 == g and c.geom2 == self.floor_gid) or \
                        (c.geom2 == g and c.geom1 == self.floor_gid):
                    mujoco.mj_contactForce(self.m, d, k, f6)
                    flag[i] = True
                    force[i] += f6[0]
        return flag, force

    def run(self, cfg):
        m = self.m
        rng = np.random.default_rng(cfg["seed"] + 7919)
        z, tparams = T.make(cfg["terrain"], np.random.default_rng(cfg["seed"]))
        T.apply(m, z)
        ground = T.Ground(z)
        self.randomize(cfg)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mju_axisAngle2Quat(d.qpos[3:7], np.array([0, 0, 1.0]), cfg["heading"])
        mujoco.mj_forward(m, d)
        c = G.Go2Controller(m, d, ctrl_dt=CTRL_DT)
        c.ground = ground
        if cfg["terrain"] == "stairs":
            c.clearance = abs(tparams["step_h"]) + 0.03
        elif cfg["terrain"] == "rough":
            c.clearance = 0.7 * tparams["amp"]

        push_max = PUSH_LEVELS[cfg["push_level"]]
        next_push = rng.uniform(1.5, 3.5) if push_max > 0 else np.inf
        seg_end, gait, cmd = 0.0, "stand", np.zeros(3)
        log = {k: [] for k in ("qpos", "qvel", "state", "q_des", "dq_des", "kp", "kd", "tau_ff",
                               "tau", "cmd", "gait", "phase", "stance_sched", "contact",
                               "contact_force", "feet", "push", "scan", "ground_z")}
        fell_at, truncated = None, False
        for k in range(EP_LEN):
            t = k * CTRL_DT
            if t >= seg_end:                  # first second: stand still
                if t == 0.0:
                    gait, cmd, dur = "stand", np.zeros(3), 1.0
                else:
                    gait, cmd, dur = sample_segment(rng, cfg["terrain"])
                c.set_gait(gait)
                c.cmd = cmd.copy()
                seg_end = t + dur
            push = np.zeros(6)
            if t >= next_push and fell_at is None:
                ang = rng.uniform(-np.pi, np.pi)
                mag = rng.uniform(0.2, 1.0) * push_max
                push[:2] = mag * np.array([np.cos(ang), np.sin(ang)])
                push[5] = rng.uniform(-1.0, 1.0) * push_max / 1.8
                d.qvel[:2] += push[:2]
                d.qvel[5] += push[5]
                next_push = t + rng.uniform(1.5, 3.5)

            qpos, qvel = d.qpos.copy(), d.qvel.copy()
            s = c.state()
            R = s["R"]
            gz = ground(*s["p"][:2])
            yaw = s["rpy"][2]
            cy, sy = np.cos(yaw), np.sin(yaw)
            scan_w = s["p"][:2] + SCAN @ np.array([[cy, sy], [-sy, cy]])
            scan = np.array([ground(x, y) for x, y in scan_w]) - s["p"][2]
            feat = np.concatenate([[s["p"][2] - gz], R.T @ np.array([0, 0, -1.0]), R.T @ s["v"],
                                   s["w_body"], s["q"], s["dq"], (R.T @ (s["feet"] - s["p"]).T).T.ravel()])

            lc = c.step()
            tau = np.zeros(12)
            for _ in range(SUBSTEPS):
                tau += c.apply()
                mujoco.mj_step(m, d)
            flag, force = self.contacts(d)

            log["qpos"].append(qpos)
            log["qvel"].append(qvel)
            log["state"].append(feat)
            for key in ("q_des", "dq_des", "kp", "kd", "tau_ff"):
                log[key].append(lc[key])
            log["tau"].append(tau / SUBSTEPS * m.actuator_gainprm[:, 0])
            log["cmd"].append(cmd)
            log["gait"].append(GAIT_IDS[gait])
            log["phase"].append(c.phase)
            log["stance_sched"].append(lc["stance"])
            log["contact"].append(flag)
            log["contact_force"].append(force)
            log["feet"].append(s["feet"].ravel())
            log["push"].append(push)
            log["scan"].append(scan)
            log["ground_z"].append(gz)

            s2 = c.state()
            h = s2["p"][2] - ground(*s2["p"][:2])
            if fell_at is None and (abs(s2["rpy"][0]) > 1.0 or abs(s2["rpy"][1]) > 1.0 or h < 0.12):
                fell_at = k + 1
            if fell_at is not None and k + 1 - fell_at >= POST_FALL:
                break
            if np.linalg.norm(s2["p"][:2]) > 7.0:
                truncated = True
                break
        out = {key: np.asarray(v) for key, v in log.items()}
        return out, dict(fell=fell_at is not None, fall_step=-1 if fell_at is None else fell_at,
                         truncated=truncated, terrain_params=tparams)


DOCS = {
    "qpos": "MuJoCo qpos (19): base pos (3), base quat wxyz (4), joint angles (12), at the start of the step",
    "qvel": "MuJoCo qvel (18): base lin vel world (3), base ang vel body (3), joint vel (12)",
    "state": "46-d features: height above terrain (1), gravity in body frame (3), body lin vel (3), "
             "body ang vel (3), joint pos (12), joint vel (12), feet pos in body frame (12)",
    "q_des": "LowCmd target joint angles (12), held for the step",
    "dq_des": "LowCmd target joint velocities (12)",
    "kp": "LowCmd stiffness per joint (12): 40 in swing, 0 in stance",
    "kd": "LowCmd damping per joint (12)",
    "tau_ff": "LowCmd feedforward torque (12) [Nm]",
    "tau": "applied joint torque averaged over the 10 physics substeps (12) [Nm]",
    "cmd": "commanded vx, vy (heading frame) [m/s], yaw rate [rad/s]",
    "gait": "gait id: 0 stand, 1 walk, 2 trot, 3 bound, 4 pace",
    "phase": "gait clock in [0, 1)",
    "stance_sched": "scheduled stance per leg (FL, FR, RL, RR)",
    "contact": "actual foot-ground contact per leg at the end of the step",
    "contact_force": "normal contact force per foot [N] at the end of the step",
    "feet": "foot positions in world frame (4 x 3)",
    "push": "base velocity change applied at this step: dvx, dvy, dvz, dwx, dwy, dwz (world)",
    "scan": "terrain height minus base height on a 7 x 7 grid (0.15 m spacing) in the heading frame",
    "ground_z": "terrain height under the base [m]",
}
FLOAT16 = {"scan"}
BOOL = {"stance_sched", "contact"}
INT = {"gait"}


def write_episode(grp, out, info, cfg):
    for key, v in out.items():
        dt = np.float16 if key in FLOAT16 else bool if key in BOOL else np.int8 if key in INT \
            else np.float32
        grp.create_dataset(key, data=v.astype(dt), compression="gzip", compression_opts=4,
                           shuffle=True)
    for k, v in cfg.items():
        grp.attrs[k] = v
    grp.attrs["fell"] = info["fell"]
    grp.attrs["fall_step"] = info["fall_step"]
    grp.attrs["truncated"] = info["truncated"]
    grp.attrs["n_steps"] = len(out["state"])
    for k, v in info["terrain_params"].items():
        grp.attrs["terrain_" + k] = v
    grp.attrs["split"] = "test" if cfg["seed"] % 10 == 0 else "val" if cfg["seed"] % 10 == 1 \
        else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model = mujoco.MjModel.from_xml_path(SCENE)
    ep = Episode(model)
    t0 = time.time()
    with h5py.File(args.out, "a") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["ctrl_hz"] = 50
        f.attrs["physics_hz"] = 500
        for k, v in DOCS.items():
            f.attrs["doc_" + k] = v
        eg = f.require_group("episodes")
        for seed in range(args.start, args.start + args.count):
            name = f"ep_{seed:06d}"
            if name in eg:
                continue
            cfg = sample_config(seed)
            out, info = ep.run(cfg)
            write_episode(eg.create_group(name), out, info, cfg)
            f.flush()
            print(f"{name} {cfg['terrain']:6s} push={cfg['push_level']:6s} steps={len(out['state'])} "
                  f"fell={info['fell']} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
