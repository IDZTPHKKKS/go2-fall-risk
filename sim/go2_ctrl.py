"""Model-based locomotion controller for the Unitree Go2 in MuJoCo.

Two rates, like the real robot's low-level interface:
- high level (50 Hz): gait scheduler, stance-force balance (body wrench
  distributed over stance feet by least squares, friction-cone clipped),
  Raibert swing-foot placement + analytic leg IK. Output is a Unitree-LowCmd
  style command per joint: q_des, dq_des, kp, kd, tau_ff.
- low level (500 Hz, every physics step): tau = kp (q_des - q) + kd (dq_des - dq) + tau_ff,
  clipped to the actuator limits.
"""
import numpy as np
import mujoco

LEGS = ("FL", "FR", "RL", "RR")
SIDE = np.array([1, -1, 1, -1])           # +1 left legs
L1, L2, L3 = 0.0955, 0.213, 0.213          # abduction offset, thigh, calf
HIP = np.array([[0.1934, 0.0465, 0], [0.1934, -0.0465, 0],
                [-0.1934, 0.0465, 0], [-0.1934, -0.0465, 0]])
TAU_MAX = np.array([23.7, 23.7, 45.43] * 4)

# phase offsets (FL, FR, RL, RR), duty factor, period [s], swing height [m]
GAITS = {
    "stand": dict(offset=(0, 0, 0, 0), duty=1.0, period=0.5, height=0.0),
    "walk":  dict(offset=(0, 0.5, 0.75, 0.25), duty=0.75, period=0.6, height=0.07),
    "trot":  dict(offset=(0, 0.5, 0.5, 0), duty=0.6, period=0.4, height=0.08),
    "pace":  dict(offset=(0, 0.5, 0, 0.5), duty=0.6, period=0.3, height=0.07),
    "bound": dict(offset=(0, 0, 0.5, 0.5), duty=0.6, period=0.36, height=0.08),
}


def quat_to_rot(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def rpy(R):
    return np.array([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)),
                     np.arctan2(R[1, 0], R[0, 0])])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def leg_ik(p, side):
    """Foot position p relative to the hip (abduction) joint, in the body
    frame -> (q_hip, q_thigh, q_calf). Knee-backward branch (q_calf < 0)."""
    x, y, z = p
    d = side * L1
    L = np.sqrt(max(y * y + z * z - d * d, 1e-6))
    q1 = np.arctan2(z, y) - np.arctan2(-L, d)
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    D2 = x * x + L * L
    c3 = np.clip((D2 - L2 * L2 - L3 * L3) / (2 * L2 * L3), -1, 1)
    q3 = -np.arccos(c3)
    q2 = np.arctan2(-x, L) - np.arctan2(L3 * np.sin(q3), L2 + L3 * np.cos(q3))
    return np.array([q1, q2, q3])


class Go2Controller:
    def __init__(self, model, data, ctrl_dt=0.02, height=0.30, rng=None):
        self.m, self.d = model, data
        self.dt = ctrl_dt
        self.h0 = height
        self.rng = rng or np.random.default_rng(0)
        self.mass = model.body_subtreemass[1]
        self.I = np.diag(model.body_inertia[1]) * 1.6      # legs add inertia
        self.foot_gid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in LEGS]
        self.calf_bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n + "_calf")
                         for n in LEGS]
        self.gait_name = "stand"
        self.gait = GAITS["stand"]
        self.phase = 0.0
        self.cmd = np.zeros(3)                  # vx, vy (yaw frame), wz
        self.liftoff = np.zeros((4, 3))
        self.prev_stance = np.ones(4, bool)
        self.kp_swing, self.kd = 40.0, 1.5
        self.kp_stance = 0.0
        self.k_raibert = 0.15                   # capture-point style foot placement gain
        self.lowcmd = None
        self.xy_des = None
        self.i_err = np.zeros(3)                # integral of (z, roll, pitch) error
        self.yaw_des = None
        self.ground = lambda x, y: 0.0          # terrain height lookup (sim/terrain.Ground)
        self.clearance = 0.0                    # extra swing height, e.g. for stairs [m]
        self.z_ground = None                    # filtered ground height under the stance feet

    # ---------------------------------------------------------------- state
    def state(self):
        d = self.d
        R = quat_to_rot(d.qpos[3:7])
        feet = np.array([d.geom_xpos[g] for g in self.foot_gid])
        return dict(p=d.qpos[:3].copy(), R=R, rpy=rpy(R), v=d.qvel[:3].copy(),
                    w_body=d.qvel[3:6].copy(), q=d.qpos[7:].copy(), dq=d.qvel[6:].copy(),
                    feet=feet)

    def set_gait(self, name):
        if name != self.gait_name:
            self.gait_name, self.gait = name, GAITS[name]

    def leg_phase(self):
        return (self.phase + np.array(self.gait["offset"])) % 1.0

    def stance_mask(self):
        if self.gait["duty"] >= 1.0:
            return np.ones(4, bool)
        return self.leg_phase() < self.gait["duty"]

    # ---------------------------------------------------------------- high level
    def step(self):
        """Advance the gait clock and compute the LowCmd for the next 20 ms."""
        s = self.state()
        R, yaw = s["R"], s["rpy"][2]
        Ryaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        self.phase = (self.phase + self.dt / self.gait["period"]) % 1.0
        stance = self.stance_mask()
        lph = self.leg_phase()
        for i in range(4):
            if self.prev_stance[i] and not stance[i]:
                self.liftoff[i] = s["feet"][i]
        self.prev_stance = stance

        v_cmd = Ryaw @ np.array([self.cmd[0], self.cmd[1], 0.0])
        w_cmd = np.array([0, 0, self.cmd[2]])
        w_world = R @ s["w_body"]
        # desired pose integrated from the command; errors clamped so a push or
        # stumble does not cause a violent catch-up
        if self.xy_des is None:
            self.xy_des, self.yaw_des = s["p"][:2].copy(), yaw
        self.xy_des = self.xy_des + v_cmd[:2] * self.dt
        self.yaw_des = self.yaw_des + self.cmd[2] * self.dt
        e_xy = self.xy_des - s["p"][:2]
        n = np.linalg.norm(e_xy)
        if n > 0.15:
            self.xy_des = s["p"][:2] + e_xy * 0.15 / n
            e_xy *= 0.15 / n
        e_yaw = (self.yaw_des - yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(e_yaw) > 0.3:
            self.yaw_des = yaw + np.sign(e_yaw) * 0.3
            e_yaw = np.sign(e_yaw) * 0.3

        # body height is held relative to the ground under the stance feet
        z_st = s["feet"][stance, 2].mean() - 0.022 if stance.any() else None
        if self.z_ground is None:
            self.z_ground = s["feet"][:, 2].mean() - 0.022
        elif z_st is not None:
            self.z_ground += 0.1 * (z_st - self.z_ground)
        h_err = self.z_ground + self.h0 - s["p"][2]

        # --- stance: desired body wrench -> foot forces
        g = 9.81
        com = self.d.subtree_com[1]
        acc = np.array([30.0 * e_xy[0] + 12.0 * (v_cmd[0] - s["v"][0]),
                        30.0 * e_xy[1] + 12.0 * (v_cmd[1] - s["v"][1]),
                        150.0 * h_err + 18.0 * (0 - s["v"][2]) + g])
        r, p_ = s["rpy"][0], s["rpy"][1]
        # integral action absorbs unmodeled swing-leg reaction moments
        self.i_err = np.clip(self.i_err + np.array([h_err, -r, -p_]) * self.dt,
                             [-0.05, -0.15, -0.15], [0.05, 0.15, 0.15])
        acc[2] += 300.0 * self.i_err[0]
        # roll/pitch act about the heading (yaw) frame axes -> rotate to world
        alpha = Ryaw @ np.array([500.0 * -r + 300.0 * self.i_err[1],
                                 500.0 * -p_ + 1500.0 * self.i_err[2], 0.0]) \
            + np.array([0.0, 0.0, 300.0 * e_yaw]) + 40.0 * (w_cmd - w_world)
        Iw = R @ self.I @ R.T
        b = np.concatenate([self.mass * acc, Iw @ alpha])
        f = np.zeros((4, 3))
        idx = np.nonzero(stance)[0]
        if len(idx):
            A = np.zeros((6, 3 * len(idx)))
            for j, i in enumerate(idx):
                A[:3, 3 * j:3 * j + 3] = np.eye(3)
                A[3:, 3 * j:3 * j + 3] = skew(s["feet"][i] - com)
            W = np.diag([1, 1, 1, 4, 4, 2.0])
            reg = 1e-3 * np.eye(3 * len(idx))
            x = np.linalg.solve(A.T @ W @ A + reg, A.T @ W @ b)
            for j, i in enumerate(idx):
                fi = x[3 * j:3 * j + 3]
                fi[2] = np.clip(fi[2], 0.0, 180.0)
                fxy = np.linalg.norm(fi[:2])
                if fxy > 0.6 * fi[2]:
                    fi[:2] *= 0.6 * fi[2] / (fxy + 1e-9)
                f[i] = fi

        q_des = s["q"].copy()
        dq_des = np.zeros(12)
        tau_ff = np.zeros(12)
        kp = np.full(12, self.kp_stance)
        jacp = np.zeros((3, self.m.nv))
        T_st = self.gait["duty"] * self.gait["period"]
        T_sw = (1 - self.gait["duty"]) * self.gait["period"]
        for i in range(4):
            sl = slice(3 * i, 3 * i + 3)
            mujoco.mj_jac(self.m, self.d, jacp, None, s["feet"][i], self.calf_bid[i])
            J = jacp[:, 6 + 3 * i: 9 + 3 * i]
            if stance[i]:
                tau_ff[sl] = -J.T @ f[i]
                # joint velocity that keeps the foot planted under the commanded body motion
                v_foot_rel = -(v_cmd + np.cross(w_cmd, s["feet"][i] - com))
                dq_des[sl] = np.linalg.lstsq(J, v_foot_rel, rcond=None)[0]
                continue
            # --- swing: Raibert placement + cycloid height profile
            sw = (lph[i] - self.gait["duty"]) / max(1 - self.gait["duty"], 1e-6)
            sw_next = min(sw + self.dt / max(T_sw, 1e-6), 1.0)
            hip_w = s["p"] + Ryaw @ HIP[i]
            t_left = (1 - sw) * T_sw
            hip_td = hip_w + v_cmd * t_left
            # yaw contribution and Raibert feedback
            hip_td += np.cross(w_cmd, Ryaw @ HIP[i]) * (t_left + T_st / 2)
            target = hip_td + v_cmd * T_st / 2 + self.k_raibert * (s["v"] - v_cmd)
            target += Ryaw @ np.array([0, SIDE[i] * L1, 0])
            target[2] = self.ground(target[0], target[1]) + 0.022
            z0 = self.liftoff[i][2]
            qs = []
            for u in (sw_next, min(sw_next + self.dt / max(T_sw, 1e-6), 1.0)):
                a = 0.5 - 0.5 * np.cos(np.pi * u)
                pos = (1 - a) * self.liftoff[i] + a * target
                # rise early when stepping up, descend late when stepping down
                uz = min(2 * u, 1.0) if target[2] > z0 else max(2 * u - 1, 0.0)
                az = 0.5 - 0.5 * np.cos(np.pi * uz)
                pos[2] = (1 - az) * z0 + az * target[2] \
                    + (self.gait["height"] + self.clearance) * np.sin(np.pi * u)
                qs.append(leg_ik(R.T @ (pos - s["p"]) - HIP[i], SIDE[i]))
            q_des[sl] = qs[0]
            dq_des[sl] = (qs[1] - qs[0]) / self.dt
            kp[sl] = self.kp_swing
            tau_ff[sl] = 0.0
        self.lowcmd = dict(q_des=q_des, dq_des=dq_des, kp=kp, kd=np.full(12, self.kd), tau_ff=tau_ff,
                           stance=stance, grf_des=f)
        return self.lowcmd

    # ---------------------------------------------------------------- low level
    def apply(self):
        c = self.lowcmd
        q, dq = self.d.qpos[7:], self.d.qvel[6:]
        tau = c["kp"] * (c["q_des"] - q) + c["kd"] * (c["dq_des"] - dq) + c["tau_ff"]
        self.d.ctrl[:] = np.clip(tau, -TAU_MAX, TAU_MAX)
        return self.d.ctrl.copy()
