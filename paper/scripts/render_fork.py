"""Render one fork of the final branch set: the robot is simulated up to an anchor, then
forked with new push noise, as in sim/generate2.py. Saves frames of one future that
recovers and one that falls, plus the fall fraction over the simulated branches.

Re-simulated here, so results can differ slightly from the cluster files (floating point).

    python render_fork.py --seed 971670 --anchor 256 --branches 24
"""
import argparse
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sim"))
import generate2 as G2                              # noqa: E402
from generate import SCENE, Episode, sample_config  # noqa: E402

FRAMES = (0, 20, 40, 60, 80, 100)


def tilt(qpos):
    """Body tilt from vertical [rad] for qpos rows."""
    out = []
    for q in qpos:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q[3:7])
        out.append(np.arccos(np.clip(R[8], -1, 1)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=971670)
    ap.add_argument("--anchor", type=int, default=256)
    ap.add_argument("--branches", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(ROOT, "paper/data/render_fork.npz"))
    ap.add_argument("--gif", default="", help="also write a side-by-side GIF of the two futures")
    args = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(SCENE)
    ep = Episode(m)
    cfg = sample_config(args.seed)
    anchors = G2.pick_anchors(np.random.default_rng(args.seed + 31337), 4)
    assert args.anchor in anchors, anchors
    bseed = args.seed * 100 + anchors.index(args.anchor)
    r = G2.Rollout(ep, cfg)
    while r.k < args.anchor:
        r.step()
    assert r.fell_at is None
    runs = []
    for b in range(args.branches):
        f = r.fork(push_seed=bseed * 1000 + b)
        for _ in range(args.horizon + 1):
            f.step()
        runs.append(dict(qpos=np.asarray(f.log["qpos"]), push=np.asarray(f.log["push"]),
                         fell_at=f.fell_at))
        print(f"branch {b}: fell at {f.fell_at}", flush=True)
    fell = np.array([x["fell_at"] is not None for x in runs])
    print(cfg["terrain"], cfg["push_level"], "fall fraction", fell.mean())
    # clearest examples: the recovering future with the smallest late tilt, and the falling
    # future whose fall is closest to mid-horizon
    late_tilt = [tilt(x["qpos"][-30:]).max() for x in runs]
    rec = np.nonzero(~fell)[0]
    fal = np.nonzero(fell)[0]
    t_fall = np.array([runs[b]["fell_at"] - args.anchor for b in fal])
    pick = [int(rec[np.argmin(np.array(late_tilt)[rec])]), int(fal[np.argmin(abs(t_fall - 55))])]

    rend = mujoco.Renderer(m, 480, 640)
    mujoco.mjr_uploadHField(m, rend._mjr_context, 0)
    d = mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    q0 = runs[0]["qpos"][0]
    cam.lookat[:] = [q0[0], q0[1], q0[2] - 0.05]
    cam.distance, cam.elevation, cam.azimuth = 1.7, -22, 120
    frames = np.zeros((2, len(FRAMES), 480, 640, 3), np.uint8)
    pushed = np.zeros((2, len(FRAMES)), bool)
    for row, b in enumerate(pick):
        for c, k in enumerate(FRAMES):
            d.qpos[:] = runs[b]["qpos"][k]
            mujoco.mj_forward(m, d)
            cam.lookat[:2] = d.qpos[:2]               # follow the robot, fixed height
            rend.update_scene(d, cam)
            frames[row, c] = rend.render()
            lo = 0 if c == 0 else FRAMES[c - 1] + 1
            pushed[row, c] = np.abs(runs[b]["push"][lo:k + 1, :2]).sum() > 0
    np.savez_compressed(args.out, frames=frames, pushed=pushed, frame_steps=np.array(FRAMES),
                        fall_fraction=fell.mean(), n_branches=len(runs),
                        fell_at=np.array([-1 if x["fell_at"] is None else x["fell_at"] for x in runs]),
                        picked=np.array(pick), terrain=cfg["terrain"], push_level=cfg["push_level"])
    print("saved", args.out)
    if args.gif:
        save_gif(args.gif, m, d, rend, cam, [runs[b] for b in pick])


def save_gif(path, m, d, rend, cam, pair, every=2, scale=0.5, ms=80):
    """Side-by-side animation of the two picked futures (A recovers, B falls); a red border
    marks steps with a push."""
    from PIL import Image, ImageDraw
    n = min(len(r["qpos"]) for r in pair)
    w, h = int(640 * scale), int(480 * scale)
    out, last_push = [], [-99, -99]
    for k in range(0, n, every):
        canvas = Image.new("RGB", (2 * w + 6, h + 26), (232, 238, 241))
        for i, r in enumerate(pair):
            d.qpos[:] = r["qpos"][k]
            mujoco.mj_forward(m, d)
            cam.lookat[:2] = d.qpos[:2]
            rend.update_scene(d, cam)
            img = Image.fromarray(rend.render()).resize((w, h), Image.LANCZOS)
            if np.abs(r["push"][max(0, k - every + 1):k + 1, :2]).sum() > 0:
                last_push[i] = k
            if k - last_push[i] < 8:
                ImageDraw.Draw(img).rectangle([0, 0, w - 1, h - 1], outline=(212, 71, 60), width=4)
            canvas.paste(img, (i * (w + 6), 26))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 7), "future A: recovers", fill=(21, 33, 47))
        draw.text((w + 12, 7), "future B: falls", fill=(21, 33, 47))
        draw.text((2 * w - 70, 7), f"t = {k * 0.02:.2f} s", fill=(88, 103, 121))
        out.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=128))
    out[0].save(path, save_all=True, append_images=out[1:], duration=ms, loop=0, optimize=True)
    print("saved", path)


if __name__ == "__main__":
    main()
