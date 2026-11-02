"""Render dataset episodes to MP4 by replaying the logged qpos (no re-simulation)."""
import argparse
import os
import sys

import h5py
import imageio
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrain as T
from generate import SCENE

GAITS = ["stand", "walk", "trot", "bound", "pace"]


def render(h5, names, out, fps=25):
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 360, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance, cam.elevation, cam.azimuth = 2.2, -20, 135
    frames = []
    with h5py.File(h5) as f:
        for name in names:
            g = f["episodes"][name]
            z, _ = T.make(g.attrs["terrain"], np.random.default_rng(int(g.attrs["seed"])))
            T.apply(m, z)
            mujoco.mjr_uploadHField(m, r._mjr_context, 0)
            qpos, gait, push = g["qpos"][()], g["gait"][()], g["push"][()]
            for k in range(0, len(qpos), 2):
                d.qpos[:] = qpos[k]
                mujoco.mj_forward(m, d)
                r.update_scene(d, cam)
                if np.abs(push[max(0, k - 6):k + 1, :2]).sum() > 0:   # mark pushes with a red tint
                    img = r.render().copy()
                    img[:12] = [220, 40, 40]
                else:
                    img = r.render()
                frames.append(img)
            print(name, g.attrs["terrain"], g.attrs["push_level"], "fell" if g.attrs["fell"] else "ok",
                  "gaits:", sorted({GAITS[i] for i in gait}))
    imageio.mimsave(out, frames, fps=fps, quality=7, macro_block_size=8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--out", default="../media/episodes.mp4")
    a = ap.parse_args()
    render(a.h5, a.episodes, a.out)
