"""Procedural terrain for the 16 x 16 m heightfield in scene_terrain.xml.

All terrains are centred on the start position (0, 0), with a flat start pad,
so the robot walks onto the feature whatever direction it is commanded.
Heights are in metres relative to z = 0 and must stay within +/- 0.3 m.
"""
import numpy as np
import mujoco

N = 401                 # cells per side
HALF = 8.0              # half-size [m]
ELEV, OFFSET = 0.6, 0.3  # hfield elevation range and geom z offset
CELL = 2 * HALF / (N - 1)
_xs = np.linspace(-HALF, HALF, N)
X, Y = np.meshgrid(_xs, _xs)   # row index = y, column index = x (MuJoCo convention)

KINDS = ("flat", "rough", "slope", "stairs")


def _smooth_noise(rng, corr, amp):
    """Gaussian-correlated noise (FFT filter), zero mean, peak-to-peak ~ amp."""
    w = rng.standard_normal((N, N))
    k = np.fft.fftfreq(N, d=CELL)
    KX, KY = np.meshgrid(k, k)
    filt = np.exp(-0.5 * (KX ** 2 + KY ** 2) * (2 * np.pi * corr) ** 2)
    z = np.real(np.fft.ifft2(np.fft.fft2(w) * filt))
    z -= z.mean()
    return z * amp / (np.percentile(z, 99) - np.percentile(z, 1) + 1e-12)


def _pad(z, radius=0.6):
    """Blend to a flat start pad around the origin."""
    r = np.maximum(np.abs(X), np.abs(Y))
    w = np.clip((r - radius) / 0.4, 0, 1)
    return z * w


def make(kind, rng):
    """Return (heights (N, N) in metres, params dict)."""
    p = dict(kind=kind)
    if kind == "flat":
        z = np.zeros((N, N))
    elif kind == "rough":
        p["amp"] = float(rng.uniform(0.02, 0.06))
        p["corr"] = float(rng.uniform(0.08, 0.25))
        z = _pad(_smooth_noise(rng, p["corr"], p["amp"]))
    elif kind == "slope":
        p["deg"] = float(rng.uniform(5, 15)) * rng.choice([-1, 1])   # + uphill away from start
        r = np.maximum(np.abs(X), np.abs(Y))
        z = np.tan(np.radians(p["deg"])) * np.clip(r - 0.8, 0, None)
        z += _smooth_noise(rng, 0.15, 0.01)
    elif kind == "stairs":
        p["step_h"] = float(rng.uniform(0.04, 0.10)) * rng.choice([-1, 1])  # + up, - down
        p["step_w"] = float(rng.uniform(0.28, 0.40))
        r = np.maximum(np.abs(X), np.abs(Y))
        z = p["step_h"] * np.floor(np.clip(r - 0.8, 0, None) / p["step_w"] + 1) * (r > 0.8)
    else:
        raise ValueError(kind)
    return np.clip(z, -OFFSET + 0.005, ELEV - OFFSET - 0.005), p


def apply(model, z):
    """Write heights into the model's heightfield (physics uses it directly)."""
    model.hfield_data[:] = ((z + OFFSET) / ELEV).ravel().astype(np.float32)


class Ground:
    """Bilinear lookup of terrain height, for swing-foot targets."""

    def __init__(self, z):
        self.z = z

    def __call__(self, x, y):
        fx = np.clip((x + HALF) / CELL, 0, N - 1.001)
        fy = np.clip((y + HALF) / CELL, 0, N - 1.001)
        i, j = int(fy), int(fx)
        a, b = fy - i, fx - j
        z = self.z
        return float((1 - a) * ((1 - b) * z[i, j] + b * z[i, j + 1])
                     + a * ((1 - b) * z[i + 1, j] + b * z[i + 1, j + 1]))


def heights(z, x, y):
    """Vectorized Ground lookup: bilinear terrain height at arrays x, y [m]."""
    fx = np.clip((np.asarray(x) + HALF) / CELL, 0, N - 1.001)
    fy = np.clip((np.asarray(y) + HALF) / CELL, 0, N - 1.001)
    i, j = fy.astype(int), fx.astype(int)
    a, b = fy - i, fx - j
    return ((1 - a) * ((1 - b) * z[i, j] + b * z[i, j + 1])
            + a * ((1 - b) * z[i + 1, j] + b * z[i + 1, j + 1]))
