import numpy as np
from engine import *


def grid_segments(M, n=12, step=1.0, extent=None, offset=(0, 0)):
    """Grid lines under linear map M (2x2). Returns (minor_segs, axis_segs). Lines are subdivided so
    non-linear camera effects stay smooth."""
    ext = n * step if extent is None else extent
    ks = np.arange(-n, n + 1) * step
    segs = []
    axes = []
    for k in ks:
        a = np.array([[k, -ext], [k, ext]])
        b = np.array([[-ext, k], [ext, k]])
        for s in (a, b):
            s2 = s @ np.asarray(M).T + offset
            (axes if abs(k) < 1e-9 else segs).append(s2)
    return np.array(segs), np.array(axes)


def dust(fr, t, n=500, seed=3, alpha=1.0, spread=(9, 5.2), color=None, drift=0.08):
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(-1, 1, (n, 2)) * spread
    v = rng.normal(0, drift, (n, 2))
    ph = rng.uniform(0, 6.28, n)
    br = rng.uniform(0.15, 0.7, n) * (0.6 + 0.4 * np.sin(t * 1.3 + ph)) * alpha
    p = p0 + v * t
    p[:, 0] = (p[:, 0] + spread[0]) % (2 * spread[0]) - spread[0]
    p[:, 1] = (p[:, 1] + spread[1]) % (2 * spread[1]) - spread[1]
    base = np.array([0.55, 0.7, 1.0], np.float32) if color is None else color
    cols = (br[:, None] * base[None]).astype(np.float32)
    fr.splat(p, None, 1.0, screen=True, cols=cols)


def column_vec(fr, vals, pos, size=0.3, color=WHITE, alpha=1.0, fmt="{:g}"):
    fr.matrix([[v] for v in vals], pos, size, [color], alpha, fmt=fmt)


def flash_in(t, dur=0.45, amt=1.2):
    return amt * (1 - seg(t, 0, dur)) ** 2


def chord_wave(x, t):
    """C-major triad (4:5:6) as a travelling wave, shared by the SVD dissolve and the Fourier scene."""
    return 0.55 * (np.sin(2.2 * x - 5.0 * t) + np.sin(2.75 * x - 6.25 * t + 0.7) + np.sin(3.3 * x - 7.5 * t + 1.9))
