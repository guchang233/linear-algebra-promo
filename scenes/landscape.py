"""Procedural 'photo': sunset over layered mountains and a lake."""
import numpy as np
import cv2


def fbm1d(n, seed, octaves=8, rough=0.52, base=3):
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    amp = 1.0
    for o in range(octaves):
        k = base * 2 ** o
        pts = rng.uniform(-1, 1, k + 1)
        out += amp * np.interp(np.linspace(0, k, n), np.arange(k + 1), pts)
        amp *= rough
    return out


def fbm2d(h, w, seed, octaves=6, sx=1, sy=1):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    for o in range(octaves):
        gh, gw = max(2, int(2 ** (o + 1) * sy)), max(2, int(2 ** (o + 1) * sx))
        out += amp * cv2.resize(rng.standard_normal((gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        amp *= 0.5
    return out


def make_landscape(w=960, h=540):
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    y = Y / h
    x = X / w
    stops = [(0.0, (0.03, 0.04, 0.16)), (0.30, (0.18, 0.09, 0.33)), (0.47, (0.62, 0.22, 0.36)),
             (0.56, (0.98, 0.45, 0.30)), (0.62, (1.0, 0.72, 0.42)), (1.0, (1.0, 0.8, 0.5))]
    img = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        img[..., c] = np.interp(y, [s[0] for s in stops], [s[1][c] for s in stops])
    sx, sy = 0.63, 0.555
    d = np.sqrt(((x - sx) * w / h) ** 2 + (y - sy) ** 2)
    img += np.array([1.0, 0.55, 0.25]) * (np.exp(-d * 9) * 0.55)[..., None]
    img += np.array([1.0, 0.8, 0.5]) * (np.exp(-d * 30) * 0.5)[..., None]
    sun = np.clip((0.045 - d) * h / 1.5, 0, 1)
    img = img * (1 - sun[..., None]) + np.array([1.0, 0.93, 0.75]) * sun[..., None]
    # stars
    rng = np.random.default_rng(4)
    n = 260
    ps = rng.uniform(0, 1, (n, 2)) * [w, h * 0.35]
    for (px, py), b in zip(ps.astype(int), rng.uniform(0.2, 1, n)):
        img[py, px] += b * (1 - py / (h * 0.35)) * 0.8
    # clouds
    cn = fbm2d(h, w, 9, 6, sx=1.2, sy=5)
    band = np.exp(-((y - 0.34) / 0.13) ** 2)
    cm = np.clip(cn * 0.8 - 0.1, 0, 1) ** 1.3 * band
    lit = np.exp(-np.abs(x - sx) * 2.2) * np.clip(1 - (0.5 - y) * 1.5, 0, 1)
    ccol = np.array([0.35, 0.14, 0.32])[None, None] * (1 - lit[..., None]) + np.array([1.0, 0.55, 0.45])[None, None] * lit[..., None]
    img = img * (1 - cm[..., None] * 0.85) + ccol * cm[..., None] * 0.85
    # mountain layers (far -> near)
    layers = [(0.585, 0.16, (0.48, 0.24, 0.40), 21), (0.64, 0.15, (0.30, 0.13, 0.28), 22),
              (0.71, 0.11, (0.14, 0.06, 0.17), 23), (0.765, 0.04, (0.05, 0.025, 0.07), 24)]
    shore = 0.8
    for i, (base, amp, col, seed) in enumerate(layers):
        ridge = base - amp * np.clip(1 - np.abs(fbm1d(w, seed, rough=0.5)) * 1.1, 0, 1) ** 1.6
        if i == 3:  # pine trees on the nearest ridge
            tr = np.random.default_rng(5)
            for cx in tr.uniform(0, w, 160):
                th, tw = tr.uniform(0.012, 0.035), tr.uniform(2.5, 5)
                xs = np.arange(max(0, int(cx - tw)), min(w, int(cx + tw) + 1))
                ridge[xs] = np.minimum(ridge[xs], ridge[int(cx) % w] - th * (1 - np.abs(xs - cx) / tw))
        mask = (y >= ridge[None, :]) & (y < shore + 0.002)
        mist = np.clip((y - ridge[None, :]) / 0.12, 0, 1)
        colv = np.array(col, np.float32)
        lay = colv[None, None] * (1 + 0.35 * (1 - mist[..., None]) * (i < 3)) + np.array([0.9, 0.5, 0.4]) * (mist * 0.12 * (i < 3))[..., None]
        edge = np.clip((y - ridge[None, :]) * h, 0, 1)
        m = (mask * edge)[..., None]
        img = img * (1 - m) + lay * m
    # lake: mirrored reflection with ripples
    r0 = int(shore * h)
    rows = np.arange(r0, h)
    src = np.clip(2 * r0 - rows, 0, h - 1)
    rip = fbm2d(h - r0, w, 13, 5, sx=0.6, sy=6) * 2
    refl = np.zeros((h - r0, w, 3), np.float32)
    for k, r in enumerate(src):
        shift = rip[k].astype(int)
        cols = np.clip(np.arange(w) + shift, 0, w - 1)
        refl[k] = img[r, cols]
    fadef = np.linspace(0.75, 0.45, h - r0)[:, None, None]
    img[r0:] = refl * fadef + np.array([0.02, 0.02, 0.06])
    img[r0:r0 + 2] *= 0.6
    # grade
    img = np.clip(img, 0, 1) ** 1.05
    return img.astype(np.float32)


if __name__ == "__main__":
    im = make_landscape()
    cv2.imencode(".png", cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))[1].tofile("out/stills/landscape.png")
