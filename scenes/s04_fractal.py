"""Fractals: four affine maps grow a fern; morphing the matrices turns it into a galaxy, a maple leaf, a dragon."""
from numba import njit
from scenes.common import *
from engine import _splat_rgb


def ifs(rows):
    A = np.array([[[r[0], r[1]], [r[2], r[3]]] for r in rows], float)
    e = np.array([[r[4], r[5]] for r in rows], float)
    p = np.array([r[6] for r in rows], float)
    return A, e, p / p.sum()


FERN = ifs([(0, 0, 0, 0.16, 0, 0, 0.01), (0.85, 0.04, -0.04, 0.85, 0, 1.6, 0.85),
            (0.2, -0.26, 0.23, 0.22, 0, 1.6, 0.07), (-0.15, 0.28, 0.26, 0.24, 0, 0.44, 0.07)])
SPIRAL = ifs([(.787879, -.424242, .242424, .859848, 1.758647, 1.408065, .447826),
              (.787879, -.424242, .242424, .859848, 1.758647, 1.408065, .447826),
              (-.121212, .257576, .151515, .053030, -6.721654, 1.377236, .052174),
              (.181818, -.136364, .090909, .181818, 6.086107, 1.568035, .052174)])
MAPLE = ifs([(.14, .01, 0, .51, -.08, -1.31, .10), (.43, .52, -.45, .5, 1.49, -.75, .35),
             (.45, -.49, .47, .47, -1.62, -.74, .35), (.49, 0, 0, .51, .02, 1.62, .20)])
DRAGON = ifs([(.5, -.5, .5, .5, 0, 0, .25), (.5, -.5, .5, .5, 0, 0, .25),
              (-.5, -.5, .5, -.5, 1, 0, .25), (-.5, -.5, .5, -.5, 1, 0, .25)])

PAL_FERN = [hexc("ffd27a"), hexc("3dff9a"), hexc("34e1ff"), hexc("b6ff3d")]
PAL_SPIRAL = [hexc("9b6bff"), hexc("34e1ff"), hexc("ff3d8b"), hexc("ffc23d")]
PAL_MAPLE = [hexc("ffd27a"), hexc("ff5a3d"), hexc("ffa53d"), hexc("ff3d6e")]
PAL_DRAGON = [hexc("34e1ff"), hexc("3d7bff"), hexc("ff3d8b"), hexc("9b6bff")]


@njit(cache=True)
def chaos(A, e, cp, n, seed, out, idx):
    np.random.seed(seed)
    x = 0.0
    y = 0.0
    for i in range(n + 30):
        r = np.random.random()
        k = 0
        while k < 3 and r > cp[k]:
            k += 1
        nx = A[k, 0, 0] * x + A[k, 0, 1] * y + e[k, 0]
        ny = A[k, 1, 0] * x + A[k, 1, 1] * y + e[k, 1]
        x = nx
        y = ny
        if i >= 30:
            out[i - 30, 0] = x
            out[i - 30, 1] = y
            idx[i - 30] = k


def normalise(sys_, box=(9.5, 7.0)):
    A, e, p = sys_
    n = 200000
    out = np.zeros((n, 2))
    idx = np.zeros(n, np.int64)
    chaos(A, e, np.cumsum(p), n, 1, out, idx)
    lo, hi = np.percentile(out, 0.2, 0), np.percentile(out, 99.8, 0)
    c = (lo + hi) / 2
    s = min(box[0] / (hi - lo)[0], box[1] / (hi - lo)[1])
    e2 = np.array([s * (A[k] @ c + e[k] - c) for k in range(4)])
    return A.copy(), e2, p


NMAX = 2_400_000


class S(Scene):
    DUR = 16.0

    def setup(self):
        self.keys = [normalise(FERN, (9.0, 6.5)), normalise(SPIRAL, (9.0, 6.6)), normalise(MAPLE, (8.0, 6.8)),
                     normalise(DRAGON, (9.0, 5.6))]
        self.pals = [PAL_FERN, PAL_SPIRAL, PAL_MAPLE, PAL_DRAGON]
        self.out = np.zeros((NMAX, 2))
        self.idx = np.zeros(NMAX, np.int64)
        self.acc = np.zeros((H, W, 3), np.float32)

    def params(self, t):
        # morph schedule: (start, end) of each transition
        sched = [(8.6, 9.7), (11.3, 12.4), (13.9, 15.0)]
        k, x = 0, 0.0
        for i, (a, b) in enumerate(sched):
            if t >= a:
                k, x = i, ease_io(seg(t, a, b))
        A0, e0, p0 = self.keys[k]
        if t < sched[0][0]:
            A, e, p = A0.copy(), e0.copy(), p0.copy()
            pal = np.array(self.pals[0])
        else:
            A1, e1, p1 = self.keys[k + 1]
            A, e, p = lerp(A0, A1, x), lerp(e0, e1, x), lerp(p0, p1, x)
            pal = lerp(np.array(self.pals[k]), np.array(self.pals[k + 1]), x)
        # the wind: sway the main map
        sway = (0.025 * math.sin(t * 1.6) + 0.012 * math.sin(t * 3.7)) * (1 - seg(t, 8.6, 9.6))
        A[1] = rot2(sway) @ A[1]
        return A, e, p / p.sum(), pal

    def draw(self, t, fi):
        cam = Cam(zoom=lerp(0.95, 1.08, ease_io(seg(t, 0, 8.6))), x=-1.25, y=-0.3)
        fr = Frame(cam)
        A, e, p, pal = self.params(t)
        g = seg(t, 0.4, 7.0)
        n = int(1500 * (NMAX / 1500) ** (g ** 1.6)) if t < 8.6 else NMAX // 2
        chaos(A, e, np.cumsum(p), n, 7, self.out, self.idx)
        pts = self.out[:n]
        cols = np.array(pal, np.float32)[self.idx[:n]]
        self.acc[:] = 0
        px = cam.to_px(pts)
        _splat_rgb(self.acc, np.ascontiguousarray(px[:, 0]), np.ascontiguousarray(px[:, 1]), cols, 1.0 / (K * K))
        L = self.acc.sum(-1) + 1e-6
        lg = np.log1p(L)
        sub = lg[::3, ::3]
        nz = sub[sub > 0.05]
        q = float(np.percentile(nz, 99.3)) if nz.size > 100 else 1.0
        den = max(0.75, q * 0.85)
        v = lg / den
        vis = 1 - ease_in(seg(t, 15.3, 16.0)) * 0.3
        fr.img += self.acc / L[..., None] * (np.minimum(v, 1.6) * 1.3 * vis)[..., None]

        # chaos game "cursor": the last few jumps while sparse
        ca = 1 - seg(n, 2500, 60000)
        if ca > 0 and n > 20:
            trail = pts[n - 12:n]
            fr.polylines([trail], WHITE, 1.2, 0.35 * ca)
            fr.glow(pts[n - 1], 12, WHITE, 1.3 * ca)

        # matrices of the four maps
        pa = fade(t, 0.8, 15.6, 0.6, 0.4)
        if pa > 0:
            fr.fill(np.array([[-6.95, -1.5], [-3.1, -1.5], [-3.1, 3.85], [-6.95, 3.85]]), np.zeros(3), 0.55 * pa,
                    screen=True, mode="over")
            for k in range(4):
                y = 3.2 - k * 1.3
                c = np.asarray(pal[k]) * 1.2
                fr.text(f"f{'₁₂₃₄'[k]}", (-6.55, y), 0.26, c, pa, "math")
                fr.matrix(A[k], (-5.25, y), 0.2, [c, c], pa, fmt="{:.2f}")
                fr.text("+", (-4.3, y), 0.24, WHITE, pa * 0.8, "num")
                fr.matrix([[e[k][0]], [e[k][1]]], (-3.75, y), 0.2, [c], pa, fmt="{:.2f}")
        fr.slogan(t, 0.9, 4.9, "四个仿射变换", "随机迭代数百万次，长出一片蕨叶")
        fr.slogan(t, 5.3, 8.7, "分形", "同一条规则，在每一个尺度上重复", size=0.8)
        fr.slogan(t, 9.3, 15.5, "改变几个数字", "蕨叶化作星系、枫叶，与龙")
        flash = flash_in(t) + 1.4 * ease_in(seg(t, 15.45, 16.0))
        return fr.finish(bloom=0.9, flash=flash, fi=fi)
