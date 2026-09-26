"""3D: rotation matrices spin a torus; a synthwave terrain flyover; a 4D tesseract."""
import itertools
from scenes.common import *

T_TER = 9.0     # terrain starts
T_TES = 16.0    # tesseract (hard cut on the bar)


def torus_segments(R=1.9, r=0.78, nu=64, nv=26):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    P = np.stack([(R + r * np.cos(V)) * np.cos(U), (R + r * np.cos(V)) * np.sin(U), r * np.sin(V)], -1)
    segs = []
    for i in range(nu):
        for j in range(nv):
            segs.append((P[i, j], P[(i + 1) % nu, j]))
            segs.append((P[i, j], P[i, (j + 1) % nv]))
    return np.array(segs)


def terrain_h(x, z):
    side = np.clip((np.abs(x) - 1.4) / 4.5, 0, 1) ** 1.4
    n = (np.sin(x * 0.9 + z * 0.35) * 0.5 + np.sin(x * 0.37 - z * 0.61 + 1.3) * 0.8 +
         np.sin(z * 0.23 + x * 1.7) * 0.35 + np.sin(z * 0.9 - x * 0.5) * 0.25 + 1.9) / 3.8
    return side * (0.5 + 3.4 * n)


def rot4(i, j, a):
    R = np.eye(4)
    c, s = math.cos(a), math.sin(a)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


class S(Scene):
    DUR = 24.0

    def setup(self):
        self.tor = torus_segments()
        rng = np.random.default_rng(2)
        self.stars = np.stack([rng.uniform(-FW / 2, FW / 2, 700), rng.uniform(0.4, 4.1, 700)], 1)
        self.star_b = rng.uniform(0.1, 0.9, 700)
        V4 = np.array(list(itertools.product([-1, 1], repeat=4)), float)
        self.V4 = V4
        self.E4 = np.array([(i, j) for i in range(16) for j in range(i + 1, 16) if np.sum(V4[i] != V4[j]) == 1])

    # ------------------------------------------------------------------ torus
    def torus(self, fr, t):
        R = rot3(1.0 + 0.32 * t, 0.55 * t, 0.12 * t)
        grow = expo_out(seg(t, 0, 1.1))
        away = ease_in(seg(t, 8.4, 9.6))
        if away >= 1:
            return
        P = (self.tor.reshape(-1, 3) @ R.T) * grow * (1 - 0.7 * away)
        xy, f = project(P, dist=9.0)
        xy = xy * 1.25 + np.array([-1.7, 0.1 + 7 * away])
        segs = xy.reshape(-1, 2, 2)
        z = P[:, 2].reshape(-1, 2).mean(1)
        d = (z - z.min()) / (np.ptp(z) + 1e-9)
        cols = mix(VIOLET[None], CYAN[None] * 1.2, d[:, None] ** 1.2)
        fr.segments_graded(segs, CYAN, (0.12 + 0.95 * d ** 1.5) * (1 - away), width=1.6, bins=7, cols=cols)
        # rotation matrix panel
        pa = fade(t, 1.2, 9.0, 0.6, 0.5)
        if pa > 0:
            fr.fill(np.array([[2.9, -0.2], [6.85, -0.2], [6.85, 2.75], [2.9, 2.75]]), np.zeros(3), 0.5 * pa,
                    screen=True, mode="over")
            fr.matrix(R, (5.15, 1.55), 0.27, [CYAN * 1.1] * 3, pa, fmt="{:.2f}", label="R =")
            fr.text("v′ = R v", (4.9, 0.25), 0.3, GOLD * 1.2, pa, "math")
        fr.slogan(t, 0.9, 4.7, "三维旋转", "只是一个 3×3 矩阵", size=0.8)
        fr.slogan(t, 4.9, 8.7, "物体的每一个顶点", "都乘上同一个矩阵，世界就转动了")

    # ------------------------------------------------------------------ terrain
    def terrain(self, fr, t):
        tt = t - T_TER
        a = smooth(seg(tt, 0, 1.0))
        horizon = 0.55
        focal = 4.2
        eye = 1.05 + 0.15 * math.sin(tt * 0.9)
        roll = 0.035 * math.sin(tt * 0.7)
        speed = 7.0
        # sky stars
        fr.splat(self.stars, None, 1.0, screen=True,
                 cols=(np.outer(self.star_b * a * (0.6 + 0.4 * np.sin(tt * 3 + self.star_b * 20)), [0.8, 0.85, 1.0])).astype(np.float32))
        # sun
        self.sun(fr, horizon, a, tt)
        # terrain painter layer
        L = np.zeros((H, W, 3), np.uint8)
        C = np.zeros((H, W), np.uint8)
        step = 0.9
        xs = np.arange(-16, 16.01, 0.8)
        nz = 48
        zs = 0.6 + step * np.arange(nz) - (speed * tt) % step
        Z, X = np.meshgrid(zs, xs, indexing="ij")
        Y = terrain_h(X, Z + speed * tt)
        sx = focal * X / Z
        sy = horizon + focal * (Y - eye) / Z
        c, s = math.cos(roll), math.sin(roll)
        sx, sy = c * sx - s * (sy - horizon), s * sx + c * (sy - horizon) + horizon
        P = np.stack([W / 2 + sx * UNIT, H / 2 - sy * UNIT], -1)
        IP = np.round(np.clip(P, -30000, 30000) * SS).astype(np.int32)
        th = max(1, int(round(2 * K)))
        for r in range(nz - 1, 0, -1):
            fog = np.clip(1.25 - zs[r] / 36, 0, 1) * a
            poly = np.concatenate([IP[r], IP[r - 1][::-1]]).reshape(-1, 1, 2)
            cv2.fillPoly(L, [poly], (4, 2, 10), cv2.LINE_AA, SHIFT)
            cv2.fillPoly(C, [poly], 255, cv2.LINE_AA, SHIFT)
            col = (np.array([255, 60, 170]) * fog).astype(int).tolist()
            colv = (np.array([90, 150, 255]) * fog).astype(int).tolist()
            vert = np.stack([IP[r], IP[r - 1]], 1).reshape(-1, 2, 1, 2)
            cv2.polylines(L, list(vert), False, colv, th, cv2.LINE_AA, SHIFT)
            cv2.polylines(L, [IP[r].reshape(-1, 1, 2)], False, col, th, cv2.LINE_AA, SHIFT)
        Cf = (C.astype(np.float32) * (a / 255.0))[..., None]
        fr.img *= (1 - Cf)
        fr.img += L.astype(np.float32) * (1.5 / 255.0)
        # view-projection matrix panel
        pa = fade(tt, 1.2, T_TES - T_TER, 0.6, 0.3)
        if pa > 0:
            Pm = np.array([[focal / (FW / 2), 0, 0, 0], [0, focal / 4, 0, 0], [0, 0, -1.02, -0.2], [0, 0, -1, 0]])
            V = np.eye(4)
            V[:2, :2] = rot2(roll)
            V[1, 3] = -eye
            MVP = Pm @ V
            fr.fill(np.array([[3.0, 1.0], [6.9, 1.0], [6.9, 3.8], [3.0, 3.8]]), np.zeros(3), 0.85 * pa, screen=True, mode="over")
            fr.matrix(MVP, (5.25, 2.4), 0.22, [MAGENTA * 1.2, MAGENTA * 1.2, CYAN * 1.2, CYAN * 1.2], pa, fmt="{:.2f}", label="MVP =")
            fr.text("4 × 4 投影矩阵 · 每帧数百万次", (4.95, 1.3), 0.17, GREY * 1.3, pa, "sansm")
        fr.slogan(tt, 0.8, 6.6, "游戏里的每一帧画面", "都是数百万次矩阵乘法")

    def sun(self, fr, horizon, a, tt):
        R = 2.1
        cx, cy = 0.0, horizon + 1.05
        p = SCREEN.to_px((cx, cy))
        rp = int(R * UNIT)
        x0, y0 = int(p[0]) - rp, int(p[1]) - rp
        yy, xx = np.mgrid[0:2 * rp, 0:2 * rp].astype(np.float32)
        dx = (xx - rp) / rp
        dy = (yy - rp) / rp
        inside = np.clip((1 - np.sqrt(dx * dx + dy * dy)) * rp / 1.5, 0, 1)
        # stripes in lower half, scrolling down
        ph = (dy * 9 + tt * 0.9) % 1.0
        gap = np.clip((dy - 0.05) * 0.55, 0, 0.5)
        stripe = np.where((dy > 0.05) & (ph < gap), 0.0, 1.0)
        g = ((dy + 1) / 2)[..., None]
        col = GOLD * 1.05 * (1 - g) + MAGENTA * 1.0 * g
        src = col * (inside * stripe)[..., None]
        blit_rgb(fr.img, src.astype(np.float32), x0, y0, a, mode="add")
        fr.glow((cx, cy), 170, MAGENTA, 0.35 * a, screen=True)
        fr.glow((cx, cy), 60, GOLD, 0.25 * a, screen=True)

    # ------------------------------------------------------------------ tesseract
    def tesseract(self, fr, t):
        tt = t - T_TES
        dust(fr, t, n=600, alpha=0.8)
        R4 = rot4(0, 3, 0.75 * tt + 0.3) @ rot4(1, 2, 0.4 * tt) @ rot4(2, 3, 0.28 * tt)
        P = self.V4 @ R4.T
        w = P[:, 3]
        f = 3.2 / (3.2 - w)
        P3 = P[:, :3] * f[:, None]
        grow = expo_out(seg(tt, 0, 0.9)) * (1 + 6 * expo_in(seg(t, 22.9, 24.0)))
        P3 = P3 @ rot3(0.35, 0.25 * tt + 0.4, 0.1).T * 1.05 * grow
        xy, _ = project(P3, dist=10.0)
        xy = xy + np.array([-1.3, 0.25])
        e = self.E4
        segs = np.stack([xy[e[:, 0]], xy[e[:, 1]]], 1)
        we = (w[e[:, 0]] + w[e[:, 1]]) / 2
        q = np.clip((we + 2) / 4, 0, 1)
        bins = np.minimum((q * 5).astype(int), 4)
        for b in range(5):
            sel = bins == b
            if not sel.any():
                continue
            c = mix(CYAN, MAGENTA, (b + 0.5) / 5) * 1.2
            fr.segments(segs[sel], c, 9, 0.22)
            fr.segments(segs[sel], c, 3, 1.0)
        for i in range(16):
            c = mix(CYAN, MAGENTA, float(np.clip((w[i] + 2) / 4, 0, 1)))
            fr.glow(xy[i], 9 * f[i], WHITE, 1.1)
            fr.glow(xy[i], 26 * f[i], c, 0.5)
        pa = fade(tt, 0.8, 8.0 - 1.1, 0.6, 0.4)
        if pa > 0:
            fr.fill(np.array([[3.2, -0.55], [6.9, -0.55], [6.9, 2.6], [3.2, 2.6]]), np.zeros(3), 0.5 * pa,
                    screen=True, mode="over")
            fr.matrix(R4, (5.3, 1.1), 0.22, [CYAN * 1.1, CYAN * 1.1, MAGENTA * 1.1, MAGENTA * 1.1], pa, fmt="{:.2f}",
                      label="R =")
            fr.text("四维旋转矩阵", (5.05, -0.2), 0.17, GREY * 1.3, pa, "sansm")
        fr.slogan(tt, 0.9, 6.7, "四维超立方体", "矩阵带你看见，看不见的维度", size=0.75)

    def draw(self, t, fi):
        fr = Frame(Cam())
        if t < T_TER + 1.0:
            self.torus(fr, t)
        if T_TER <= t < T_TES:
            self.terrain(fr, t)
        if t >= T_TES:
            self.tesseract(fr, t)
        flash = flash_in(t) + (flash_in(t - T_TES, 0.4, 1.0) if t >= T_TES else 0) + 1.4 * ease_in(seg(t, 23.4, 24.0))
        return fr.finish(bloom=1.0, flash=flash, fi=fi)
