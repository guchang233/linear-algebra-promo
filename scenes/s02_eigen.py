"""Eigenvectors: the directions that survive a transformation; then the flows they govern."""
from scenes.common import *
from engine import _splat_rgb

A_FAN = np.array([[1.5, 0.5], [0.5, 1.5]])
R30 = rot2(np.radians(28))
SADDLE = R30 @ np.diag([0.95, -0.95]) @ R30.T
SPIRAL = np.array([[-0.22, -1.7], [1.7, -0.22]])
ELLIPSE = np.array([[0.05, -2.3], [0.85, 0.05]])
PART_B = 9.4


def flow_A(t):
    if t < 14.2:
        return SADDLE
    if t < 19.2:
        return lerp(SADDLE, SPIRAL, ease_io(seg(t, 14.2, 15.4)))
    return lerp(SPIRAL, ELLIPSE, ease_io(seg(t, 19.2, 20.4)))


GRAD = [hexc("4a2aa8"), hexc("3d7bff"), hexc("34e1ff"), hexc("ff6fb0"), hexc("ffd6a0")]


def grad(x):
    x = np.clip(x, 0, 0.999) * (len(GRAD) - 1)
    i = x.astype(int)
    f = (x - i)[:, None]
    G = np.array(GRAD)
    return G[i] * (1 - f) + G[i + 1] * f


class S(Scene):
    DUR = 24.0
    STATEFUL = True

    def setup(self):
        self.rng = np.random.default_rng(5)
        self.N = 26000
        self.p = self.spawn(self.N)
        self.age = self.rng.uniform(0, 3, self.N)
        self.life = self.rng.uniform(1.8, 4.0, self.N)
        self.trail = np.zeros((H, W, 3), np.float32)
        self.cam = Cam()

    def spawn(self, n):
        return self.rng.uniform(-1, 1, (n, 2)) * np.array([FW / 2 + 0.5, 4.5])

    def update(self, t, dt):
        if t < PART_B - 0.2:
            return
        self.trail *= math.exp(-dt / 0.22)
        A = flow_A(t)
        sub = 3
        vis = smooth(seg(t, PART_B - 0.2, PART_B + 1.2)) * (1 - ease_in(seg(t, 23.0, 23.9)))
        for k in range(sub):
            v = self.p @ A.T
            self.p = self.p + v * (dt / sub)
            sp = np.linalg.norm(v, axis=1)
            a = np.minimum(1, np.minimum(self.age / 0.4, (self.life - self.age) / 0.4)).clip(0, 1)
            cols = (grad(sp / 6.0) * (a * vis * 0.55)[:, None]).astype(np.float32)
            px = self.cam.to_px(self.p)
            _splat_rgb(self.trail, np.ascontiguousarray(px[:, 0]), np.ascontiguousarray(px[:, 1]), cols, 1.0 / sub * 2.2 * K)
        self.age += dt
        dead = (self.age > self.life) | (np.abs(self.p[:, 0]) > FW / 2 + 1) | (np.abs(self.p[:, 1]) > 5) | (
                np.linalg.norm(self.p, axis=1) < 0.04)
        nd = int(dead.sum())
        if nd:
            self.p[dead] = self.spawn(nd)
            self.age[dead] = 0
            self.life[dead] = self.rng.uniform(1.8, 4.0, nd)

    def draw(self, t, fi):
        fr = Frame(self.cam)
        if t < PART_B + 0.5:
            self.part_a(fr, t)
        if t > PART_B - 0.3:
            self.part_b(fr, t)
        flash = flash_in(t) + 1.4 * ease_in(seg(t, 23.4, 24.0)) + 0.5 * (1 - seg(t, PART_B - 0.05, PART_B + 0.4)) * (t > PART_B - 0.05)
        return fr.finish(bloom=1.0, flash=flash, fi=fi)

    def part_a(self, fr, t):
        out = 1 - smooth(seg(t, PART_B - 0.8, PART_B + 0.1))
        e = ease_io(seg(t, 1.0, 3.4))
        M = lerp(np.eye(2), A_FAN, e)
        segs, axes = grid_segments(M, n=12)
        fr.segments(segs, BLUE, 1.2, 0.28 * out)
        fr.segments(axes, WHITE, 1.6, 0.35 * out)
        ang = np.radians(np.arange(0, 360, 5)) + 1e-3
        R = 1.8
        grow = expo_out(seg(t, 0.1, 1.0))
        eig_on = smooth(seg(t, 3.2, 4.0))
        for a in ang:
            v = np.array([math.cos(a), math.sin(a)]) * R * grow
            w = M @ v
            dev = abs(math.atan2(v[0] * w[1] - v[1] * w[0], v @ w))
            is_eig = min(abs((a - np.pi / 4) % (np.pi / 2)), abs((a - np.pi / 4) % (np.pi / 2) - np.pi / 2)) < 0.02
            if is_eig:
                c = mix(WHITE, GOLD * 1.4, eig_on)
                fr.arrow((0, 0), w, c, 4.5, out, head=0.26, tip_glow=0.9 * eig_on * out)
            else:
                c = mix(CYAN, VIOLET, min(1, dev / 0.35))
                fr.arrow((0, 0), w, c, 2.0, out * (0.75 - 0.35 * eig_on), head=0.16)
        if eig_on > 0:
            for d, lam in ((np.array([1, 1]), "λ = 2"), (np.array([1, -1]), "λ = 1")):
                u = d / np.linalg.norm(d)
                fr.segments([[-u * 20, u * 20]], GOLD, 2.0, 0.55 * eig_on * out)
                p = u * (R * (2 if lam.endswith("2") else 1) + 0.9)
                fr.text(lam, (p[0] + 0.25, p[1]), 0.34, GOLD * 1.3, eig_on * out, "math")
        # the formula
        fa = fade(t, 4.2, 9.0, 0.6, 0.5)
        if fa > 0:
            fr.band(3.0, fa, 1.6)
            fr.title("A v = λ v", (0, 3.05), 0.62, t, 4.2, 9.0, color=GOLD * 1.25, font_key="math", stagger=0.05,
                     glow=0.3)
        fr.slogan(t, 3.9, 8.8, "特征向量", "变换之后，唯一不改变方向的向量", size=0.8)

    def part_b(self, fr, t):
        vis = smooth(seg(t, PART_B - 0.2, PART_B + 1.0))
        A = flow_A(t)
        fr.segments(grid_segments(np.eye(2), n=12)[0], BLUE, 1.0, 0.08 * vis)
        fr.img += self.trail
        # eigen lines when real
        tr, det = np.trace(A), np.linalg.det(A)
        disc = tr * tr / 4 - det
        if disc > 0:
            ev, V = np.linalg.eig(A)
            a = min(1.0, disc * 3) * vis * (1 - seg(t, 23.0, 23.6))
            for k in range(2):
                u = np.real(V[:, k])
                u /= np.linalg.norm(u)
                col = GOLD if np.real(ev[k]) > 0 else MAGENTA
                fr.segments([[-u * 20, u * 20]], col, 5, 0.3 * a)
                fr.segments([[-u * 20, u * 20]], col * 1.3, 2, 0.9 * a)
        # matrix + eigenvalue readout
        pa = fade(t, PART_B + 0.6, 23.4, 0.6, 0.5)
        if pa > 0:
            fr.fill(np.array([[3.35, 1.55], [6.8, 1.55], [6.8, 3.75], [3.35, 3.75]]), np.zeros(3), 0.55 * pa,
                    screen=True, mode="over")
            fr.matrix(A, (5.5, 3.0), 0.28, [WHITE, WHITE], pa, fmt="{:.2f}", label="A =")
            if disc >= 0:
                l1 = tr / 2 + math.sqrt(disc)
                l2 = tr / 2 - math.sqrt(disc)
                s = f"λ₁ = {l1:.2f}    λ₂ = {l2:.2f}"
                col = GOLD
            else:
                s = f"λ = {tr / 2:.2f} ± {math.sqrt(-disc):.2f} i"
                col = CYAN
            fr.text(s.replace("-", "−"), (5.1, 1.95), 0.26, col * 1.2, pa, "num")
        fr.slogan(t, PART_B + 0.7, 14.2, "让变换一次次作用下去", "特征向量，决定了一切流动的方向")
        fr.slogan(t, 15.1, 19.1, "复数特征值", "旋转，化作漩涡与星系", size=0.8)
        fr.slogan(t, 19.8, 23.5, "从行星轨道到电路振荡", "动力系统的命运，都写在特征值里")
