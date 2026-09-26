"""Finale: beat-cut montage of every chapter, collapse to a point, explosion, title."""
import importlib
import engine as E
from scenes.common import *

CLIPS = [
    ("s00_open", 9.4), ("s01_space", 8.0), ("s02_eigen", 12.0), ("s03_3d", 11.5),
    ("s04_fractal", 7.2), ("s06_fourier", 16.0), ("s07_network", 8.0), ("s08_ai", 14.5),
    ("s02_eigen", 17.0), ("s03_3d", 19.0), ("s04_fractal", 12.8), ("s05_image", 16.0),
    ("s08_ai", 7.0), ("s01_space", 16.9), ("s04_fractal", 10.6), ("s06_fourier", 5.5),
]
CUT = 0.5
T_MONT = len(CLIPS) * CUT      # 8.0
T_BOOM = 10.0
PAL = [CYAN, MAGENTA, GOLD, GREEN, VIOLET, BLUE]


class S(Scene):
    DUR = 24.0

    def setup(self):
        self.sc = {}
        self.next_t = {}
        for name, _ in CLIPS:
            if name not in self.sc:
                s = importlib.import_module(f"scenes.{name}").S()
                s.setup()
                self.sc[name] = s
                self.next_t[name] = 0.0
        rng = np.random.default_rng(99)
        self.conv = rng.uniform(-1, 1, (3000, 2)) * np.array([FW / 2 + 1, 5])
        n = 7000
        ang = rng.uniform(0, 2 * np.pi, n)
        self.bdir = np.stack([np.cos(ang), np.sin(ang) * 0.75], 1)
        self.bspd = rng.uniform(1.5, 16, n) ** 1.0
        self.bcol = np.array(PAL, np.float32)[rng.integers(0, len(PAL), n)] * rng.uniform(0.3, 1.0, n)[:, None]

    # --------------------------------------------------------------- montage
    def montage(self, t, fi):
        k = min(len(CLIPS) - 1, int(t / CUT))
        name, t0 = CLIPS[k]
        tl = t - k * CUT
        local = t0 + tl
        sc = self.sc[name]
        dt = 1.0 / FPS
        if getattr(sc, "STATEFUL", False):
            while self.next_t[name] <= local + 1e-6:
                sc.update(self.next_t[name], dt)
                self.next_t[name] += dt
        E.SUPPRESS_SLOGAN = True
        img = sc.draw(local, fi)
        E.SUPPRESS_SLOGAN = False
        # punch-in zoom + chromatic split + cut flash
        z = 1.0 + 0.09 * (1 - ease_out(seg(tl, 0, CUT)))
        M = np.array([[z, 0, (1 - z) * W / 2], [0, z, (1 - z) * H / 2]])
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ab = int(round(10 * K * (1 - seg(tl, 0, 0.12))))
        if ab > 0:
            img = img.copy()
            img[:, ab:, 0] = img[:, :-ab, 0]
            img[:, :-ab, 2] = img[:, ab:, 2]
        fl = (1 - seg(tl, 0, 0.18)) ** 2 * 0.45
        if k == 0:
            fl = max(fl, (1 - seg(t, 0, 0.4)) ** 2 * 1.0)
        if fl > 0:
            img = np.clip(img.astype(np.float32) + fl * 255, 0, 255).astype(np.uint8)
        return img

    # --------------------------------------------------------------- finale
    def floor(self, fr, t, a):
        horizon = -0.75
        focal = 3.6
        eye = 1.0
        segs = []
        zs = 1.0 + np.arange(20) * 1.0 - (t * 1.2) % 1.0
        for z in zs:
            y = horizon + focal * (0 - eye) / z
            segs.append([[-40 * focal / z, y], [40 * focal / z, y]])
        segs = np.array(segs)
        al = np.clip(1.1 - zs / 18, 0, 1) ** 1.5
        for x in np.arange(-20, 21, 1.0):
            segs = np.vstack([segs, [[[x * focal / 1.0, horizon - focal * eye / 1.0], [x * focal / 40, horizon - focal * eye / 40]]]])
            al = np.r_[al, 0.6]
        fr.segments_graded(segs, CYAN, al * 0.5 * a, width=1.4, bins=6, screen=True)

    def draw(self, t, fi):
        if t < T_MONT:
            return self.montage(t, fi)
        cam = Cam()
        cam.sx, cam.sy = shake_amt(t, T_BOOM, 30, 0.8, seed=3)
        fr = Frame(cam, bg=BG * smooth(seg(t, T_BOOM, T_BOOM + 1.5)))
        end = 1 - smooth(seg(t, 21.8, 23.7))
        # converge to a single point
        if t < T_BOOM:
            p = ease_in(seg(t, T_MONT, T_BOOM - 0.05))
            pts = self.conv * (1 - p)
            b = 0.25 + 1.2 * p
            fr.splat(pts, np.array([0.5, 0.75, 1.0]) * b, 1.0)
            g = seg(t, T_MONT, T_BOOM)
            fr.glow((0, 0), 10 + 30 * g ** 3, WHITE, 0.6 + 3.0 * g ** 3)
            fr.glow((0, 0), 60 + 120 * g ** 3, CYAN, 0.2 + 0.8 * g ** 3)
            flash = 0.0
        else:
            tau = t - T_BOOM
            self.floor(fr, t, smooth(seg(tau, 0.2, 2.0)) * end)
            # burst particles
            d = self.bspd * (1 - np.exp(-2.2 * tau)) / 2.2 + self.bspd * 0.03 * tau
            pts = self.bdir * d[:, None]
            fade_b = (np.exp(-tau * 0.35) * 0.9 + 0.1) * end
            fr.splat(pts, None, 1.0, cols=(self.bcol * fade_b * 1.8).astype(np.float32))
            # shockwaves
            for k, (dl, col) in enumerate(((0.0, WHITE), (0.12, CYAN), (0.25, MAGENTA))):
                q = seg(tau, dl, dl + 1.6)
                if 0 < q < 1:
                    fr.circle((0, 0), 0.2 + 11 * expo_out(q), col * 1.3, 5 - k, (1 - q) ** 1.5)
            fr.glow((0, 0), 90, CYAN, 1.5 * math.exp(-tau * 2.5))
            # title block
            fr.title("线性代数", (0, 0.8), 1.45, t, T_BOOM + 0.3, 23.7, color=WHITE * 1.15, stagger=0.14, fin=0.7,
                     rise=0.22, track0=0.06, track1=0.24, glow=0.35, fout=1.8)
            fr.title("LINEAR  ALGEBRA", (0, -0.5), 0.3, t, T_BOOM + 1.5, 23.7, color=GOLD * 1.25, font_key="latin",
                     stagger=0.04, fin=0.5, rise=0.3, track0=0.25, track1=0.6, fout=1.8)
            fr.title("理解世界的底层语言", (0, -1.5), 0.4, t, T_BOOM + 3.2, 23.7, color=mix(WHITE, CYAN, 0.25),
                     font_key="sans", stagger=0.06, fin=0.6, rise=0.25, track0=0.1, track1=0.25, fout=1.8)
            fr.title("向量 · 矩阵 · 行列式 · 特征值 · 奇异值 · 变换", (0, -2.3), 0.2, t, T_BOOM + 4.8, 23.7,
                     color=GREY * 1.3, font_key="sansm", stagger=0.02, fin=0.5, rise=0.2, track0=0.1, track1=0.3, fout=1.8)
            flash = 1.6 * (1 - seg(tau, 0, 0.45)) ** 3
        out = fr.finish(bloom=1.1, flash=flash, fi=fi, exposure=end if t > 20 else 1.0)
        return out
