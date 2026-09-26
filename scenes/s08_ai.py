"""AI: a digit becomes a vector and flows through matrices; lifting data to 3D; word vectors."""
from PIL import Image, ImageDraw
from scenes.common import *

T2 = 9.0     # lifting part
T3 = 17.0    # embeddings part

L_X = [-4.5, -1.7, 0.9, 3.5]
L_N = [64, 16, 16, 10]
CY = 0.45


def digit_pixels():
    im = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(im).text((32, 34), "7", font=font("sans", 62), fill=255, anchor="mm")
    a = np.asarray(im, np.float32) / 255
    return cv2.resize(a, (8, 8), interpolation=cv2.INTER_AREA).clip(0, 1) ** 0.8


class S(Scene):
    DUR = 28.0

    def setup(self):
        rng = np.random.default_rng(8)
        self.dig = digit_pixels()
        # node positions
        self.nodes = []
        for x, n in zip(L_X, L_N):
            sp = 5.4 / n if n == 64 else (0.33 if n == 16 else 0.52)
            ys = CY + (np.arange(n) - (n - 1) / 2) * -sp
            self.nodes.append(np.stack([np.full(n, x), ys], 1))
        self.W = [rng.normal(0, 1, (L_N[l + 1], L_N[l])) for l in range(3)]
        x = self.dig.flatten()
        h1 = np.clip(rng.uniform(0, 1, 16) ** 2 * 1.2, 0, 1)
        h2 = np.clip(rng.uniform(0, 1, 16) ** 2 * 1.2, 0, 1)
        out = rng.uniform(0.03, 0.18, 10)
        out[7] = 1.0
        self.act = [x, h1, h2, out]
        # strongest edges per layer carry the pulses
        self.pulse_edges = []
        for l in range(3):
            score = np.abs(self.W[l]) * self.act[l][None, :]
            idx = np.dstack(np.unravel_index(np.argsort(-score.ravel())[:110], score.shape))[0]
            self.pulse_edges.append(idx)
        # lifting dataset
        n = 650
        r_in = np.sqrt(rng.uniform(0, 1, n)) * 1.0
        r_out = rng.uniform(1.6, 2.3, n)
        th1, th2 = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        self.pa = np.stack([r_in * np.cos(th1), r_in * np.sin(th1)], 1)
        self.pb = np.stack([r_out * np.cos(th2), r_out * np.sin(th2)], 1)
        # embeddings
        g = np.array([1.5, 0.15, -0.4])
        r = np.array([0.25, 1.6, 0.5])
        c = np.array([1.6, -0.3, 0.6])
        man = np.array([-2.3, -1.2, 0.2])
        self.words = {"男人": man, "女人": man + g, "国王": man + r, "女王": man + g + r + rng.normal(0, 0.08, 3)}
        for cn, cap, p in (("中国", "北京", (0.9, 2.0, -0.5)), ("日本", "东京", (1.2, 0.7, -0.8)), ("法国", "巴黎", (0.8, -0.6, -0.6))):
            p = np.array(p)
            self.words[cn] = p
            self.words[cap] = p + c + rng.normal(0, 0.1, 3)
        extra = ["猫", "狗", "音乐", "数学", "星星", "苹果", "河流", "诗歌", "咖啡", "宇宙", "电影", "春天"]
        anchors = list(self.words.values())
        for w in extra:
            while True:
                v = rng.normal(0, 1.0, 3) * np.array([3.2, 1.3, 1.8]) + np.array([0, 0.5, 0])
                if min(np.linalg.norm(v[:2] - a[:2]) for a in anchors) > 1.1:
                    break
            self.words[w] = v
            anchors.append(v)
        self.extra = set(extra)
        self.cloud = rng.normal(0, 1, (900, 3)) * np.array([2.8, 1.9, 1.9])
        self.cloud_b = rng.uniform(0.15, 0.6, 900)

    def draw(self, t, fi):
        fr = Frame(Cam())
        if t < T2 + 0.2:
            self.part_nn(fr, t)
        if T2 <= t < T3 + 0.2:
            self.part_lift(fr, t)
        if t >= T3:
            self.part_embed(fr, t)
        flash = flash_in(t)
        for tb in (T2, T3):
            if t >= tb:
                flash += flash_in(t - tb, 0.35, 0.8)
        flash += 1.4 * ease_in(seg(t, 27.4, 28.0))
        return fr.finish(bloom=1.0, flash=flash, fi=fi)

    # ------------------------------------------------------------------ neural net
    def part_nn(self, fr, t):
        out = 1 - smooth(seg(t, T2 - 0.5, T2))
        # digit grid -> column
        fly = ease_io(seg(t, 1.8, 3.3))
        cell0 = 0.42
        g0 = np.array([-5.2, CY])
        colp = self.nodes[0]
        m = np.zeros((H, W), np.uint8)
        vals = self.dig.flatten()
        sq = []
        for k in range(64):
            i, j = divmod(k, 8)
            a = g0 + np.array([(j - 3.5) * cell0, (3.5 - i) * cell0])
            b = colp[k]
            p = lerp(a, b, fly)
            half = lerp(cell0 * 0.46, 0.036, fly)
            sq.append((p, half, vals[k]))
        for p, half, v in sq:
            poly = np.array([p + [-half, -half], p + [half, -half], p + [half, half], p + [-half, half]])
            fr.fill(poly, mix(np.array([0.06, 0.08, 0.16]), CYAN * 1.3, v), out * smooth(seg(t, 0, 0.6)), mode="over")
        # edges
        ea = smooth(seg(t, 3.0, 3.8)) * out
        if ea > 0:
            for l in range(3):
                A, B = self.nodes[l], self.nodes[l + 1]
                Wl = self.W[l]
                ii, jj = np.meshgrid(np.arange(len(B)), np.arange(len(A)), indexing="ij")
                segs = np.stack([A[jj.ravel()] + [0.05, 0], B[ii.ravel()]], 1)
                w = Wl.ravel()
                pos = w > 0
                fr.segments_graded(segs[pos], CYAN, np.abs(w[pos]) * 0.1 * ea, 1.0, bins=4)
                fr.segments_graded(segs[~pos], MAGENTA, np.abs(w[~pos]) * 0.1 * ea, 1.0, bins=4)
                fr.text(f"W{'₁₂₃'[l]}", ((A[0, 0] + B[0, 0]) / 2, 3.55), 0.3, WHITE, ea, "math")
        # forward pulses
        for l in range(3):
            t0 = 3.9 + 0.75 * l
            ph = seg(t, t0, t0 + 0.75)
            if 0 < ph < 1:
                A, B = self.nodes[l], self.nodes[l + 1]
                e = self.pulse_edges[l]
                a = A[e[:, 1]]
                b = B[e[:, 0]]
                q = np.clip(ph * 1.25 - np.linspace(0, 0.25, len(e)), 0, 1)[:, None]
                pts = a + (b - a) * q
                fr.segments(np.stack([a, b], 1), GOLD, 1.2, 0.35 * math.sin(math.pi * ph) * out)
                fr.splat(pts, WHITE, 3.0 * out)
                for p in pts[::6]:
                    fr.glow(p, 7, GOLD, 0.7 * out)
        # hidden/output nodes
        for l in range(1, 4):
            lit = smooth(seg(t, 3.9 + 0.75 * l - 0.1, 3.9 + 0.75 * l + 0.3))
            na = smooth(seg(t, 2.9, 3.6)) * out
            for k, p in enumerate(self.nodes[l]):
                v = self.act[l][k] * lit
                col = mix(CYAN, GOLD, 1.0 if (l == 3 and k == 7) else 0.0)
                fr.circle(p, 0.1, WHITE, 1.4, 0.45 * na)
                if v > 0.02:
                    fr.circle(p, 0.085, col * 1.2, fill=True, alpha=v * na)
                    fr.glow(p, 10, col, 0.5 * v * na)
                if l == 3:
                    fr.text(str(k), (p[0] + 0.38, p[1]), 0.22, mix(GREY * 1.3, GOLD * 1.4, v if k == 7 else 0), na, "num")
        ra = fade(t, 6.4, T2 - 0.2, 0.4, 0.4)
        if ra > 0:
            fr.text("识别结果", (5.75, 1.75), 0.22, GREY * 1.4, ra, "sansm")
            fr.title("7", (5.75, 0.55), 1.4, t, 6.4, T2 - 0.2, color=GOLD * 1.35, font_key="num", glow=0.5, rise=0.2)
        fr.slogan(t, 0.5, 3.9, "人工智能怎样看懂一个数字？", "先把图像，展开成一个向量", size=0.58)
        fr.slogan(t, 4.3, 8.7, "一层，又一层的矩阵乘法", "h = σ(W x + b)", size=0.62)

    # ------------------------------------------------------------------ lift to 3D
    def proj_lift(self, P, el, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        x = c * P[:, 0] - s * P[:, 1]
        y = s * P[:, 0] + c * P[:, 1]
        z = P[:, 2]
        sy = y * math.sin(el) + z * math.cos(el)
        depth = -y * math.cos(el) + z * math.sin(el)
        f = 9.0 / (9.0 - depth * 0.6)
        return np.stack([x * f, sy * f - 0.35], 1) * 1.15

    def part_lift(self, fr, t):
        tt = t - T2
        a = smooth(seg(tt, 0, 0.6)) * (1 - smooth(seg(t, T3 - 0.4, T3)))
        lift = ease_io(seg(tt, 2.2, 4.4))
        el = lerp(math.pi / 2, 0.42, ease_io(seg(tt, 1.9, 4.2)))
        yaw = 0.25 * ease_io(seg(tt, 1.9, 8)) + 0.12 * tt
        k = 0.42
        A3 = np.c_[self.pa, k * lift * (self.pa ** 2).sum(1)]
        B3 = np.c_[self.pb, k * lift * (self.pb ** 2).sum(1)]
        # base grid
        gl = []
        for v in np.linspace(-3, 3, 13):
            gl.append(np.array([[v, -3, 0], [v, 3, 0]]))
            gl.append(np.array([[-3, v, 0], [3, v, 0]]))
        gl = np.array(gl)
        g2 = self.proj_lift(gl.reshape(-1, 3), el, yaw).reshape(-1, 2, 2)
        fr.segments(g2, BLUE, 1.0, 0.25 * a)
        # separating plane
        pa_ = smooth(seg(tt, 4.4, 5.4)) * a
        zc = k * 1.75
        if pa_ > 0:
            sq = np.array([[-2.9, -2.9, zc], [2.9, -2.9, zc], [2.9, 2.9, zc], [-2.9, 2.9, zc]])
            s2 = self.proj_lift(sq, el, yaw)
            fr.fill(s2, GOLD, 0.12 * pa_)
            fr.polylines([s2], GOLD * 1.3, 2.5, 0.9 * pa_, closed=True)
        for P, col in ((A3, CYAN), (B3, MAGENTA)):
            p2 = self.proj_lift(P, el, yaw)
            m = np.zeros((H, W), np.uint8)
            ip = Frame._ip(fr.cam.to_px(p2))
            rr = int(3.2 * K * SS)
            for q in ip:
                cv2.circle(m, (int(q[0]), int(q[1])), rr, 255, -1, cv2.LINE_AA, SHIFT)
            fr._commit(m, col * 1.2, a, "add")
        # impossible straight line attempt (flat view)
        la = fade(tt, 0.9, 2.1, 0.3, 0.3)
        if la > 0:
            ang = tt * 1.7
            d = np.array([math.cos(ang), math.sin(ang)])
            fr.segments([[-d * 4 + [0, -0.35], d * 4 + [0, -0.35]]], RED * 1.3, 3, la)
        fr.slogan(tt, 0.4, 3.9, "纠缠在一起的数据", "在平面上，一条直线无论如何也分不开", size=0.6)
        fr.slogan(tt, 4.3, 7.8, "升到更高的维度", "一个平面，就能干净利落地切开", size=0.6)

    # ------------------------------------------------------------------ word vectors
    def part_embed(self, fr, t):
        tt = t - T3
        a = smooth(seg(tt, 0, 0.8))
        yaw = -0.3 + 0.045 * tt
        R = rot3(-0.28, yaw, 0)
        suck = ease_in(seg(t, 26.6, 27.9))

        def P(v):
            v = np.atleast_2d(v) @ R.T * (1 - suck)
            xy, f = project(v, dist=11.0)
            return xy * 1.12 + np.array([0, 0.25]), f

        c2, cf = P(self.cloud)
        cols = (np.outer(self.cloud_b * cf ** 2 * a, [0.45, 0.6, 1.0])).astype(np.float32)
        fr.splat(c2, None, 1.0, cols=cols * 2.5)
        pos = {w: P(v) for w, v in self.words.items()}
        key = {"男人", "女人", "国王", "女王"}
        caps = {"中国", "北京", "日本", "东京", "法国", "巴黎"}
        for w, (p, f) in pos.items():
            p = p[0]
            f = float(f[0])
            if w in self.extra:
                col, al, sz = GREY * 1.2, 0.55, 0.19
            elif w in key:
                col, al, sz = WHITE, 1.0, 0.26
            else:
                col, al, sz = mix(WHITE, VIOLET, 0.3), 0.9 * smooth(seg(tt, 5.5, 6.3)) + 0.4, 0.22
            fr.glow(p, 6 * f, col, 0.9 * a * al)
            fr.text(w, (p[0], p[1] + 0.26), sz * f, col, a * al, "sans", screen=False)
        # the parallelogram
        def arr(w1, w2, col, t0, width=4):
            e = expo_out(seg(tt, t0, t0 + 0.7))
            if e <= 0:
                return
            p1, p2 = pos[w1][0][0], pos[w2][0][0]
            fr.arrow(p1, p1 + (p2 - p1) * e, col, width, a * (1 - suck), head=0.22, tip_glow=0.5)
        arr("男人", "国王", GOLD * 1.2, 2.6)
        arr("女人", "女王", GOLD * 1.2, 2.9)
        arr("男人", "女人", CYAN * 1.2, 3.4, 3)
        arr("国王", "女王", CYAN * 1.2, 3.6, 3)
        arr("中国", "北京", VIOLET * 1.3, 6.0, 3)
        arr("日本", "东京", VIOLET * 1.3, 6.2, 3)
        arr("法国", "巴黎", VIOLET * 1.3, 6.4, 3)
        fa = fade(tt, 4.6, 9.8, 0.5, 0.4)
        if fa > 0:
            fr.band(3.3, fa, 1.4)
            fr.title("国王 − 男人 + 女人 ≈ 女王", (0, 3.3), 0.5, tt, 4.6, 9.8, color=GOLD * 1.3, font_key="sans",
                     stagger=0.04, glow=0.25)
        fr.slogan(tt, 0.4, 4.4, "词语，也是向量", "意思相近的词，在空间里彼此靠近", size=0.68)
        fr.slogan(tt, 4.8, 10.4, "语义，变成了几何", "大语言模型写下的每一个字，都诞生于这样的向量空间", size=0.68)
