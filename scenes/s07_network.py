"""PageRank: random surfers on a web graph converge to an eigenvector."""
from scenes.common import *
from engine import _splat_1

PAL = [CYAN, MAGENTA, GREEN, VIOLET, ORANGE, BLUE]
T_SURF = 2.0


class S(Scene):
    DUR = 18.0
    STATEFUL = True

    def setup(self):
        rng = np.random.default_rng(21)
        nc, per = 6, 8
        N = nc * per
        ang = np.linspace(0, 2 * np.pi, nc, endpoint=False) + 0.4
        cc = np.stack([-1.9 + 3.7 * np.cos(ang), 0.75 + 2.1 * np.sin(ang)], 1)
        pos = np.repeat(cc, per, 0) + rng.normal(0, 0.62, (N, 2))
        for _ in range(200):  # relax overlaps
            d = pos[:, None] - pos[None]
            r = np.linalg.norm(d, axis=-1) + np.eye(N)
            f = (d / r[..., None] ** 3 * (r < 0.9)[..., None]).sum(1) * 0.02
            pos += f
            pos[:, 0] = np.clip(pos[:, 0], -6.6, 2.5)
            pos[:, 1] = np.clip(pos[:, 1], -1.9, 3.5)
        self.pos = pos
        self.cl = np.repeat(np.arange(nc), per)
        hubs = rng.choice(N, 5, replace=False)
        edges = set()
        for i in range(N):
            same = [j for j in range(N) if self.cl[j] == self.cl[i] and j != i]
            for j in rng.choice(same, rng.integers(2, 4), replace=False):
                edges.add((i, int(j)))
            if rng.random() < 0.7:
                j = int(rng.choice(hubs)) if rng.random() < 0.6 else int(rng.integers(N))
                if j != i:
                    edges.add((i, j))
        self.E = np.array(sorted(edges))
        self.N = N
        self.out = [self.E[self.E[:, 0] == i, 1] for i in range(N)]
        # Google matrix + PageRank
        A = np.zeros((N, N))
        for i, j in self.E:
            A[j, i] = 1.0 / len(self.out[i])
        G = 0.85 * A + 0.15 / N
        r = np.ones(N) / N
        for _ in range(200):
            r = G @ r
        self.G, self.pr = G, r
        # surfers
        self.M = 3200
        self.cur = rng.integers(N, size=self.M)
        self.nxt = np.array([self.pick(c, rng) for c in self.cur])
        self.prog = rng.uniform(0, 1, self.M)
        self.dur = rng.uniform(0.45, 0.9, self.M)
        self.tele = np.zeros(self.M, bool)
        self.counts = np.ones(N)
        self.rng = rng
        self.trail = np.zeros((H, W, 3), np.float32)
        self.cam = Cam()

    def pick(self, c, rng):
        if rng.random() < 0.15 or len(self.out[c]) == 0:
            return int(rng.integers(self.N))
        return int(rng.choice(self.out[c]))

    def update(self, t, dt):
        if t < T_SURF:
            return
        self.trail *= math.exp(-dt / 0.13)
        speed = 1.0 + 1.2 * smooth(seg(t, 6, 10))
        self.prog += dt * speed / self.dur
        arr = np.where(self.prog >= 1)[0]
        for m in arr:
            n = self.nxt[m]
            self.counts[n] += 1
            self.cur[m] = n
            tele = self.rng.random() < 0.15 or len(self.out[n]) == 0
            self.nxt[m] = int(self.rng.integers(self.N)) if tele else int(self.rng.choice(self.out[n]))
            self.tele[m] = tele
            self.prog[m] -= 1
        vis = smooth(seg(t, T_SURF, T_SURF + 1.0))
        q = np.clip(self.prog, 0, 1)
        q = q * q * (3 - 2 * q)
        p = self.pos[self.cur] * (1 - q[:, None]) + self.pos[self.nxt] * q[:, None]
        p = p[~self.tele]
        px = self.cam.to_px(p)
        _splat_1(self.trail, np.ascontiguousarray(px[:, 0]), np.ascontiguousarray(px[:, 1]), 0.35, 0.85, 1.0,
                 2.6 * vis)

    def draw(self, t, fi):
        fr = Frame(self.cam)
        N = self.N
        # edges
        ea = smooth(seg(t, 0.3, 1.6))
        segs = np.stack([self.pos[self.E[:, 0]], self.pos[self.E[:, 1]]], 1)
        grow = ease_out(seg(t, 0.3, 1.6))
        segs_g = segs.copy()
        segs_g[:, 1] = segs[:, 0] + (segs[:, 1] - segs[:, 0]) * grow
        fr.segments(segs_g, BLUE * 1.2, 1.3, 0.35 * ea)
        fr.img += self.trail
        # node sizes: live visit share -> PageRank
        share = self.counts / self.counts.sum()
        conv = smooth(seg(t, T_SURF, 9.0))
        rr = 0.06 + lerp(1.0 / N, share, conv) ** 0.5 * 0.62
        top = np.argsort(-share)[:5]
        rank_on = smooth(seg(t, 11.0, 12.0))
        for c in range(6):
            m = np.zeros((H, W), np.uint8)
            for i in np.where(self.cl == c)[0]:
                pop = back_out(seg(t, 0.05 * i / 2, 0.05 * i / 2 + 0.5))
                if pop <= 0:
                    continue
                p = Frame._ip(self.cam.to_px(self.pos[i]))
                cv2.circle(m, (int(p[0]), int(p[1])), int(rr[i] * pop * UNIT * SS), 255, -1, cv2.LINE_AA, SHIFT)
            fr._commit(m, PAL[c] * 0.9, 0.9, "add")
        for i in range(N):
            is_top = i in top
            col = mix(PAL[self.cl[i]], GOLD, rank_on if is_top else 0)
            fr.glow(self.pos[i], 20 * rr[i] / 0.15, col, 0.35 + (0.8 * rank_on if is_top else 0))
        if rank_on > 0:
            for k, i in enumerate(top):
                p = self.pos[i]
                fr.text(f"#{k + 1}", (p[0] + rr[i] + 0.3, p[1] + rr[i] + 0.12), 0.26, GOLD * 1.3, rank_on, "latinb", screen=False)
        # right panel: Google matrix heatmap + live ranking
        pa = fade(t, 9.6, 17.6, 0.6, 0.4)
        if pa > 0:
            fr.fill(np.array([[3.05, -2.0], [6.95, -2.0], [6.95, 3.85], [3.05, 3.85]]), np.zeros(3), 0.6 * pa,
                    screen=True, mode="over")
            G = np.sqrt(self.G / self.G.max())
            heat = (np.clip(G, 0, 1)[..., None] * np.array([0.25, 0.85, 1.0])[None, None]).astype(np.float32)
            side = int(2.3 * UNIT)
            hm = cv2.resize(heat, (side, side), interpolation=cv2.INTER_NEAREST)
            c = SCREEN.to_px((5.0, 2.35))
            blit_rgb(fr.img, hm, c[0] - side / 2, c[1] - side / 2, pa, mode="add")
            fr.text("G", (3.5, 2.35), 0.4, WHITE, pa, "math")
            fr.text("G r = r", (5.0, 0.72), 0.4, GOLD * 1.3, pa, "math")
            for k, i in enumerate(top):
                y = 0.05 - k * 0.38
                w = share[i] / share[top[0]] * 2.4
                fr.text(f"#{k + 1}", (3.45, y), 0.2, GOLD * 1.2, pa, "latinb")
                fr.polylines([np.array([[3.8, y], [3.8 + w, y]])], PAL[self.cl[i]] * 1.2, 9, pa, screen=True)
                fr.text(f"{share[i]:.3f}", (3.9 + w + 0.35, y), 0.17, WHITE, pa, "num")
        fr.slogan(t, 1.0, 5.2, "互联网，是一张巨大的网", "网页是节点，链接是边")
        fr.slogan(t, 5.6, 9.8, "无数用户随机点击", "他们最终会停留在哪里？")
        fr.slogan(t, 10.3, 17.2, "PageRank", "答案是一个特征向量 —— 它撑起了搜索引擎的排序", size=0.7)
        flash = flash_in(t) + 1.4 * ease_in(seg(t, 17.45, 18.0))
        return fr.finish(bloom=1.0, flash=flash, fi=fi)
