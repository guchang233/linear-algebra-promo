"""A photo is a matrix: zoom to the numbers, then rebuild it with SVD rank by rank."""
from scenes.common import *
from scenes.landscape import make_landscape

IW, IH = 960, 540
RANKS = [(10.5, 1), (11.5, 2), (12.5, 3), (13.5, 5), (14.5, 8), (15.0, 12), (15.5, 20), (16.0, 30), (16.5, 50),
         (17.0, 80), (17.5, 540)]
T_SVD = 9.9
T_DIS = 19.6


class S(Scene):
    DUR = 22.0

    def setup(self):
        self.img = make_landscape(IW, IH)
        self.lum = (self.img @ np.array([0.299, 0.587, 0.114], np.float32))
        U, Sg, Vt = [], [], []
        for c in range(3):
            u, s, vt = np.linalg.svd(self.img[..., c].astype(np.float64), full_matrices=False)
            U.append(u)
            Sg.append(s)
            Vt.append(vt)
        self.sv = np.mean(Sg, 0)
        self.recon = {}
        for _, k in RANKS:
            self.recon[k] = np.clip(np.stack([(U[c][:, :k] * Sg[c][:k]) @ Vt[c][:k] for c in range(3)], -1), 0, 1).astype(np.float32)
        e = np.cumsum(self.sv ** 2)
        self.energy = e / e[-1]
        mu = cv2.blur(self.lum, (14, 8))
        var = cv2.blur(self.lum ** 2, (14, 8)) - mu ** 2
        var[:230] = 0
        var[430:] = 0
        var[:, :120] = 0
        var[:, -120:] = 0
        tv, tu = np.unravel_index(np.argmax(var), var.shape)
        self.target = (tu + 0.5, tv + 0.5)
        rng = np.random.default_rng(3)
        n = 140000
        self.pu = rng.uniform(0, IW - 1, n)
        self.pv = rng.uniform(0, IH - 1, n)
        self.pc = self.img[self.pv.astype(int), self.pu.astype(int)]
        self.pvel = rng.normal(0, 1, (n, 2))
        self.pdel = rng.uniform(0, 0.8, n)

    # -------------------------------------------------------------- viewing helper
    def view(self, fr, src, cx, cy, w, h, u0, v0, s, alpha=1.0):
        """Draw src with image coord (u0,v0) at screen-px centre (cx,cy), s screen px per image px."""
        w, h = int(w), int(h)
        Minv = np.array([[1 / s, 0, u0 - w / (2 * s)], [0, 1 / s, v0 - h / (2 * s)]], np.float64)
        flag = cv2.INTER_LINEAR if s < 3 else cv2.INTER_NEAREST
        out = cv2.warpAffine(src, Minv, (w, h), flags=flag | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        blit_rgb(fr.img, out, cx - w / 2, cy - h / 2, alpha)
        fr.polylines([np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2],
                                [cx - w / 2, cy + h / 2]])], WHITE, 1.2, 0.35 * alpha, closed=True, screen=False) if False else None
        return Minv

    def frame_rect(self, fr, cx, cy, w, h, a):
        pts = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
        m = np.zeros((H, W), np.uint8)
        cv2.polylines(m, [np.round(pts * SS).astype(np.int32).reshape(-1, 1, 2)], True, 255, max(1, int(1.5 * K)), cv2.LINE_AA, SHIFT)
        fr._commit(m, WHITE, 0.35 * a, "add")

    def draw(self, t, fi):
        fr = Frame(Cam())
        dust(fr, t, n=300, alpha=0.5)
        if t < T_SVD + 0.6:
            self.part_zoom(fr, t)
        if t >= T_SVD:
            self.part_svd(fr, t)
        flash = flash_in(t) + 1.4 * ease_in(seg(t, 21.45, 22.0))
        return fr.finish(bloom=0.8, thresh=0.6, flash=flash, fi=fi)

    # -------------------------------------------------------------- zoom into the pixels
    def part_zoom(self, fr, t):
        out = 1 - smooth(seg(t, T_SVD, T_SVD + 0.6))
        dw, dh = 1344 * K, 756 * K
        cx, cy = W / 2, H / 2 - 40 * K
        s_full = dw / IW
        # target region: ridge + sun glow
        tu, tv = self.target
        zin = ease_io(seg(t, 3.6, 7.2))
        zout = ease_io(seg(t, 8.7, T_SVD))
        z = zin * (1 - zout)
        s = math.exp(lerp(math.log(s_full), math.log(96 * K), z))
        u0 = lerp(IW / 2, tu, min(1, z * 1.6) if zout == 0 else z)
        v0 = lerp(IH / 2, tv, min(1, z * 1.6) if zout == 0 else z)
        # pixelated reveal
        src = self.img
        blk = [48, 24, 12, 6, 3, 1]
        bi = min(len(blk) - 1, int(seg(t, 0.2, 2.6) * len(blk)))
        if blk[bi] > 1:
            b = blk[bi]
            small = cv2.resize(self.img, (IW // b, IH // b), interpolation=cv2.INTER_AREA)
            src = cv2.resize(small, (IW, IH), interpolation=cv2.INTER_NEAREST)
        a = smooth(seg(t, 0, 0.5)) * out
        self.view(fr, src, cx, cy, dw, dh, u0, v0, s, a)
        self.frame_rect(fr, cx, cy, dw, dh, a)
        # grid + numbers when zoomed in
        ga = seg(s, 14 * K, 34 * K) * a
        if ga > 0:
            u_lo, u_hi = int(u0 - dw / (2 * s)) - 1, int(u0 + dw / (2 * s)) + 2
            v_lo, v_hi = int(v0 - dh / (2 * s)) - 1, int(v0 + dh / (2 * s)) + 2
            segs = []
            for u in range(u_lo, u_hi + 1):
                x = cx + (u - u0) * s
                if cx - dw / 2 <= x <= cx + dw / 2:
                    segs.append([[x, cy - dh / 2], [x, cy + dh / 2]])
            for v in range(v_lo, v_hi + 1):
                y = cy + (v - v0) * s
                if cy - dh / 2 <= y <= cy + dh / 2:
                    segs.append([[cx - dw / 2, y], [cx + dw / 2, y]])
            m = np.zeros((H, W), np.uint8)
            cv2.polylines(m, [np.round(np.array(sg) * SS).astype(np.int32).reshape(-1, 1, 2) for sg in segs], False, 255,
                          max(1, int(1.2 * K)), cv2.LINE_AA, SHIFT)
            fr._commit(m, np.zeros(3), 0.7 * ga, "over")
            na = seg(s, 50 * K, 80 * K) * a
            if na > 0:
                fs = s * 0.26 / UNIT
                for u in range(max(0, u_lo), min(IW, u_hi)):
                    for v in range(max(0, v_lo), min(IH, v_hi)):
                        x = cx + (u + 0.5 - u0) * s
                        y = cy + (v + 0.5 - v0) * s
                        if not (cx - dw / 2 + s * 0.4 < x < cx + dw / 2 - s * 0.4 and cy - dh / 2 + s * 0.4 < y < cy + dh / 2 - s * 0.4):
                            continue
                        val = int(self.lum[v, u] * 255)
                        col = np.zeros(3) if val > 150 else WHITE
                        fr.text(str(val), ((x - W / 2) / UNIT, -(y - H / 2) / UNIT), fs, col, na, "num")
        fr.slogan(t, 0.8, 4.6, "一张照片", "就是一个巨大的数字矩阵", size=0.8)
        fr.slogan(t, 5.2, 9.4, "放大，再放大", "每一个像素，都是矩阵里的一个数")

    # -------------------------------------------------------------- SVD rebuild
    def part_svd(self, fr, t):
        a = smooth(seg(t, T_SVD, T_SVD + 0.7))
        dis = seg(t, T_DIS, 22.0)
        dw, dh = 1056 * K, 594 * K
        cx, cy = W / 2 - 250 * K * a - 100 * K * (1 - a), H / 2 - 45 * K
        # current rank
        ki = 0
        for i, (tk, k) in enumerate(RANKS):
            if t >= tk:
                ki = i
        k = RANKS[ki][1]
        img = self.recon[k]
        if t < RANKS[0][0]:
            img = self.img
        else:
            # quick crossfade from previous rank
            if ki > 0:
                x = seg(t, RANKS[ki][0], RANKS[ki][0] + 0.22)
                img = lerp(self.recon[RANKS[ki - 1][1]], img, x)
        ia = a * (1 - smooth(seg(dis, 0, 0.12)))
        if ia > 0:
            self.view(fr, img, cx, cy, dw, dh, IW / 2, IH / 2, dw / IW, ia)
            self.frame_rect(fr, cx, cy, dw, dh, ia)
            # beat pulse on rank change
            if t >= RANKS[0][0]:
                pulse = 1 - seg(t, RANKS[ki][0], RANKS[ki][0] + 0.3)
                if pulse > 0:
                    m = np.zeros((H, W), np.uint8)
                    pts = np.array([[cx - dw / 2, cy - dh / 2], [cx + dw / 2, cy - dh / 2], [cx + dw / 2, cy + dh / 2], [cx - dw / 2, cy + dh / 2]])
                    cv2.polylines(m, [np.round(pts * SS).astype(np.int32).reshape(-1, 1, 2)], True, 255, max(1, int(4 * K)), cv2.LINE_AA, SHIFT)
                    fr._commit(m, GOLD, 1.2 * pulse * ia, "add")
        # particles dissolve
        if dis > 0:
            tt = t - T_DIS
            s = dw / IW
            x = cx + (self.pu - IW / 2) * s
            y = cy + (self.pv - IH / 2) * s
            p0 = np.stack([(x - W / 2) / UNIT, -(y - H / 2) / UNIT], 1)
            xt = (self.pu / IW - 0.5) * 15.2
            p1 = np.stack([xt, chord_wave(xt, 0.0)], 1)
            q = np.clip((tt - self.pdel * 0.6) / 1.3, 0, 1)
            q = q * q * (3 - 2 * q)
            sw = np.sin(np.pi * q)[:, None] * self.pvel * 0.9
            pts = p0 * (1 - q[:, None]) + p1 * q[:, None] + sw
            cols = self.pc * 2.2 * (1 - q[:, None]) + CYAN[None] * 0.9 * q[:, None]
            fr.splat(pts, None, 1.0, screen=True, cols=cols.astype(np.float32))

        # right panel: rank, storage, spectrum
        pa = a * (1 - smooth(seg(dis, 0, 0.3)))
        if pa > 0 and t >= RANKS[0][0] - 0.3:
            pa *= smooth(seg(t, RANKS[0][0] - 0.3, RANKS[0][0] + 0.2))
            px0 = 3.3
            full = k >= 540
            fr.text("秩" if not full else "完整", (px0 + 0.05, 2.85), 0.28, GREY * 1.4, pa, "sansm", anchor="l")
            fr.text(f"k = {k}" if not full else "k = 540", (px0, 2.2), 0.62, GOLD * 1.3, pa, "math", anchor="l")
            store = min(1.0, k * (IW + IH + 1) / (IW * IH))
            fr.text(f"数据量   {store * 100:.1f}%", (px0 + 0.02, 1.45), 0.24, WHITE, pa, "sansm", anchor="l")
            fr.text(f"能量保留 {self.energy[min(k, 539) - 1] * 100:.2f}%", (px0 + 0.02, 1.0), 0.24, CYAN * 1.2, pa, "sansm", anchor="l")
            # spectrum (log scale)
            nb = 64
            sv = np.log10(self.sv[:nb])
            sv = (sv - sv.min()) / (sv.max() - sv.min())
            bx0, bx1, by0, bh = px0, 6.7, -1.9, 2.4
            bw = (bx1 - bx0) / nb
            on = [np.array([[bx0 + i * bw + bw * 0.15, by0], [bx0 + i * bw + bw * 0.15, by0 + 0.05 + sv[i] * bh]]) for i in range(nb) if i < k]
            off = [np.array([[bx0 + i * bw + bw * 0.15, by0], [bx0 + i * bw + bw * 0.15, by0 + 0.05 + sv[i] * bh]]) for i in range(nb) if i >= k]
            fr.polylines(on, GOLD * 1.2, max(2.0, bw * UNIT * 0.6 / K), pa, screen=True)
            fr.polylines(off, GREY, max(2.0, bw * UNIT * 0.6 / K), 0.35 * pa, screen=True)
            fr.text("奇异值 σ", (bx0, by0 - 0.3), 0.2, GREY * 1.3, pa, "sansm", anchor="l")
        # formula
        fa = fade(t, T_SVD + 0.4, T_DIS + 0.3, 0.6, 0.4)
        if fa > 0:
            fr.title("A = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + σ₃u₃v₃ᵀ + ⋯", (-1.95, 3.55), 0.36, t, T_SVD + 0.4, T_DIS + 0.3,
                     color=WHITE, font_key="math", stagger=0.02, track0=0.0, track1=0.02)
        fr.slogan(t, T_SVD + 0.4, 14.3, "奇异值分解 SVD", "把一幅画，拆成一层层最简单的矩阵", size=0.66)
        fr.slogan(t, 14.6, T_DIS + 0.2, "不到 10% 的数据", "就能留住几乎全部画面", size=0.66)
