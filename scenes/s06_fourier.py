"""Fourier: a chord splits into pure tones; then epicycles draw the character 美."""
from PIL import Image, ImageDraw
from scenes.common import *

KS = (2.2, 2.75, 3.3)
WS = (5.0, 6.25, 7.5)
PH = (0.0, 0.7, 1.9)
COMP_COL = [GOLD, MAGENTA, GREEN]
NOTES = ["Do  262 Hz", "Mi  330 Hz", "Sol  392 Hz"]
T_EPI = 8.8
T_TRACE0, T_TRACE1 = 9.6, 19.3
NTERMS = 700
NDRAW = 110


def glyph_path(ch="美", size=1000, n=4096):
    f = font("title", size)
    im = Image.new("L", (int(size * 1.3), int(size * 1.3)), 0)
    ImageDraw.Draw(im).text((size * 0.65, size * 0.65), ch, font=f, fill=255, anchor="mm")
    a = np.array(im)
    cs, hier = cv2.findContours(a, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cs = [c[:, 0, :].astype(float) for c in cs if len(c) > 40]
    # greedy ordering to shorten pen-up jumps
    order = [max(range(len(cs)), key=lambda i: cs[i][:, 0].min() * -1)]
    rest = set(range(len(cs))) - set(order)
    pts, pen = [], []
    cur = cs[order[0]]
    path = [cur]
    while rest:
        end = path[-1][-1]
        best, bi, bj = 1e18, None, 0
        for i in rest:
            d = np.sum((cs[i] - end) ** 2, 1)
            j = int(np.argmin(d))
            if d[j] < best:
                best, bi, bj = d[j], i, j
        c = np.roll(cs[bi], -bj, 0)
        c = np.concatenate([c, c[:1]])
        path.append(c)
        rest.remove(bi)
    path[0] = np.concatenate([path[0], path[0][:1]])
    # stitch with explicit jump segments
    for i, c in enumerate(path):
        if i > 0:
            a0, a1 = path[i - 1][-1], c[0]
            m = max(2, int(np.hypot(*(a1 - a0)) / 3))
            jump = np.linspace(a0, a1, m)[1:-1]
            pts.append(jump)
            pen.append(np.zeros(len(jump), bool))
        pts.append(c)
        pen.append(np.ones(len(c), bool))
    a0, a1 = path[-1][-1], path[0][0]
    m = max(2, int(np.hypot(*(a1 - a0)) / 3))
    pts.append(np.linspace(a0, a1, m)[1:-1])
    pen.append(np.zeros(m - 2, bool))
    P = np.concatenate(pts)
    PEN = np.concatenate(pen)
    # arc-length resample
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    s = np.linspace(0, d[-1], n, endpoint=False)
    x = np.interp(s, d, P[:, 0])
    y = np.interp(s, d, P[:, 1])
    pen_s = PEN[np.clip(np.searchsorted(d, s), 0, len(PEN) - 1)]
    z = (x - size * 0.65) - 1j * (y - size * 0.65)
    z = z / size * 5.4 + (0 + 0.62j)
    return z, pen_s, [np.stack([(c[:, 0] - size * 0.65) / size * 5.4, -(c[:, 1] - size * 0.65) / size * 5.4 + 0.62], 1) for c in cs]


class S(Scene):
    DUR = 22.0

    def setup(self):
        z, self.pen, self.contours = glyph_path()
        n = len(z)
        C = np.fft.fft(z) / n
        freqs = np.fft.fftfreq(n, 1 / n)
        order = np.argsort(-np.abs(C))
        # keep DC first, then by magnitude
        order = [o for o in order if freqs[o] != 0]
        dc = C[freqs == 0][0]
        self.dc = dc
        self.C = C[order][:NTERMS]
        self.F = freqs[order][:NTERMS]
        self.n = n
        th = 2 * np.pi * np.arange(n) / n
        self.recon = dc + (self.C[None, :] * np.exp(1j * np.outer(th, self.F))).sum(1)

    def draw(self, t, fi):
        fr = Frame(Cam())
        dust(fr, t, n=350, alpha=0.6)
        if t < T_EPI + 0.3:
            self.part_wave(fr, t)
        if t >= T_EPI:
            self.part_epi(fr, t)
        flash = flash_in(t, 0.3, 0.6) + 1.4 * ease_in(seg(t, 21.45, 22.0))
        return fr.finish(bloom=1.0, flash=flash, fi=fi)

    def part_wave(self, fr, t):
        out = 1 - smooth(seg(t, T_EPI - 0.9, T_EPI))
        x = np.linspace(-7.6, 7.6, 900)
        comb = chord_wave(x, t)
        e = ease_io(seg(t, 2.6, 4.3))
        comb_y = lerp(0, 2.35, e) + comb * lerp(1.0, 0.6, e)
        fr.polylines([np.stack([x, comb_y], 1)], CYAN * 1.3, 5, 0.35 * out)
        fr.polylines([np.stack([x, comb_y], 1)], mix(CYAN, WHITE, 0.4) * 1.2, 2.2, out)
        if e > 0:
            for i in range(3):
                yi = 0.95 - i * 1.2
                comp = yi + 0.42 * np.sin(KS[i] * x - WS[i] * t + PH[i])
                y = lerp(comb_y, comp, e)
                fr.polylines([np.stack([x, y], 1)], COMP_COL[i], 4.5, 0.3 * out * e)
                fr.polylines([np.stack([x, y], 1)], COMP_COL[i] * 1.2, 2.0, out * e)
                la = seg(t, 3.9, 4.5) * out
                fr.band(yi, 0.5 * la, 0.6) if False else None
                fr.text(NOTES[i], (-5.9, yi + 0.5), 0.22, COMP_COL[i] * 1.3, la, "sansm")
            fa = seg(t, 4.0, 4.6) * out
            fr.text("f(t) = sin ω₁t + sin ω₂t + sin ω₃t", (0, 3.55), 0.3, WHITE, fa, "math")
        fr.slogan(t, 0.6, 4.3, "一段和弦", "其实是三个纯音的叠加", size=0.8, y=-2.85)
        fr.slogan(t, 4.6, 8.4, "傅里叶变换", "换一组基底，声音显露出它的本质", size=0.8, y=-2.85)

    def part_epi(self, fr, t):
        a = smooth(seg(t, T_EPI, T_EPI + 0.8))
        p = ease_io(seg(t, T_TRACE0, T_TRACE1)) if t < T_TRACE1 else 1.0
        th = 2 * np.pi * p
        fill = smooth(seg(t, T_TRACE1 + 0.1, T_TRACE1 + 1.2))
        circ_a = a * (1 - fill)
        # epicycle chain
        terms = self.C[:NDRAW] * np.exp(1j * self.F[:NDRAW] * th)
        centres = self.dc + np.r_[0, np.cumsum(terms)]
        tip = self.recon[int(p * (self.n - 1))] if p < 1 else self.recon[-1]
        if circ_a > 0:
            cs = np.stack([centres[:-1].real, centres[:-1].imag], 1)
            rs = np.abs(self.C[:NDRAW])
            fr.circles(cs[:40], rs[:40], BLUE * 1.1, 1.4, 0.55 * circ_a)
            fr.circles(cs[40:], rs[40:], BLUE, 1.0, 0.35 * circ_a)
            arm = np.stack([centres.real, centres.imag], 1)
            arm = np.vstack([arm, [[tip.real, tip.imag]]])
            fr.polylines([arm], WHITE, 1.6, 0.8 * circ_a)
            fr.glow((tip.real, tip.imag), 10, GOLD, 1.8 * circ_a)
            fr.glow((tip.real, tip.imag), 4, WHITE, 1.5 * circ_a)
        # traced path (pen-up segments skipped)
        m = int(p * (self.n - 1))
        if m > 2:
            pts = np.stack([self.recon[:m + 1].real, self.recon[:m + 1].imag], 1)
            pen = self.pen[:m + 1]
            runs, cur = [], []
            for i in range(m + 1):
                if pen[i]:
                    cur.append(pts[i])
                elif cur:
                    runs.append(np.array(cur))
                    cur = []
            if cur:
                runs.append(np.array(cur))
            runs = [r for r in runs if len(r) > 1]
            fr.polylines(runs, GOLD * 1.1, 6, 0.25 * a)
            fr.polylines(runs, mix(GOLD, WHITE, 0.35) * 1.2, 2.4, a)
        if fill > 0:
            fr.fill(self.contours, GOLD * 1.25, 0.9 * fill)
            fr.glow((0, 0.62), 260, GOLD, 0.25 * fill)
        fr.slogan(t, 9.8, 14.4, "傅里叶级数", "用旋转的圆，画出任何形状", size=0.75)
        fr.slogan(t, 14.8, 19.2, "每一个圆，都是一个基向量", "这是无穷维空间里的线性代数")
        fr.slogan(t, 19.9, 21.9, "这，就是线性代数之美", None, size=0.66)
