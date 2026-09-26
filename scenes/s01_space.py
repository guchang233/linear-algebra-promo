"""Matrices bend space: shear, rotate, stretch, then det -> 0 collapses the plane."""
from scenes.common import *

I2 = np.eye(2)
SHEAR = np.array([[1.0, 1.0], [0.0, 1.0]])
A3 = np.array([[1.5, -0.8], [0.6, 1.1]])
SING = np.array([[1.5, 0.75], [0.6, 0.3]])
T_HIT = 17.0


def butterfly(n=900):
    th = np.linspace(0, 24 * np.pi, n * 6)
    r = np.exp(np.sin(th)) - 2 * np.cos(4 * th) + np.sin((2 * th - np.pi) / 24) ** 5
    return np.stack([r * np.sin(th), r * np.cos(th)], 1) * 0.36 + np.array([-3.4, 1.15])


class S(Scene):
    DUR = 24.0

    def setup(self):
        self.bf = butterfly()

    def M(self, t):
        if t < 3.0:
            return I2
        if t < 6.0:
            return lerp(I2, SHEAR, ease_io(seg(t, 3.0, 5.5)))
        if t < 9.0:
            return rot2(np.radians(50) * ease_io(seg(t, 6.0, 8.5))) @ SHEAR
        R = rot2(np.radians(50)) @ SHEAR
        if t < 13.5:
            return lerp(R, A3, ease_io(seg(t, 9.0, 11.5)))
        if t < 19.5:
            return lerp(A3, SING, ease_in(seg(t, 13.5, T_HIT)) ** 0.8)
        e = ease_io(seg(t, 19.5, 22.6))
        return rot2(2 * np.pi * e) @ lerp(SING, I2, e)

    def draw(self, t, fi):
        z = lerp(1.7, 1.0, expo_out(seg(t, 0, 1.4))) * lerp(1.0, 1.08, seg(t, 1.4, 22.5))
        z *= lerp(1.0, 7.0, expo_in(seg(t, 22.6, 24.0)))
        cam = Cam(zoom=z)
        cam.sx, cam.sy = shake_amt(t, T_HIT, 26, 0.7)
        fr = Frame(cam)
        grow = expo_out(seg(t, 0.0, 1.3))
        M = self.M(t) * grow

        # static reference grid
        segs0, _ = grid_segments(I2, n=16)
        fr.segments(segs0, GREY, 1.0, 0.10)
        # transformed grid
        segs, axes = grid_segments(M, n=14)
        hit = np.exp(-max(0.0, t - T_HIT) * 2.5) if t >= T_HIT else 0.0
        fr.segments(segs, BLUE * 1.15, 1.6, 0.55 + 0.8 * hit)
        fr.segments(axes, WHITE, 2.2, 0.55 + 1.5 * hit)
        if hit > 0:
            u = SING[:, 0] / np.linalg.norm(SING[:, 0])
            fr.segments([[-u * 30, u * 30]], GOLD, 7, 1.4 * hit)
            fr.segments([[-u * 30, u * 30]], WHITE, 2.5, 1.5 * hit)

        # determinant parallelogram
        da = fade(t, 11.8, 19.4, 0.6, 0.6)
        if da > 0:
            sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) @ M.T
            fr.fill(sq, GOLD, 0.22 * da)
            fr.polylines([sq], GOLD, 2.5, 0.9 * da, closed=True)
            det = np.linalg.det(M)
            c = sq.mean(0)
            pc = cam.to_px(c)
            sp = ((pc[0] - W / 2) / UNIT, -(pc[1] - H / 2) / UNIT)
            lbl = f"det = {det:.2f}".replace("-", "−") if t < T_HIT else "det = 0"
            col = GOLD if t < T_HIT else mix(RED * 1.4, GOLD, 1 - hit)
            fr.text(lbl, (sp[0] + 1.25, sp[1] + 0.55), 0.3, col * 1.2, da, "math")

        # butterfly riding the transformation
        ba = ease_io(seg(t, 0.6, 2.8))
        if ba > 0:
            pts = self.bf[: max(2, int(len(self.bf) * ba))] @ M.T
            fr.polylines([pts], GOLD, 2.0, 0.85)

        # basis vectors
        fr.arrow((0, 0), M[:, 0], GREEN * 1.2, 5, 1.0, head=0.3, tip_glow=0.6)
        fr.arrow((0, 0), M[:, 1], MAGENTA * 1.2, 5, 1.0, head=0.3, tip_glow=0.6)
        la = seg(t, 1.0, 1.6) * (1 - seg(t, 12, 12.5))
        if la > 0:
            for v, nm, c in ((M[:, 0], "î", GREEN), (M[:, 1], "ĵ", MAGENTA)):
                p = cam.to_px(v * 1.12 + np.array([0.18, 0.18]))
                fr.text(nm, ((p[0] - W / 2) / UNIT, -(p[1] - H / 2) / UNIT), 0.34, c * 1.2, la, "math")

        # matrix panel
        ma = fade(t, 2.4, 22.8, 0.6, 0.6)
        if ma > 0:
            fr.fill(np.array([[3.55, 2.0], [6.75, 2.0], [6.75, 3.75], [3.55, 3.75]]), BLACK[0, 0], 0.55 * ma,
                    screen=True, mode="over")
            Mv = self.M(t)
            fr.matrix(Mv, (5.55, 2.88), 0.3, [GREEN * 1.2, MAGENTA * 1.2], ma, fmt="{:.2f}", label="A =")

        fr.slogan(t, 0.8, 4.8, "矩阵", "让整个空间随之变形", size=0.8)
        fr.slogan(t, 6.0, 10.4, "每一列，都是基向量的新落点", "第一列 → î 去哪了　·　第二列 → ĵ 去哪了")
        fr.slogan(t, 12.0, 16.5, "行列式", "面积被放大了多少倍", size=0.8)
        fr.slogan(t, 17.3, 21.8, "行列式为零", "整个平面，坍缩成一条直线", color=mix(WHITE, GOLD, 0.3))

        flash = flash_in(t) + 0.3 * hit ** 3 + 1.4 * ease_in(seg(t, 23.4, 24.0))
        return fr.finish(bloom=1.0, flash=flash, fi=fi)
