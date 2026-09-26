"""Cold open: a point of light becomes a vector, the vector becomes a universe of vectors."""
from scenes.common import *

PAL = [CYAN, MAGENTA, GOLD, GREEN, VIOLET, BLUE]


class S(Scene):
    DUR = 12.0

    def setup(self):
        rng = np.random.default_rng(11)
        n = 110
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = rng.uniform(1.2, 7.5, n) ** 1.0
        self.targets = np.stack([np.cos(ang) * rad * 1.25, np.sin(ang) * rad * 0.8], 1)
        self.launch = 7.4 + np.sort(rng.uniform(0, 3.2, n) ** 1.4 / 3.2 ** 0.4)
        self.cols = [PAL[i % len(PAL)] for i in range(n)]
        self.hero = np.array([3.0, 2.0])

    def draw(self, t, fi):
        cam = Cam(zoom=lerp(1.3, 1.0, ease_io(seg(t, 0, 7.5))) * lerp(1.0, 0.78, ease_io(seg(t, 7.2, 11.4))))
        fr = Frame(cam, bg=BLACK * 0 + BG * smooth(seg(t, 0, 3)))
        dust(fr, t, alpha=smooth(seg(t, 0.3, 3.0)))

        # faint grid materialises as the vectors multiply
        ga = 0.10 * smooth(seg(t, 7.6, 10.0)) * (1 - ease_in(seg(t, 11.0, 11.8)))
        if ga > 0:
            segs, axes = grid_segments(np.eye(2), n=14)
            fr.segments(segs, BLUE, 1.2, ga)
        # axes
        ax = ease_out(seg(t, 2.6, 4.2)) * (1 - ease_in(seg(t, 11.0, 11.8)))
        if ax > 0:
            L = 20 * ax
            fr.segments([[[-L, 0], [L, 0]], [[0, -L], [0, L]]], WHITE, 1.5, 0.35 * ax)

        # the swarm of vectors
        shrink = 1 - ease_in(seg(t, 11.0, 11.75))
        for tg, l0, c in zip(self.targets, self.launch, self.cols):
            p = expo_out(seg(t, l0, l0 + 0.7))
            if p <= 0:
                continue
            e = tg * p * shrink
            fr.arrow((0, 0), e, c, 2.2, 0.75, head=0.2, tip_glow=0.25)

        # the hero vector
        ig = smooth(seg(t, 1.2, 2.6))
        pulse = 1 + 0.25 * np.sin(t * 5.0) * (1 - seg(t, 3, 4))
        if t < 3.0:
            fr.glow((0, 0), 16, WHITE, 1.6 * ig * pulse)
            fr.glow((0, 0), 60, CYAN, 0.35 * ig)
        p = expo_out(seg(t, 3.0, 3.75)) * shrink
        if p > 0:
            e = self.hero * p
            fr.arrow((0, 0), e, CYAN * 1.3, 6, 1.0, head=0.42, tip_glow=1.2)
            fr.glow((0, 0), 10, WHITE, 1.0)
            # speed streak while shooting
            st = 1 - seg(t, 3.0, 3.9)
            if st > 0:
                fr.segments([[[0, 0], e]], WHITE, 14, 0.5 * st)
        la = fade(t, 3.9, 10.9, 0.5, 0.6)
        if la > 0:
            tip = cam.to_px(self.hero)
            pos = ((tip[0] - W / 2) / UNIT + 0.75, -(tip[1] - H / 2) / UNIT + 0.35)
            fr.matrix([[3], [2]], pos, 0.3, [CYAN], la, fmt="{:g}")
        # origin collapse glow
        cg = ease_in(seg(t, 10.8, 12.0))
        if cg > 0:
            fr.glow((0, 0), 30, WHITE, 3 * cg)
            fr.glow((0, 0), 120, CYAN, 1.2 * cg)

        fr.slogan(t, 4.3, 7.6, "它只是一支箭头", "VECTOR  ·  向量")
        fr.slogan(t, 8.0, 11.3, "却撑起了整个现代世界", "从游戏画面到人工智能")
        flash = 1.6 * ease_in(seg(t, 11.55, 12.0))
        return fr.finish(bloom=1.0, flash=flash, fi=fi)
