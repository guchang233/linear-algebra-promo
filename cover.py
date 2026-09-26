"""Video cover: 1920x1080 PNG."""
import numpy as np, cv2
import engine as E
from engine import *
from engine import _splat_rgb
from scenes.common import grid_segments
from scenes.s04_fractal import chaos, normalise, FERN, PAL_FERN
from scenes.s09_finale import S as Finale

fin = Finale.__new__(Finale)
rng = np.random.default_rng(99)
fr = Frame(Cam())
# floor grid
Finale.floor(fin, fr, 3.0, 1.0)
# sheared glowing grid behind, faint
segs, axes = grid_segments(np.array([[1.0, 0.45], [0.2, 1.0]]), n=14)
fr.segments(segs, BLUE, 1.2, 0.18)
# fern on the right
A, e, p = normalise(FERN, (6.0, 7.0))
n = 2_000_000
out = np.zeros((n, 2)); idx = np.zeros(n, np.int64)
chaos(A, e, np.cumsum(p), n, 7, out, idx)
acc = np.zeros((H, W, 3), np.float32)
px = Cam(x=-3.7, y=0.1).to_px(out)
_splat_rgb(acc, np.ascontiguousarray(px[:, 0]), np.ascontiguousarray(px[:, 1]),
           np.array(PAL_FERN, np.float32)[idx], 1.0)
L = acc.sum(-1) + 1e-6
fr.img += acc / L[..., None] * np.minimum(np.log1p(L) / 4.0, 1.4)[..., None] * 1.2
# arrows bursting from a point on the left
o = np.array([-6.0, -2.6])
for k in range(40):
    a = rng.uniform(0.1, 1.45)
    r = rng.uniform(1.5, 5.5)
    c = [CYAN, MAGENTA, GOLD, VIOLET][k % 4]
    fr.arrow(o, o + r * np.array([np.cos(a), np.sin(a)]), c, 2.0, 0.28, head=0.2, tip_glow=0.25)
fr.glow(o, 30, WHITE, 2.0)
# title block (left)
fr.band(0.6, 0.8, 5.5)
fr.text("线性代数", (-2.6, 1.05), 1.55, WHITE * 1.15, 1.0, "title")
fr.text("线性代数", (-2.6, 1.05), 1.55, GOLD, 0.25, "title", mode="add")
fr.text("L I N E A R   A L G E B R A", (-2.6, -0.35), 0.3, GOLD * 1.3, 1.0, "latin")
fr.text("看见世界的底层结构", (-2.6, -1.2), 0.46, mix(WHITE, CYAN, 0.3), 1.0, "sans")
fr.matrix([[1, 0], [0, 1]], (-5.0, 2.9), 0.3, [GREEN * 1.2, MAGENTA * 1.2], 0.9, fmt="{:g}", label="A =")
fr.text("A v = λ v", (1.0, 3.3), 0.4, GOLD * 1.2, 0.85, "math")
img = fr.finish(bloom=1.0)
cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile("线性代数宣传片_封面.png")
cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tofile("线性代数宣传片_封面.jpg")
print("ok")
