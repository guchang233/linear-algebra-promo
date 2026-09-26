"""Tiny cinematic 2D/3D render engine: float HDR canvas + cv2 AA drawing + numba splats + bloom."""
import os
import math
import subprocess
from functools import lru_cache

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from numba import njit, prange

W = int(os.environ.get("LA_W", "1920"))
H = int(os.environ.get("LA_H", "1080"))
FPS = int(os.environ.get("LA_FPS", "60"))
K = H / 1080.0            # pixel scale relative to 1080p
UNIT = H / 8.0            # pixels per world unit (frame is 8 units tall)
FW = W / UNIT             # frame width in units (~14.22)
SHIFT = 4
SUPPRESS_SLOGAN = False
SS = 1 << SHIFT

FONTS = {
    "title": ("C:/Windows/Fonts/Source Han Serif SC Heavy (TrueType).ttf", 0),
    "sans": ("C:/Windows/Fonts/Noto Sans SC Bold (TrueType).otf", 0),
    "sansm": ("C:/Windows/Fonts/Noto Sans SC Medium (TrueType).otf", 0),
    "math": ("C:/Windows/Fonts/cambriai.ttf", 0),
    "num": ("C:/Windows/Fonts/cambria.ttc", 0),
    "mono": ("C:/Windows/Fonts/consola.ttf", 0),
    "latin": ("C:/Windows/Fonts/segoeuil.ttf", 0),
    "latinb": ("C:/Windows/Fonts/segoeuib.ttf", 0),
}


def hexc(h, gain=1.0):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32) * gain


CYAN = hexc("34e1ff")
MAGENTA = hexc("ff3d8b")
GOLD = hexc("ffc23d")
GREEN = hexc("3dff9a")
VIOLET = hexc("9b6bff")
BLUE = hexc("3d7bff")
ORANGE = hexc("ff7a3d")
RED = hexc("ff4d4d")
WHITE = np.ones(3, np.float32)
GREY = hexc("8a93b8")


def mix(a, b, x):
    return (np.asarray(a, np.float32) * (1 - x) + np.asarray(b, np.float32) * x).astype(np.float32)


def hsv(h, s=1.0, v=1.0):
    """h in [0,1) -> rgb float array"""
    h = (h % 1.0) * 6
    i = int(h)
    f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return np.array([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6], np.float32)


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return a if x < a else (b if x > b else x)


def seg(t, a, b):
    if b <= a:
        return 1.0 if t >= a else 0.0
    return clamp((t - a) / (b - a))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_io(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp(x)
    return x ** 3


def expo_out(x):
    x = clamp(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def expo_in(x):
    x = clamp(x)
    return 0.0 if x <= 0 else 2 ** (10 * x - 10)


def back_out(x, s=1.70158):
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def lerp(a, b, x):
    return a + (b - a) * x


def fade(t, a, b, fin=0.4, fout=0.4):
    return min(smooth(seg(t, a, a + fin)), 1 - smooth(seg(t, b - fout, b)))


def rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def rot3(ax, ay, az):
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------- numba kernels
@njit(parallel=True, fastmath=True, cache=True)
def _add_mask(img, m, r, g, b):
    h, w = m.shape
    for y in prange(h):
        for x in range(w):
            v = m[y, x]
            if v:
                f = v * (1.0 / 255.0)
                img[y, x, 0] += f * r
                img[y, x, 1] += f * g
                img[y, x, 2] += f * b


@njit(parallel=True, fastmath=True, cache=True)
def _over_mask(img, m, r, g, b, a):
    h, w = m.shape
    for y in prange(h):
        for x in range(w):
            v = m[y, x]
            if v:
                f = v * (a / 255.0)
                img[y, x, 0] = img[y, x, 0] * (1 - f) + r * f
                img[y, x, 1] = img[y, x, 1] * (1 - f) + g * f
                img[y, x, 2] = img[y, x, 2] * (1 - f) + b * f


@njit(fastmath=True, cache=True)
def _splat_rgb(img, xs, ys, cols, w):
    h, wd = img.shape[0], img.shape[1]
    for i in range(xs.shape[0]):
        x = xs[i] - 0.5
        y = ys[i] - 0.5
        if not (x >= 0 and y >= 0 and x < wd - 1 and y < h - 1):
            continue
        x0 = int(x)
        y0 = int(y)
        fx = x - x0
        fy = y - y0
        w00 = (1 - fx) * (1 - fy) * w
        w10 = fx * (1 - fy) * w
        w01 = (1 - fx) * fy * w
        w11 = fx * fy * w
        for c in range(3):
            v = cols[i, c]
            img[y0, x0, c] += v * w00
            img[y0, x0 + 1, c] += v * w10
            img[y0 + 1, x0, c] += v * w01
            img[y0 + 1, x0 + 1, c] += v * w11


@njit(fastmath=True, cache=True)
def _splat_1(img, xs, ys, r, g, b, w):
    h, wd = img.shape[0], img.shape[1]
    for i in range(xs.shape[0]):
        x = xs[i] - 0.5
        y = ys[i] - 0.5
        if not (x >= 0 and y >= 0 and x < wd - 1 and y < h - 1):
            continue
        x0 = int(x)
        y0 = int(y)
        fx = x - x0
        fy = y - y0
        w00 = (1 - fx) * (1 - fy) * w
        w10 = fx * (1 - fy) * w
        w01 = (1 - fx) * fy * w
        w11 = fx * fy * w
        img[y0, x0, 0] += r * w00
        img[y0, x0, 1] += g * w00
        img[y0, x0, 2] += b * w00
        img[y0, x0 + 1, 0] += r * w10
        img[y0, x0 + 1, 1] += g * w10
        img[y0, x0 + 1, 2] += b * w10
        img[y0 + 1, x0, 0] += r * w01
        img[y0 + 1, x0, 1] += g * w01
        img[y0 + 1, x0, 2] += b * w01
        img[y0 + 1, x0 + 1, 0] += r * w11
        img[y0 + 1, x0 + 1, 1] += g * w11
        img[y0 + 1, x0 + 1, 2] += b * w11


@njit(fastmath=True, cache=True)
def _splat_density(d, xs, ys, ws):
    h, wd = d.shape
    for i in range(xs.shape[0]):
        x = xs[i] - 0.5
        y = ys[i] - 0.5
        if not (x >= 0 and y >= 0 and x < wd - 1 and y < h - 1):
            continue
        x0 = int(x)
        y0 = int(y)
        fx = x - x0
        fy = y - y0
        w = ws[i]
        d[y0, x0] += (1 - fx) * (1 - fy) * w
        d[y0, x0 + 1] += fx * (1 - fy) * w
        d[y0 + 1, x0] += (1 - fx) * fy * w
        d[y0 + 1, x0 + 1] += fx * fy * w


@njit(parallel=True, fastmath=True, cache=True)
def _finish(img, bl, vig, grain, exposure, flash, out):
    h, w = vig.shape
    for y in prange(h):
        for x in range(w):
            v = vig[y, x]
            gr = grain[y, x]
            r = (img[y, x, 0] + bl[y, x, 0] + flash) * exposure * v
            g = (img[y, x, 1] + bl[y, x, 1] + flash) * exposure * v
            b = (img[y, x, 2] + bl[y, x, 2] + flash) * exposure * v
            mx = max(r, max(g, b))
            spill = 0.0
            if mx > 1.0:
                spill = (mx - 1.0) * 0.35
            c3 = (r + spill, g + spill, b + spill)
            for c in range(3):
                u = c3[c]
                if u < 0:
                    u = 0.0
                if u > 0.78:
                    u = 0.78 + 0.22 * (1.0 - math.exp(-(u - 0.78) / 0.22))
                u = u + gr
                u = u * 255.0 + 0.5
                if u < 0:
                    u = 0.0
                if u > 255:
                    u = 255.0
                out[y, x, c] = np.uint8(u)


# ---------------------------------------------------------------- static buffers
def _make_bg():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    c0 = hexc("0b1024")
    c1 = hexc("020308")
    t = np.clip(r / 1.3, 0, 1)[..., None]
    return (c0 * (1 - t) + c1 * t).astype(np.float32)


BG = _make_bg()
BLACK = np.zeros((H, W, 3), np.float32)
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_rr = np.sqrt(((_xx - W / 2) / (W / 2)) ** 2 * 0.8 + ((_yy - H / 2) / (H / 2)) ** 2)
VIG = (1.0 - 0.38 * np.clip(_rr - 0.35, 0, None) ** 1.6).astype(np.float32)
_rng = np.random.default_rng(7)
GRAIN = [(_rng.standard_normal((H, W)).astype(np.float32) * 0.012) for _ in range(6)]
del _yy, _xx, _rr


def gauss_sprite(radius_px):
    r = max(2, int(radius_px * 3))
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    return np.exp(-(x * x + y * y) / (2 * radius_px * radius_px)).astype(np.float32)


@lru_cache(maxsize=64)
def _sprite(radius_q):
    return gauss_sprite(radius_q / 4.0)


# ---------------------------------------------------------------- camera
class Cam:
    def __init__(self, x=0.0, y=0.0, zoom=1.0, rot=0.0):
        self.x, self.y, self.zoom, self.rot = x, y, zoom, rot
        self.sx, self.sy = 0.0, 0.0  # shake, px

    def to_px(self, P):
        P = np.asarray(P, np.float64)
        x = P[..., 0] - self.x
        y = P[..., 1] - self.y
        if self.rot:
            c, s = math.cos(self.rot), math.sin(self.rot)
            x, y = c * x - s * y, s * x + c * y
        s = self.zoom * UNIT
        return np.stack([W / 2 + x * s + self.sx, H / 2 - y * s + self.sy], -1)

    @property
    def scale(self):
        return self.zoom * UNIT


SCREEN = Cam()


def shake_amt(t, t_hit, amp=18.0, dur=0.5, seed=0):
    if t < t_hit or t > t_hit + dur:
        return 0.0, 0.0
    k = (1 - (t - t_hit) / dur) ** 2
    rng = np.random.default_rng(int((t - t_hit) * 60) + seed * 1000)
    return tuple(rng.uniform(-1, 1, 2) * amp * K * k)


# ---------------------------------------------------------------- fonts / text
@lru_cache(maxsize=128)
def font(key, size):
    path, idx = FONTS[key]
    return ImageFont.truetype(path, max(4, int(size)), index=idx)


@lru_cache(maxsize=16)
def _cmap(key):
    from fontTools.ttLib import TTFont
    path, idx = FONTS[key]
    tt = TTFont(path, fontNumber=idx, lazy=True) if path.endswith(".ttc") else TTFont(path, lazy=True)
    return set(tt.getBestCmap().keys())


FALLBACK = ["sans", "num", "math"]


def glyph_font(key, ch):
    if ch == " " or ord(ch) in _cmap(key):
        return key
    for k in FALLBACK:
        if ord(ch) in _cmap(k):
            return k
    return key


def str_font(key, s):
    for ch in s:
        if glyph_font(key, ch) != key:
            return glyph_font(key, ch) if len(s) == 1 else "sans"
    return key


@lru_cache(maxsize=8192)
def text_mask(s, key, size):
    """Returns float mask (h,w) whose logical centre (advance box, ascent+descent) is its centre."""
    f = font(key, size)
    asc, desc = f.getmetrics()
    adv = f.getlength(s)
    pad = int(size * 0.35) + 2
    w = int(math.ceil(adv)) + 2 * pad
    h = asc + desc + 2 * pad
    im = Image.new("L", (max(w, 1), max(h, 1)), 0)
    ImageDraw.Draw(im).text((pad, pad), s, font=f, fill=255, anchor="la")
    return np.asarray(im, np.float32) * (1 / 255.0), adv


def blit(img, m, cx, cy, color, alpha=1.0, mode="over"):
    """Composite float mask m centred at pixel (cx,cy)."""
    if alpha <= 0.002:
        return
    h, w = m.shape
    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(W, x0 + w), min(H, y0 + h)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    mm = m[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0][..., None] * alpha
    reg = img[iy0:iy1, ix0:ix1]
    col = np.asarray(color, np.float32)
    if mode == "over":
        reg *= (1 - mm)
        reg += mm * col
    else:
        reg += mm * col


def blit_rgb(img, src, x0, y0, alpha=1.0, mode="over"):
    """Paste float rgb src (h,w,3) with top-left at integer pixel (x0,y0)."""
    h, w = src.shape[:2]
    x0, y0 = int(round(x0)), int(round(y0))
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(W, x0 + w), min(H, y0 + h)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    s = src[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    if mode == "over":
        img[iy0:iy1, ix0:ix1] = img[iy0:iy1, ix0:ix1] * (1 - alpha) + s * alpha
    else:
        img[iy0:iy1, ix0:ix1] += s * alpha


# ---------------------------------------------------------------- frame
class Frame:
    def __init__(self, cam=None, bg=None):
        self.img = (BG if bg is None else bg).copy()
        self.cam = cam or Cam()

    def P(self, pts, screen=False):
        return (SCREEN if screen else self.cam).to_px(pts)

    # --- mask helpers
    @staticmethod
    def _ip(p):
        p = np.clip(np.asarray(p, np.float64), -30000, 30000)
        return np.round(p * SS).astype(np.int32)

    def _commit(self, m, color, alpha, mode):
        c = np.asarray(color, np.float32) * (alpha if mode == "add" else 1.0)
        if mode == "add":
            _add_mask(self.img, m, float(c[0]), float(c[1]), float(c[2]))
        else:
            _over_mask(self.img, m, float(c[0]), float(c[1]), float(c[2]), float(alpha))

    def polylines(self, polys, color, width=2.0, alpha=1.0, closed=False, screen=False, mode="add"):
        """polys: list of (n,2) world arrays OR array (N,n,2)."""
        if alpha <= 0.002 or len(polys) == 0:
            return
        m = np.zeros((H, W), np.uint8)
        arrs = [self._ip(self.P(p, screen)).reshape(-1, 1, 2) for p in polys]
        cv2.polylines(m, arrs, closed, 255, max(1, int(round(width * K))), cv2.LINE_AA, SHIFT)
        self._commit(m, color, alpha, mode)

    def segments(self, segs, color, width=2.0, alpha=1.0, screen=False, mode="add"):
        """segs: (N,2,2) world."""
        segs = np.asarray(segs)
        if alpha <= 0.002 or len(segs) == 0:
            return
        m = np.zeros((H, W), np.uint8)
        ip = self._ip(self.P(segs, screen)).reshape(-1, 2, 1, 2)
        cv2.polylines(m, list(ip), False, 255, max(1, int(round(width * K))), cv2.LINE_AA, SHIFT)
        self._commit(m, color, alpha, mode)

    def segments_graded(self, segs, color, alphas, width=2.0, bins=8, screen=False, cols=None):
        """Segments with per-segment brightness (binned). cols: optional per-segment colour (N,3)."""
        segs = np.asarray(segs)
        alphas = np.asarray(alphas)
        if len(segs) == 0:
            return
        amax = alphas.max()
        if amax <= 0.002:
            return
        edges = np.linspace(0, amax, bins + 1)
        idx = np.clip(np.digitize(alphas, edges) - 1, 0, bins - 1)
        for b in range(bins):
            sel = idx == b
            if not sel.any():
                continue
            a = (edges[b] + edges[b + 1]) / 2
            c = color if cols is None else cols[sel].mean(0)
            self.segments(segs[sel], c, width, a, screen)

    def fill(self, pts, color, alpha=1.0, screen=False, mode="add"):
        if alpha <= 0.002:
            return
        m = np.zeros((H, W), np.uint8)
        polys = pts if isinstance(pts, list) else [pts]
        cv2.fillPoly(m, [self._ip(self.P(p, screen)).reshape(-1, 1, 2) for p in polys], 255, cv2.LINE_AA, SHIFT)
        self._commit(m, color, alpha, mode)

    def circle(self, c, r, color, width=2.0, alpha=1.0, fill=False, screen=False, mode="add"):
        if alpha <= 0.002:
            return
        cam = SCREEN if screen else self.cam
        m = np.zeros((H, W), np.uint8)
        p = self._ip(cam.to_px(c))
        rr = int(round(r * cam.scale * SS))
        if rr <= 0:
            return
        cv2.circle(m, (int(p[0]), int(p[1])), rr, 255, -1 if fill else max(1, int(round(width * K))), cv2.LINE_AA, SHIFT)
        self._commit(m, color, alpha, mode)

    def circles(self, cs, rs, color, width=1.5, alpha=1.0, screen=False):
        if alpha <= 0.002:
            return
        cam = SCREEN if screen else self.cam
        m = np.zeros((H, W), np.uint8)
        ps = self._ip(cam.to_px(cs))
        th = max(1, int(round(width * K)))
        for p, r in zip(ps, rs):
            rr = int(round(r * cam.scale * SS))
            if rr > 2:
                cv2.circle(m, (int(p[0]), int(p[1])), rr, 255, th, cv2.LINE_AA, SHIFT)
        self._commit(m, color, alpha, "add")

    def arrow(self, a, b, color, width=4.0, alpha=1.0, head=0.28, screen=False, mode="add", tip_glow=0.0):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        d = b - a
        L = float(np.hypot(*d))
        if L < 1e-4 or alpha <= 0.002:
            return
        u = d / L
        n = np.array([-u[1], u[0]])
        hl = min(head, L * 0.45)
        base = b - u * hl
        m = np.zeros((H, W), np.uint8)
        cam = SCREEN if screen else self.cam
        pa, pbase = self._ip(cam.to_px(a)), self._ip(cam.to_px(base + u * hl * 0.15))
        cv2.line(m, tuple(int(v) for v in pa), tuple(int(v) for v in pbase), 255, max(1, int(round(width * K))), cv2.LINE_AA, SHIFT)
        tri = np.array([b, base + n * hl * 0.52, base - n * hl * 0.52])
        cv2.fillPoly(m, [self._ip(cam.to_px(tri)).reshape(-1, 1, 2)], 255, cv2.LINE_AA, SHIFT)
        self._commit(m, color, alpha, mode)
        if tip_glow > 0:
            self.glow(b, 10, color, tip_glow, screen=screen)

    def glow(self, c, radius_px, color, intensity=1.0, screen=False):
        """Additive gaussian sprite (radius in 1080p px)."""
        if intensity <= 0.002:
            return
        cam = SCREEN if screen else self.cam
        p = cam.to_px(c)
        sp = _sprite(max(4, int(radius_px * K * 4)))
        blit(self.img, sp, p[0], p[1], np.asarray(color, np.float32), intensity, mode="add")

    def splat(self, pts, color, weight=1.0, screen=False, cols=None):
        """Additive bilinear points. pts world (N,2). cols optional (N,3)."""
        if len(pts) == 0:
            return
        p = self.P(pts, screen)
        xs = np.ascontiguousarray(p[:, 0], np.float64)
        ys = np.ascontiguousarray(p[:, 1], np.float64)
        if cols is None:
            c = np.asarray(color, np.float32)
            _splat_1(self.img, xs, ys, float(c[0]), float(c[1]), float(c[2]), float(weight))
        else:
            _splat_rgb(self.img, xs, ys, np.ascontiguousarray(cols, np.float32), float(weight))

    # --- text
    def text(self, s, pos, size, color=WHITE, alpha=1.0, font_key="sans", screen=True, mode="over",
             anchor="c", scale=1.0):
        if alpha <= 0.002 or not s:
            return
        size_px = int(round(size * UNIT * scale))
        m, adv = text_mask(s, str_font(font_key, s), size_px)
        p = self.P(pos, screen)
        cx, cy = p[0], p[1]
        if anchor == "l":
            cx += adv / 2
        elif anchor == "r":
            cx -= adv / 2
        blit(self.img, m, cx, cy, color, alpha, mode)

    def text_width(self, s, size, font_key="sans"):
        return font(str_font(font_key, s), int(round(size * UNIT))).getlength(s) / UNIT

    def title(self, s, pos, size, t, t0, t1, color=WHITE, font_key="title", stagger=0.07, fin=0.6,
              fout=0.5, rise=0.35, track0=0.02, track1=0.12, mode="over", glow=0.0, screen=True):
        """Cinematic per-character reveal with slow tracking expansion."""
        if t < t0 or t > t1:
            return
        size_px = int(round(size * UNIT))
        f = font(font_key, size_px)
        chars = list(s)
        fks = [glyph_font(font_key, c) for c in chars]
        widths = [font(k, size_px).getlength(c) for k, c in zip(fks, chars)]
        trk = lerp(track0, track1, ease_out(seg(t, t0, t1))) * size_px
        total = sum(widths) + trk * (len(chars) - 1)
        p = self.P(pos, screen)
        x = p[0] - total / 2
        out = 1 - smooth(seg(t, t1 - fout, t1))
        for i, c in enumerate(chars):
            a0 = t0 + i * stagger
            pr = ease_out(seg(t, a0, a0 + fin))
            a = pr * out
            cx = x + widths[i] / 2
            x += widths[i] + trk
            if c == " " or a <= 0.003:
                continue
            m, _ = text_mask(c, fks[i], size_px)
            cy = p[1] + (1 - pr) * rise * size_px
            blit(self.img, m, cx, cy, color, a, mode)
            if glow > 0:
                blit(self.img, m, cx, cy, color, a * glow, "add")

    def slogan(self, t, t0, t1, main, sub=None, y=-2.72, size=0.62, sub_size=0.27, color=WHITE,
               sub_color=None, band=True, x=0.0):
        """Big main line + small sub line, bottom area, with darkening band for legibility."""
        if t < t0 or t > t1 or SUPPRESS_SLOGAN:
            return
        a = fade(t, t0, t1, 0.5, 0.5)
        if band:
            self.band(y, a)
        if main:
            self.title(main, (x, y + (0.18 if sub else 0)), size, t, t0, t1, color=color, glow=0.15)
        if sub:
            sc = GREY * 1.3 if sub_color is None else sub_color
            self.title(sub, (x, y - size * 0.72), sub_size, t, t0 + 0.25, t1, color=sc, font_key="sansm",
                       stagger=0.025, fin=0.5, rise=0.2, track0=0.08, track1=0.2)

    def band(self, y, a, height=2.4):
        """Darken a horizontal band centred on world y (screen coords)."""
        cy = SCREEN.to_px((0, y))[1]
        hh = height * UNIT / 2
        y0, y1 = int(max(0, cy - hh)), int(min(H, cy + hh))
        if y1 <= y0:
            return
        prof = np.cos(np.linspace(-np.pi / 2, np.pi / 2, y1 - y0)).astype(np.float32) ** 1.5
        self.img[y0:y1] *= (1 - 0.6 * a * prof)[:, None, None]

    def matrix(self, M, pos, size=0.34, col_colors=None, alpha=1.0, fmt="{:.2f}", label=None,
               label_color=WHITE, screen=True, cell=None, bracket_color=None):
        """Draw a bracketed matrix. M: 2D list/array of numbers or strings. pos = centre."""
        if alpha <= 0.002:
            return
        M = [list(r) for r in M]
        rows, cols = len(M), len(M[0])
        strs = [[(v if isinstance(v, str) else fmt.format(v).replace("-", "−")) for v in r] for r in M]
        if cell is None:
            cw = max(self.text_width(s, size, "num") for r in strs for s in r) + size * 0.9
        else:
            cw = cell
        rh = size * 1.45
        wtot = cw * cols
        htot = rh * rows
        cx, cy = pos
        if label:
            lw = self.text_width(label, size * 1.1, "math")
            self.text(label, (cx - wtot / 2 - 0.25 - lw / 2, cy), size * 1.1, label_color, alpha, "math", screen)
        for i in range(rows):
            for j in range(cols):
                col = WHITE if col_colors is None else col_colors[j]
                x = cx - wtot / 2 + cw * (j + 0.5)
                y = cy + htot / 2 - rh * (i + 0.5)
                self.text(strs[i][j], (x, y), size, col, alpha, "num", screen)
        bc = WHITE * 0.85 if bracket_color is None else bracket_color
        bx0 = cx - wtot / 2 - 0.05
        bx1 = cx + wtot / 2 + 0.05
        yt, yb = cy + htot / 2 + 0.05, cy - htot / 2 - 0.05
        tick = 0.12
        polys = [np.array([[bx0 + tick, yt], [bx0, yt], [bx0, yb], [bx0 + tick, yb]]),
                 np.array([[bx1 - tick, yt], [bx1, yt], [bx1, yb], [bx1 - tick, yb]])]
        self.polylines(polys, bc, 2.2, alpha, screen=screen, mode="over")
        return wtot, htot

    # --- whole-frame ops
    def darken(self, a):
        if a > 0:
            self.img *= (1 - a)

    def finish(self, bloom=0.9, thresh=0.5, exposure=1.0, flash=0.0, wide=0.35, grain=True, fi=0):
        img = self.img
        b = np.maximum(img - thresh, 0)
        q = cv2.resize(b, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(q, (0, 0), 1.5 * K) * 0.55 + cv2.GaussianBlur(q, (0, 0), 5 * K) * 0.45
        e = cv2.resize(q, (W // 16, H // 16), interpolation=cv2.INTER_AREA)
        e = cv2.GaussianBlur(e, (0, 0), 3 * K)
        g += cv2.resize(e, (W // 4, H // 4), interpolation=cv2.INTER_LINEAR) * wide
        bl = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR) * bloom
        out = np.empty((H, W, 3), np.uint8)
        gr = GRAIN[fi % len(GRAIN)] if grain else np.zeros((H, W), np.float32)
        _finish(img, np.ascontiguousarray(bl, np.float32), VIG, gr, float(exposure), float(flash), out)
        return out


# ---------------------------------------------------------------- 3D helpers
def project(P3, dist=8.0, fov_scale=1.0):
    """Perspective project camera-space points (x right, y up, z towards viewer). Returns (N,2) world & depth factor."""
    z = dist - P3[..., 2]
    z = np.maximum(z, 0.05)
    f = dist * fov_scale / z
    return np.stack([P3[..., 0] * f, P3[..., 1] * f], -1), f


# ---------------------------------------------------------------- video writer
class Writer:
    def __init__(self, path, w=None, h=None, fps=None):
        w, h, fps = w or W, h or H, fps or FPS
        enc = os.environ.get("LA_ENC", "nvenc")
        if enc == "nvenc":
            vc = ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "16", "-b:v", "0",
                  "-maxrate", "60M", "-bufsize", "120M"]
        else:
            vc = ["-c:v", "libx264", "-preset", "medium", "-crf", "16"]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(fps), "-i", "-"] + vc + ["-pix_fmt", "yuv420p", path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_u8):
        self.p.stdin.write(np.ascontiguousarray(frame_u8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


class Scene:
    """Subclass: set DUR, implement setup(), update(t, dt) (optional state), draw(t) -> uint8 frame."""
    DUR = 10.0

    def setup(self):
        pass

    def update(self, t, dt):
        pass

    def draw(self, t, fi):
        raise NotImplementedError
