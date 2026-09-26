"""Synthesised soundtrack, locked to the visual timeline (120 BPM, D minor)."""
import numpy as np
from scipy import signal
from scipy.io import wavfile

from timeline import TOTAL

SR = 44100
TAIL = 4.0
N = int((TOTAL + TAIL) * SR)
rng = np.random.default_rng(1)
BEAT = 0.5


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(d):
    return np.arange(int(d * SR)) / SR


def sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output="sos")


def filt(x, kind, f, order=2):
    return signal.sosfilt(sos(kind, f, order), x)


class Bus:
    def __init__(self):
        self.b = np.zeros((2, N), np.float32)

    def add(self, x, t0, gain=1.0, pan=0.0):
        i = int(round(t0 * SR))
        if i >= N:
            return
        if x.ndim == 1:
            a = (pan + 1) * np.pi / 4
            x = np.stack([x * np.cos(a), x * np.sin(a)]) * np.sqrt(2)
        if i < 0:
            x = x[:, -i:]
            i = 0
        n = min(x.shape[1], N - i)
        self.b[:, i:i + n] += (x[:, :n] * gain).astype(np.float32)


def in_win(t, wins):
    return any(a <= t < b for a, b in wins)


# ------------------------------------------------------------------ instruments
def kick():
    t = tt(0.5)
    f = 44 + 120 * np.exp(-t * 32)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6.5)
    y += rng.standard_normal(len(t)) * np.exp(-t * 280) * 0.25
    return np.tanh(1.6 * y) * 0.9


def snare():
    t = tt(0.45)
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 22) * 0.45
    nz = filt(rng.standard_normal(len(t)), "bandpass", [1200, 7000]) * np.exp(-t * 11)
    return (tone + nz * 0.9) * 0.8


def hat(open_=False):
    t = tt(0.3 if open_ else 0.06)
    nz = filt(rng.standard_normal(len(t)), "highpass", 7500, 4)
    return nz * np.exp(-t * (14 if open_ else 70)) * 0.35


def saw(f, t, detune=0.0, ph=0.0):
    return signal.sawtooth(2 * np.pi * f * (1 + detune) * t + ph)


def bass_note(m, d):
    t = tt(d)
    f = mtof(m)
    y = saw(f, t) * 0.6 + saw(f, t, 0.004) * 0.4 + np.sin(2 * np.pi * f * t) * 0.9
    y = filt(y, "lowpass", 520, 2)
    env = np.minimum(1, t / 0.004) * (0.55 + 0.45 * np.exp(-t * 9)) * np.minimum(1, (d - t) / 0.02)
    return y * env * 0.55


def pad(ms, d, att=0.8, rel=1.6, cutoff=1600, bright=0.0):
    t = tt(d + rel)
    out = np.zeros((2, len(t)))
    for m in ms:
        f = mtof(m)
        for ch in range(2):
            for k in range(5):
                dt = (k - 2) * 0.0035 + (ch - 0.5) * 0.0015
                out[ch] += saw(f, t, dt, rng.uniform(0, 6.28))
    out = signal.sosfilt(sos("lowpass", cutoff + bright, 2), out, axis=1)
    env = np.minimum(1, t / att) * np.where(t < d, 1.0, np.exp(-(t - d) / (rel / 4)))
    return out * env / (len(ms) * 5) * 0.9


def pluck(m, d=0.3, cut=3200):
    t = tt(d)
    f = mtof(m)
    y = saw(f, t) + 0.5 * signal.square(2 * np.pi * f * 1.002 * t)
    y = filt(y, "lowpass", cut, 2)
    return y * np.exp(-t * 11) * np.minimum(1, t / 0.002) * 0.35


def chime(m, d=2.5, gain=1.0):
    t = tt(d)
    f = mtof(m)
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t * 2.2) + 0.5 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 4.5)
         + 0.25 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t * 8))
    return y * np.minimum(1, t / 0.003) * 0.35 * gain


def impact(size=1.0):
    t = tt(3.5)
    f = 26 + 40 * np.exp(-t * 2.2)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * (1.2 / size))
    nz = filt(rng.standard_normal(len(t)), "lowpass", 900) * np.exp(-t * 4) * 0.8
    crash = filt(rng.standard_normal(len(t)), "highpass", 4000) * np.exp(-t * 1.3) * 0.35
    y = np.tanh(1.8 * (boom * 1.1 + nz)) + crash
    return y * 0.8 * size


def riser(d, f0=300, f1=7000):
    n = int(d * SR)
    x = rng.standard_normal(n)
    out = np.zeros(n)
    chunks = 60
    for c in range(chunks):
        a, b = c * n // chunks, (c + 1) * n // chunks
        fc = f0 * (f1 / f0) ** (c / chunks)
        out[a:b] = filt(x[max(0, a - 2000):b], "bandpass", [fc * 0.7, min(fc * 1.4, SR / 2 - 100)])[-(b - a):]
    t = np.arange(n) / SR
    sweep = np.sin(2 * np.pi * np.cumsum(180 * (6 ** (t / d))) / SR) * 0.25
    env = (t / d) ** 2.2
    return (out * 0.9 + sweep) * env * 0.7


def rev_cymbal(d=1.2):
    t = tt(d)
    y = filt(rng.standard_normal(len(t)), "highpass", 3500) * (t / d) ** 3
    return y * 0.5


def whoosh(d=0.8):
    t = tt(d)
    n = rng.standard_normal(len(t))
    y = filt(n, "bandpass", [400, 3000]) * np.sin(np.pi * t / d) ** 2
    return y * 0.35


def subdrop(d=1.8):
    t = tt(d)
    f = 30 + 70 * np.exp(-t * 2.5)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.2) * 0.9


def sine_tone(f, d):
    t = tt(d)
    return np.sin(2 * np.pi * f * t) * np.minimum(1, t / 0.08) * np.minimum(1, (d - t) / 0.6)


# ------------------------------------------------------------------ harmony
V = {"Dm": [50, 53, 57, 62], "Bb": [50, 53, 58, 62], "F": [48, 53, 57, 60], "C": [48, 52, 55, 60],
     "D": [50, 54, 57, 62]}
ROOT = {"Dm": 38, "Bb": 34, "F": 41, "C": 36, "D": 38}
PROG = ["Dm", "Bb", "F", "C"]


def chord_at(t):
    if t < 12:
        return "Dm"
    if 122 <= t < 130:
        return "C"
    if t >= 200:
        return "Dm" if t < 204 else ("Bb" if t < 208 else "D")
    return PROG[int((t - 12) // 4) % 4]


KICK_W = [(12, 29), (31, 84), (92, 119.5), (130, 141.5), (144, 188), (190, 198)]
HAT_W = [(14, 29), (31, 84), (94, 119.5), (132, 141.5), (146, 188), (190, 198)]
SNARE_W = [(68, 84), (144, 186), (190, 198)]
BASS_W = [(12, 29), (31, 84), (88, 119.5), (130, 141.5), (144, 188), (190, 198)]
ARP_W = [(36, 84), (84, 100), (144, 188), (190, 198)]
IMPACTS = [(12, 1.0), (29, 1.1), (36, 0.8), (60, 0.9), (76, 0.9), (84, 0.8), (100, 0.8), (122, 0.9), (144, 0.9),
           (162, 1.0), (171, 0.6), (179, 0.6), (190, 1.1), (200, 1.6)]
RISERS = [(8, 12), (26.5, 29), (57, 60), (81, 84), (97, 100), (119.6, 122), (141, 144), (159, 162), (186, 190),
          (196, 200)]


def build():
    drums, bass, padb, arps, fx, verb_send = Bus(), Bus(), Bus(), Bus(), Bus(), Bus()
    K, S, Hc, Ho = kick(), snare(), hat(), hat(True)
    kicks = []
    t = 0.0
    while t < TOTAL:
        if in_win(t, KICK_W):
            drums.add(K, t, 1.0)
            kicks.append(t)
        if in_win(t, HAT_W):
            drums.add(Hc, t + 0.25, 0.8, 0.25)
            if int(t / BEAT) % 4 == 3:
                drums.add(Ho, t + 0.25, 0.5, -0.2)
        if in_win(t, SNARE_W) and int(round(t / BEAT)) % 2 == 1:
            drums.add(S, t, 0.9)
            verb_send.add(S, t, 0.5)
        t += BEAT
    # snare roll into the finale + sparse 16th hats in busy sections
    t = 186.0
    while t < 190:
        step = 0.25 if t < 188 else (0.125 if t < 189.2 else 0.0625)
        g = 0.35 + 0.6 * (t - 186) / 4
        drums.add(S, t, g)
        verb_send.add(S, t, 0.3 * g)
        t += step
    # bass: rolling 8ths on chord roots
    t = 0.0
    while t < TOTAL:
        if in_win(t, BASS_W):
            m = ROOT[chord_at(t)]
            oct_ = 12 if int(t / 0.25) % 4 == 3 else 0
            bass.add(bass_note(m + oct_, 0.23), t, 0.9)
        t += 0.25
    # pads per 4-second block
    for b0 in np.arange(0, TOTAL, 4.0):
        c = chord_at(b0 + 0.01)
        d = 4.0 if b0 + 4 < TOTAL else TOTAL - b0 + 2
        lvl = 0.55
        cut = 1300
        if b0 < 12:
            lvl, cut = 0.45 * min(1, (b0 + 4) / 8), 900
        elif 84 <= b0 < 100 or 122 <= b0 < 130:
            lvl, cut = 0.5, 1100
        elif b0 >= 200:
            lvl, cut = 1.5, 2400
        elif 144 <= b0 < 190:
            lvl, cut = 0.6, 1900
        ms = V[c] + ([V[c][0] - 12] if b0 >= 200 else [])
        p = pad(ms, d, att=0.9 if b0 < 12 else 0.35, rel=2.5 if b0 >= 208 else 1.4, cutoff=cut)
        padb.add(p, b0, lvl)
        verb_send.add(p, b0, lvl * 0.4)
    # arpeggios (16ths)
    pat = [0, 1, 2, 3, 2, 1, 3, 1]
    t = 0.0
    i = 0
    while t < TOTAL:
        if in_win(t, ARP_W):
            ms = V[chord_at(t)]
            m = ms[pat[i % 8]] + 12 + (12 if (i // 8) % 2 and t > 144 else 0)
            soft = 84 <= t < 100
            g = 0.45 if soft else 0.55
            x = pluck(m, 0.28, 2200 if soft else 3400)
            pan = 0.35 * np.sin(i * 0.7)
            arps.add(x, t, g, pan)
            arps.add(x, t + 0.375, g * 0.35, -pan)  # dotted-8th echo
            verb_send.add(x, t, g * 0.25)
        t += 0.125
        i += 1
    # ---------------- sound design synced to picture
    # opening
    fx.add(chime(86, 4.0, 0.8), 1.5, 0.8)
    verb_send.add(chime(86, 4.0), 1.5, 0.6)
    fx.add(whoosh(0.7), 2.7, 1.0)
    fx.add(chime(81, 3.0), 3.0, 0.9)
    fx.add(subdrop(1.2), 3.0, 0.4)
    r2 = np.random.default_rng(11)
    launches = 7.4 + np.sort(r2.uniform(0, 3.2, 110) ** 1.4 / 3.2 ** 0.4)
    for k, lt in enumerate(launches[::2]):
        m = V["Dm"][k % 4] + 24
        fx.add(pluck(m, 0.25, 4000), lt, 0.25, r2.uniform(-0.8, 0.8))
        verb_send.add(pluck(m, 0.25, 4000), lt, 0.2)
    fx.add(rev_cymbal(1.0), 10.75, 0.8)
    # space scene transformation sweeps
    for w in (15.0, 18.0, 21.0, 25.5, 31.5):
        fx.add(whoosh(1.2), w - 0.2, 0.9)
    fx.add(subdrop(2.0), 29.0, 1.0)
    # SVD rank steps: rising blips
    ranks = [110.5, 111.5, 112.5, 113.5, 114.5, 115.0, 115.5, 116.0, 116.5, 117.0, 117.5]
    penta = [74, 77, 79, 81, 84, 86, 89, 91, 93, 96, 98]
    for tk, m in zip(ranks, penta):
        fx.add(chime(m, 1.2, 0.8), tk, 0.7)
        verb_send.add(chime(m, 1.2), tk, 0.4)
    # zoom into pixels: tick-tick digital
    for k in range(24):
        fx.add(hat() * 1.5, 103.6 + k * 0.15, 0.35 * (k / 24))
    # Fourier: the actual Do-Mi-Sol pure tones, spreading in stereo as they split
    for f, pan, dl in ((261.63, -0.7, 0.0), (329.63, 0.0, 0.15), (392.0, 0.7, 0.3)):
        d = 8.0
        x = sine_tone(f, d) * 0.22
        tpan = np.clip((tt(d) - 2.6 - dl) / 1.7, 0, 1) * pan
        a = (tpan + 1) * np.pi / 4
        st = np.stack([x * np.cos(a), x * np.sin(a)]) * np.sqrt(2)
        fx.add(st, 122.0, 1.0)
        verb_send.add(st, 122.0, 0.3)
    fx.add(pad(V["C"] + [72], 3.0, att=2.0, rel=2.0, cutoff=2600), 141.3, 0.55)  # 美 swell
    fx.add(chime(84, 4.0), 141.4, 0.9)
    # epicycle shimmer
    for k in range(40):
        tk = 131.6 + k * 0.25
        fx.add(chime([79, 84, 88, 91][k % 4], 0.8, 0.4), tk, 0.25, np.sin(k) * 0.6)
    # montage stabs on every cut
    for k in range(16):
        tk = 190 + k * 0.5
        for m in V[chord_at(tk + 200 - 190 if False else tk)]:
            fx.add(pluck(m + 12, 0.35, 5000), tk, 0.35)
        fx.add(impact(0.25), tk, 0.35)
    # converge -> big hit
    fx.add(subdrop(2.0) * 0.6, 198.0, 0.6)
    fx.add(rev_cymbal(2.0), 198.0, 1.2)
    for m in (26, 33, 38, 41, 45):  # brass-like chord hit
        t_ = tt(6.0)
        y = filt(saw(mtof(m), t_) + saw(mtof(m), t_, 0.005), "lowpass", 1400) * np.exp(-t_ * 0.45) * 0.28
        fx.add(y, 200.0, 1.0)
        verb_send.add(y, 200.0, 0.5)
    # closing melody resolving to D major
    for tk, m in ((204.0, 81), (205.0, 82), (206.0, 86), (207.0, 84), (208.0, 90), (209.5, 86), (211.0, 93)):
        fx.add(chime(m, 4.0, 0.9), tk, 0.8)
        verb_send.add(chime(m, 4.0), tk, 0.7)
    for tk in np.arange(202.0, 212.0, 1.0):
        drums.add(K * 0.5, tk, 0.5)
    fx.add(chime(86, 6.0), 200.3, 1.0)
    fx.add(chime(93, 6.0), 200.9, 0.8)
    verb_send.add(chime(86, 6.0), 200.3, 0.8)
    for a, s in IMPACTS:
        x = impact(s)
        fx.add(x, a, 1.0)
        verb_send.add(x, a, 0.35)
    for a, b in RISERS:
        x = riser(b - a)
        fx.add(x, a, 0.8)
        verb_send.add(x, a, 0.3)

    # sidechain duck from kicks
    duck = np.ones(N, np.float32)
    dn = int(0.35 * SR)
    de = 1 - 0.65 * np.exp(-np.arange(dn) / SR / 0.11)
    for k in kicks:
        i = int(k * SR)
        n = min(dn, N - i)
        duck[i:i + n] = np.minimum(duck[i:i + n], de[:n])
    for b in (bass, padb, arps):
        b.b *= duck

    # reverb
    ir_n = int(2.8 * SR)
    ti = np.arange(ir_n) / SR
    ir = np.stack([rng.standard_normal(ir_n), rng.standard_normal(ir_n)]) * np.exp(-ti * 2.4)
    ir = signal.sosfilt(sos("lowpass", 6000), ir, axis=1)
    ir /= np.sqrt((ir ** 2).sum(1, keepdims=True))
    wet = np.stack([signal.oaconvolve(verb_send.b[c], ir[c])[:N] for c in range(2)]).astype(np.float32)

    mix = drums.b * 0.85 + bass.b * 0.8 + padb.b * 0.7 + arps.b * 0.45 + fx.b * 0.8 + wet * 0.35
    mix = signal.sosfilt(sos("highpass", 28), mix, axis=1)
    mix = np.tanh(mix * 1.1) / np.tanh(1.1)
    # master fades
    tm = np.arange(N) / SR
    fade = np.clip(tm / 0.05, 0, 1) * np.clip((TOTAL + 1.5 - tm) / 5.0, 0, 1)
    mix *= fade
    mix /= np.abs(mix).max() / 0.93
    return mix


if __name__ == "__main__":
    m = build()
    wavfile.write("out/soundtrack.wav", SR, (m.T * 32767).astype(np.int16))
    print("wrote out/soundtrack.wav", m.shape[1] / SR, "s")
