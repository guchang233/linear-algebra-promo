# (module, global start, global end) in seconds. 120 BPM -> 1 bar = 2 s. All cuts land on bars.
SCENES = [
    ("s00_open", 0, 12),
    ("s01_space", 12, 36),
    ("s02_eigen", 36, 60),
    ("s03_3d", 60, 84),
    ("s04_fractal", 84, 100),
    ("s05_image", 100, 122),
    ("s06_fourier", 122, 144),
    ("s07_network", 144, 162),
    ("s08_ai", 162, 190),
    ("s09_finale", 190, 214),
]
TOTAL = SCENES[-1][2]
BPM = 120
