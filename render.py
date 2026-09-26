"""Render scenes.
  python render.py s01 --preview --still 2,5.5     # PNG stills
  python render.py s01 s02 --preview               # preview videos (960x540@30)
  python render.py all -j 6                        # full quality, parallel
"""
import argparse
import os
import subprocess
import sys
import time

from timeline import SCENES

ap = argparse.ArgumentParser()
ap.add_argument("scenes", nargs="+")
ap.add_argument("--preview", action="store_true")
ap.add_argument("--still", default="")
ap.add_argument("-j", "--jobs", type=int, default=6)
args = ap.parse_args()

names = [s[0] for s in SCENES]
if args.scenes == ["all"]:
    todo = names
else:
    todo = [n for n in names if any(n.startswith(p) for p in args.scenes)]

if args.preview:
    os.environ.setdefault("LA_W", "960")
    os.environ.setdefault("LA_H", "540")
    os.environ.setdefault("LA_FPS", "30")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out", "preview" if args.preview else "full")
os.makedirs(OUT, exist_ok=True)

if len(todo) > 1 and not args.still:
    # parallel: one subprocess per scene; finale last (it samples the other clips)
    first = [n for n in todo if n != "s09_finale"]
    procs, t0 = [], time.time()
    queue = list(first)
    running = []
    while queue or running:
        while queue and len(running) < args.jobs:
            n = queue.pop(0)
            cmd = [sys.executable, __file__, n] + (["--preview"] if args.preview else [])
            running.append((n, subprocess.Popen(cmd, env={**os.environ, 'NUMBA_NUM_THREADS': '4'})))
        time.sleep(1)
        for n, p in list(running):
            if p.poll() is not None:
                running.remove((n, p))
                print(f"[done] {n} rc={p.returncode} ({time.time() - t0:.0f}s)", flush=True)
    if "s09_finale" in todo:
        subprocess.run([sys.executable, __file__, "s09_finale"] + (["--preview"] if args.preview else []))
    print(f"all done in {time.time() - t0:.0f}s")
    sys.exit(0)

import importlib
import cv2
import engine as E

for name in todo:
    mod = importlib.import_module(f"scenes.{name}")
    sc = mod.S()
    sc.OUT_DIR = OUT
    sc.setup()
    n = int(round(sc.DUR * E.FPS))
    dt = 1.0 / E.FPS
    t0 = time.time()
    if args.still:
        need = {min(n - 1, int(round(float(s) * E.FPS))): float(s) for s in args.still.split(",")}
        last = max(need)
        stateful = getattr(sc, "STATEFUL", False)
        for i in range(last + 1):
            t = i * dt
            if stateful:
                sc.update(t, dt)
            if i in need:
                if not stateful:
                    sc.update(t, dt)
                img = sc.draw(t, i)
                p = os.path.join(ROOT, "out", "stills", f"{name}_{need[i]:06.2f}.png")
                cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(p)
                print("saved", p, flush=True)
    else:
        w = E.Writer(os.path.join(OUT, f"{name}.mp4"))
        for i in range(n):
            t = i * dt
            sc.update(t, dt)
            w.write(sc.draw(t, i))
            if i % (E.FPS * 4) == 0:
                el = time.time() - t0
                print(f"{name}: {i}/{n}  {el:.0f}s  ({el / max(i, 1) * 1000:.0f} ms/f)", flush=True)
        w.close()
    print(f"{name} finished in {time.time() - t0:.0f}s", flush=True)
