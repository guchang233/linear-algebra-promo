"""Concatenate rendered scenes and mux the soundtrack -> 线性代数宣传片.mp4"""
import os
import subprocess
import sys

from timeline import SCENES, TOTAL

sub = "preview" if "--preview" in sys.argv else "full"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
with open("out/concat.txt", "w", encoding="utf8") as f:
    for name, _, _ in SCENES:
        f.write(f"file '{sub}/{name}.mp4'\n")
out = "线性代数宣传片.mp4" if sub == "full" else "out/preview.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "out/concat.txt",
                "-i", "out/soundtrack.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                "-t", str(TOTAL), "-movflags", "+faststart", out], check=True)
print("wrote", out)
