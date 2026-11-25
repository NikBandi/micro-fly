"""Turn run videos and plots into the README's media/ files (GIFs + PNGs). Needs ffmpeg.

Usage (from the repository folder, any python):
    python3 tools/make_media.py            # skips clips whose source video is missing
Each GIF is cut from a loop_*.mp4 that fly_loop.py wrote (see the list below), scaled down
and limited to a few MB so the README loads quickly.
"""
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA = os.path.join(ROOT, "media")
os.makedirs(MEDIA, exist_ok=True)

# (output gif, [source videos], start s (negative = from the end), length s)
# (media/dodge_rv500.gif, the Milestone-1 dodge, is already in the repository)
CLIPS = [
    ("m1_escape_not_enough.gif", ["loop_loom_final_rv160.mp4"], -3.0, 3.0),
    ("m1_pass_no_escape.gif", ["loop_pass_pass_t2.mp4"], 0.0, 2.5),
    ("m2_pillar_no_steer_vs_steer.gif", ["loop_pillar_probe_right.mp4", "loop_pillar_dnb05_right.mp4"], 2.0, 5.0),
    ("m2_walk_steer_escape_pillar.gif", ["loop_pillar_m2b_right.mp4"], 2.0, 5.0),
    ("m2_walking_ball_dodge.gif", ["loop_loom_m2b_walk_right.mp4"], 3.0, 3.5),
]
PNGS = [("loom_sweep_g2_s20.png", "m1_loom_sweep.png"), ("eye_check_duck_loom.png", "eye_check_duck_loom.png")]
WIDTH, FPS = 400, 10


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip().splitlines()[-1] if r.stderr else "ffmpeg failed")


for out, srcs, start, length in CLIPS:
    paths = [os.path.join(ROOT, s) for s in srcs]
    missing = [s for s, p in zip(srcs, paths) if not os.path.exists(p)]
    if missing:
        print(f"skip  {out:<38} (missing {', '.join(missing)})")
        continue
    cmd = ["ffmpeg", "-loglevel", "error", "-y"]
    for p in paths:
        cmd += (["-sseof", str(start)] if start < 0 else ["-ss", str(start)]) + ["-t", str(length), "-i", p]
    scale = f"fps={FPS},scale={WIDTH}:-1:flags=lanczos"
    if len(paths) == 1:
        graph = f"[0:v]{scale}[v]"
    else:
        graph = ";".join(f"[{i}:v]{scale}[s{i}]" for i in range(len(paths)))
        graph += ";" + "".join(f"[s{i}]" for i in range(len(paths))) + f"hstack=inputs={len(paths)}[v]"
    graph += ";[v]split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    dst = os.path.join(MEDIA, out)
    try:
        run(cmd + ["-filter_complex", graph, "-loop", "0", dst])
        print(f"made  {out:<38} {os.path.getsize(dst) / 1e6:4.1f} MB")
    except RuntimeError as e:
        if os.path.exists(dst):
            os.remove(dst)
        print(f"FAIL  {out:<38} {e} (is the video shorter than the clip?)")

for src, out in PNGS:
    p = os.path.join(ROOT, src)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(MEDIA, out))
        print(f"copy  {out}")
    else:
        print(f"skip  {out:<38} (missing {src})")
