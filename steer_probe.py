"""Milestone 2, step 1: which fly-brain neurons could steer the duck around obstacles?

The duck walks straight ahead (no steering yet, escapes only logged) while the fly brain watches:
  * open      - nothing in the way (walking alone: its own optic flow, head bob)
  * pillar R  - a dark pillar ahead, slightly right of the walking line
  * pillar L  - the same pillar slightly left
  * corridor  - walking down a corridor with striped walls
For each brain readout group the table shows spikes right / left. A neuron type useful for
avoiding pillars should flip its left/right balance between 'pillar R' and 'pillar L' while staying
balanced in 'open'. The last column (R-L in pillar R) - (R-L in pillar L) says how: positive = fires
on the pillar's side, negative = fires on the opposite side.

Usage (duck environment, from ~/flyduck/microduck_rl):
    uv run python ~/flyduck/bridge/steer_probe.py                # 4 runs, ~35-45 min
    uv run python ~/flyduck/bridge/steer_probe.py --summary-only
Output: steer_<label>.csv + table. Per-run videos/logs as loop_<scenario>_probe_*.mp4 / .csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=7.0)
ap.add_argument("--walk", type=float, default=0.3)
ap.add_argument("--repeats", type=int, default=1, help="pillar runs per side (fresh brain noise each)")
ap.add_argument("--label", default="probe", help="names the runs (loop_<scenario>_<label>_*)")
ap.add_argument("--summary-only", action="store_true")
ap.add_argument("--out", default=HERE)
args, passthrough = ap.parse_known_args()

COMMON = ["--eye-gain", "2", "--step-ms", "20", "--threshold", "2", "--no-escape",
          "--walk", str(args.walk), "--duration", str(args.duration)]
L = args.label
runs = [("open", "none", f"{L}_open", []),
        ("pillar R", "pillar", f"{L}_right", ["--side", "right"]),
        ("pillar L", "pillar", f"{L}_left", ["--side", "left"]),
        ("corridor", "corridor", L, [])]
for i in range(2, args.repeats + 1):
    runs += [(f"pillar R{i}", "pillar", f"{L}_right_{i}", ["--side", "right"]),
             (f"pillar L{i}", "pillar", f"{L}_left_{i}", ["--side", "left"])]

if not args.summary_only:
    t_all = time.time()
    for k, (name, scen, tag, extra) in enumerate(runs, 1):
        cmd = [sys.executable, os.path.join(HERE, "fly_loop.py"), "--scenario", scen, "--tag", tag,
               "--out", args.out] + COMMON + extra + passthrough
        print(f"\n##### run {k}/{len(runs)}: {name}   [{(time.time() - t_all) / 60:.0f} min so far]",
              flush=True)
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if r.returncode != 0:
            print(f"   run failed (exit {r.returncode}); continuing", flush=True)

res = {}
for name, scen, tag, _ in runs:
    path = os.path.join(args.out, f"result_{scen}_{tag}.json")
    res[name] = json.load(open(path)) if os.path.exists(path) else None

steer_args = " ".join(passthrough)
print(f"\n=== Milestone 2 steering {L} ({steer_args or 'duck walks straight'}; spikes right / left over the run) ===")
print(f"{'run':<10}{'outcome':<16}{'gap cm':>8}{'forward cm':>12}{'drift cm':>10}"
      f"{'turned':>8}{'escape signal':>16}  fell")
for name, d in res.items():
    if d is None:
        print(f"{name:<10}MISSING (run failed?)")
        continue
    esc = f"{d['escape_side']} {d['escape_t']:.2f}s" if d.get("escaped") else "none"
    gap = d.get("obstacle_gap_cm")
    print(f"{name:<10}{d['outcome']:<16}{('-' if gap is None else f'{gap:+.1f}'):>8}"
          f"{d.get('end_x_cm', 0):>12.0f}{d.get('end_side_cm', 0):>+10.1f}{d['turned_deg']:>+8.0f}"
          f"{esc:>16}  {d['fell']}")
print("  (drift + = to the right; escape signal = first time the escape count crossed threshold; "
      "logged only)")

pil = [d for n, d in res.items() if n.startswith("pillar") and d]
if pil:
    print(f"pillar runs cleared: {sum(d['outcome'] == 'CLEAR' for d in pil)}/{len(pil)}; "
          f"gaps {', '.join(f'{d["obstacle_gap_cm"]:+.1f}' for d in pil)} cm")

groups = sorted({k.split(":")[0] for d in res.values() if d for k in d["totals"]})
names = [n for n in ("open", "pillar R", "pillar L", "corridor") if res.get(n)]
print(f"\n{'group':<8}" + "".join(f"{n:>14}" for n in names) + f"{'pillar side':>14}")
rows = []
for g in groups:
    cells, rl = [], {}
    for n in names:
        tt = res[n]["totals"]
        r, l = tt.get(g + ":right", 0), tt.get(g + ":left", 0)
        cells.append(f"{r:>6d} /{l:<6d}")
        rl[n] = r - l
    contrast = (rl.get("pillar R", 0) - rl.get("pillar L", 0)) if {"pillar R", "pillar L"} <= set(names) else None
    print(f"{g:<8}" + "".join(f"{c:>14}" for c in cells)
          + ("" if contrast is None else f"{contrast:>+14d}"))
    rows.append({"group": g, **{f"{n} R": res[n]["totals"].get(g + ":right", 0) for n in names},
                 **{f"{n} L": res[n]["totals"].get(g + ":left", 0) for n in names},
                 "pillar_side_contrast": contrast})
print("  pillar side: (R-L with the pillar on the right) - (R-L with it on the left); "
      "+ = fires on the pillar's side, - = opposite side")

# time course near the pillar: R-L per second for the groups that fired at all
for name, scen, tag, _ in runs[1:3]:
    path = os.path.join(args.out, f"loop_{scen}_{tag}.csv")
    if not os.path.exists(path):
        continue
    log = list(csv.DictReader(open(path)))
    active = [g for g in groups if sum(int(r.get(g + ":right", 0)) + int(r.get(g + ":left", 0))
                                       for r in log) >= 10]
    print(f"\n{name}: R-L spikes per second (pillar distance and bearing at the end of each second)")
    print(f"{'sec':<5}{'pillar':>16}" + "".join(f"{g:>8}" for g in active))
    for s0 in range(int(args.duration)):
        sl = [r for r in log if s0 < float(r["t"]) <= s0 + 1]
        if not sl:
            continue
        where = f"{float(sl[-1]['obs_a']) * 100:4.0f} cm {float(sl[-1]['obs_b']):+4.0f} deg"
        print(f"{s0}-{s0 + 1:<3}{where:>16}" + "".join(
            f"{sum(int(r[g + ':right']) - int(r[g + ':left']) for r in sl):>+8d}" for g in active))

with open(os.path.join(args.out, f"steer_{L}.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\ntable: {os.path.join(args.out, f'steer_{L}.csv')}")
