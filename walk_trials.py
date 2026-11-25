"""Milestone 2: one brain that walks, steers around pillars AND escapes from a thrown ball.

All runs use the same settings: walk 0.3 with heading hold, steering from DNb05 (fly brain),
escape from DNp01+DNp02+DNp11 (threshold 2) but only when the looming detector LPLC2 confirms it
on the same side (--escape-gate LPLC2:N, default 3), and LPLC2 decides the side. The walking duck's
own movement keeps the escape neurons near threshold, so LPLC2 is what separates real threats.
The ball thrown at the walking duck homes in on it until it looks 15 deg wide, then flies straight.
  open        walk, nothing around               -> should NOT escape
  pillar R/L  pillar slightly right / left        -> should steer around it
  ball R/L    ball thrown at the walking duck     -> should escape to the other side and dodge
  stand ball  Milestone-1 check: standing duck, ball from the right -> should still dodge

Usage (duck environment, from ~/flyduck/microduck_rl):
    uv run python ~/flyduck/bridge/walk_trials.py              # 6 runs
    uv run python ~/flyduck/bridge/walk_trials.py --gate 3     # stricter gate
    uv run python ~/flyduck/bridge/walk_trials.py --summary-only
Output: walk_trials_<label>.csv + table; videos loop_<scenario>_<label>_*.mp4
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
ap.add_argument("--gate", type=int, default=3, help="LPLC2 spikes needed on the threat side (100 ms)")
ap.add_argument("--rv", type=float, default=160)
ap.add_argument("--label", default="m2")
ap.add_argument("--repeats", type=int, default=1, help="run every scene this many times")
ap.add_argument("--summary-only", action="store_true")
ap.add_argument("--out", default=HERE)
args, passthrough = ap.parse_known_args()

BRAIN = ["--eye-gain", "2", "--step-ms", "20", "--threshold", "2",
         "--escape-cmd", "-0.3", "0.2", "1.5", "--escape-gate", f"LPLC2:{args.gate}"]
WALK = ["--walk", "0.3", "--steer", "DNb05", "--steer-gain", "0.04", "--steer-bias", "5"]
BALL = ["--rv", str(args.rv), "--fly-through", "--start-deg", "6"]
L = args.label
SCENES = [("open", "none", "open", WALK + ["--duration", "7"]),
          ("pillar R", "pillar", "right", WALK + ["--side", "right", "--duration", "7"]),
          ("pillar L", "pillar", "left", WALK + ["--side", "left", "--duration", "7"]),
          ("ball R", "loom", "walk_right", WALK + BALL + ["--side", "right", "--loom-start", "2"]),
          ("ball L", "loom", "walk_left", WALK + BALL + ["--side", "left", "--loom-start", "2"]),
          ("stand ball", "loom", "stand_right", ["--rv", str(args.rv), "--fly-through", "--side", "right"])]
runs = []                                       # (scene, display name, scenario, tag, extra)
for i in range(1, args.repeats + 1):
    for name, scen, t, extra in SCENES:
        runs.append((name, name + (f" #{i}" if args.repeats > 1 else ""), scen,
                     f"{L}_{t}" + (f"_{i}" if i > 1 else ""), extra))

if not args.summary_only:
    t_all = time.time()
    for k, (_, name, scen, tag, extra) in enumerate(runs, 1):
        cmd = [sys.executable, os.path.join(HERE, "fly_loop.py"), "--scenario", scen, "--tag", tag,
               "--out", args.out] + BRAIN + extra + passthrough
        print(f"##### run {k}/{len(runs)}: {name}   [{(time.time() - t_all) / 60:.0f} min so far]",
              flush=True)
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            print(f"   run failed (exit {r.returncode}); re-run it alone to see the error:\n   "
                  + " ".join(cmd), flush=True)

rows = []
print(f"\n=== Milestone 2: walk + steer + escape (gate LPLC2 >= {args.gate}) ===")
print(f"{'run':<14}{'outcome':<16}{'gap cm':>8}{'escapes':>9}  {'first escape':<16}{'drift cm':>9}"
      f"{'turned':>8}  fell")
for scene, name, scen, tag, _ in runs:
    path = os.path.join(args.out, f"result_{scen}_{tag}.json")
    if not os.path.exists(path):
        print(f"{name:<14}MISSING (run failed?)")
        continue
    d = json.load(open(path))
    gap = d.get("gap_cm") if scen == "loom" else d.get("obstacle_gap_cm")
    esc = d.get("escapes", [])
    first = f"{esc[0][1]} at {esc[0][0]:.2f}s" if esc else "-"
    print(f"{name:<14}{d['outcome']:<16}{('-' if gap is None else f'{gap:+.1f}'):>8}{len(esc):>9}  "
          f"{first:<16}{d.get('end_side_cm', 0):>+9.1f}{d['turned_deg']:>+8.0f}  {d['fell']}")
    first_side = esc[0][1] if esc else None
    rows.append({"scene": scene, "run": name, "outcome": d["outcome"], "gap_cm": gap, "n_escapes": len(esc),
                 "escapes": esc, "first_side": first_side, "escape_deg": d.get("escape_deg"),
                 "warning_ms": d.get("warning_ms"), "drift_cm": d.get("end_side_cm"), "turned_deg": d["turned_deg"],
                 "fell": d["fell"]})
print("  want: open 0 escapes; pillars CLEAR; balls MISSED with the first escape on the ball's side;"
      " stand ball MISSED")
# tally per scene (also used by tools/fill_readme.py)
tally = {}
for scene, *_ in SCENES:
    rs = [r for r in rows if r["scene"] == scene]
    if not rs:
        continue
    side = "right" if scene.endswith("R") or scene == "stand ball" else "left"
    gaps = [r["gap_cm"] for r in rs if r["gap_cm"] is not None]
    tally[scene] = {
        "n": len(rs),
        "ok": sum(r["outcome"] in ("CLEAR", "MISSED", "OK (no escape)") for r in rs),
        "runs_with_escape": sum(r["n_escapes"] > 0 for r in rs),
        "first_escape_correct_side": sum(r["first_side"] == side for r in rs if r["first_side"]),
        "gap_min": min(gaps) if gaps else None, "gap_max": max(gaps) if gaps else None,
        "deg": [r["escape_deg"] for r in rs if r["escape_deg"] is not None],
        "warning_ms": [r["warning_ms"] for r in rs if r["warning_ms"] is not None],
        "falls": sum(bool(r["fell"]) for r in rs)}
print("\nper scene:")
for scene, t in tally.items():
    extra = ""
    if t["gap_min"] is not None:
        extra += f"  gaps {t['gap_min']:+.1f}..{t['gap_max']:+.1f} cm"
    if scene != "open":
        extra += f"  first escape on the threat side {t['first_escape_correct_side']}/{t['runs_with_escape']}"
    print(f"  {scene:<11} ok {t['ok']}/{t['n']}  runs with an escape {t['runs_with_escape']}/{t['n']}"
          f"{extra}  falls {t['falls']}")
with open(os.path.join(args.out, f"walk_trials_{L}_summary.json"), "w") as fh:
    json.dump({"label": L, "gate": args.gate, "rv": args.rv, "repeats": args.repeats, "tally": tally,
               "rows": rows}, fh, indent=1)
if rows:
    with open(os.path.join(args.out, f"walk_trials_{L}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
