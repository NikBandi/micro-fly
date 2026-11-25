"""Milestone 1 statistics: repeated, independent dodge trials from both sides + false-alarm checks.

Runs fly_loop.py with the final Milestone-1 settings:
    --eye-gain 2 --step-ms 20 --threshold 2 --escape-cmd -0.3 0.2 1.5 --fly-through
  * N looming-ball trials from the right and N from the left (r/v --rv, default 160 ms)
  * one passing-ball run per side and one empty-scene run (should NOT escape)
Every run is a fresh process (the brain's random spike timing differs each time).

Usage (duck environment, from ~/flyduck/microduck_rl):
    uv run python ~/flyduck/bridge/dodge_trials.py                 # 3 trials per side, ~35 min
    uv run python ~/flyduck/bridge/dodge_trials.py --trials 5 --rv 160 500
    uv run python ~/flyduck/bridge/dodge_trials.py --summary-only  # re-print the table
Output: dodge_trials.csv + a printed table. Per-run videos/logs as loop_<scenario>_<tag>.*
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
ap.add_argument("--trials", type=int, default=3, help="looming trials per side and speed")
ap.add_argument("--rv", type=float, nargs="+", default=[160])
ap.add_argument("--sides", nargs="+", default=["right", "left"], choices=["right", "left"])
ap.add_argument("--no-controls", action="store_true", help="skip the pass / empty-scene runs")
ap.add_argument("--summary-only", action="store_true")
ap.add_argument("--out", default=HERE)
args, passthrough = ap.parse_known_args()

FINAL = ["--eye-gain", "2", "--step-ms", "20", "--threshold", "2", "--escape-cmd", "-0.3", "0.2", "1.5"]

runs = []                                   # (scenario, tag, extra args)
for rv in args.rv:
    for side in args.sides:
        for i in range(1, args.trials + 1):
            tag = f"trial_rv{rv:g}_{side}_{i}"
            runs.append(("loom", tag, ["--rv", str(rv), "--side", side, "--fly-through"]))
if not args.no_controls:
    for side in args.sides:
        runs.append(("pass", f"trial_{side}", ["--side", side]))
    runs.append(("none", "trial", []))

if not args.summary_only:
    t_all = time.time()
    for k, (scen, tag, extra) in enumerate(runs, 1):
        cmd = [sys.executable, os.path.join(HERE, "fly_loop.py"), "--scenario", scen, "--tag", tag,
               "--out", args.out] + FINAL + extra + passthrough
        print(f"\n##### run {k}/{len(runs)}: {scen} {tag}   [{(time.time() - t_all) / 60:.0f} min so far]",
              flush=True)
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL)   # per-step lines go to the CSV log instead
        if r.returncode != 0:
            print(f"   run failed (exit {r.returncode}); continuing", flush=True)

rows = []
for scen, tag, _ in runs:
    path = os.path.join(args.out, f"result_{scen}_{tag}.json")
    if os.path.exists(path):
        d = json.load(open(path))
        d["tag"] = tag
        rows.append(d)
    else:
        rows.append({"scenario": scen, "tag": tag, "outcome": "MISSING (run failed?)"})

fields = ["tag", "scenario", "side", "rv", "outcome", "gap_cm", "escape_side", "escape_deg", "warning_ms",
          "moved_cm", "turned_deg", "fell"]
with open(os.path.join(args.out, "dodge_trials.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

f = lambda v, fmt: "-" if v is None else format(v, fmt)
print("\n=== Milestone 1 dodge trials ===")
print(f"{'run':<26}{'outcome':<16}{'gap cm':>7}{'esc side':>9}{'deg':>6}{'warn ms':>8}{'moved':>7}  fell")
for d in rows:
    print(f"{d['tag']:<26}{d['outcome']:<16}{f(d.get('gap_cm'), '+.1f'):>7}"
          f"{(d.get('escape_side') or '-'):>9}{f(d.get('escape_deg'), '.0f'):>6}"
          f"{f(d.get('warning_ms'), '+.0f'):>8}{f(d.get('moved_cm'), '.0f'):>7}  {d.get('fell', '-')}")
looms = [d for d in rows if d.get("scenario") == "loom"]
dodged = sum(d.get("outcome") == "MISSED" for d in looms)
right_side = sum(d.get("escape_side") == d.get("side") for d in looms if d.get("escape_side"))
ctrl = [d for d in rows if d.get("scenario") in ("pass", "none")]
false_alarms = sum(d.get("outcome") == "FALSE ALARM" for d in ctrl)
fell = sum(bool(d.get("fell")) for d in rows if "fell" in d)
print(f"\ndodged {dodged}/{len(looms)} looming balls; escape on the threatened side in "
      f"{right_side}/{sum(1 for d in looms if d.get('escape_side'))}; false alarms {false_alarms}/{len(ctrl)}; "
      f"falls {fell}")
print(f"table: {os.path.join(args.out, 'dodge_trials.csv')}")
