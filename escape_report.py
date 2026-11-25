"""Read the walk_trials logs back and show what the escape neurons did around every escape.

No simulation - just reads loop_*.csv files. Usage (any python with the standard library):
    python3 ~/flyduck/bridge/escape_report.py            # label m2
    python3 ~/flyduck/bridge/escape_report.py --label m2
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--label", default="m2")
ap.add_argument("--out", default=HERE)
args = ap.parse_args()
L = args.label
RUNS = [("open", "none", f"{L}_open"), ("pillar R", "pillar", f"{L}_right"),
        ("pillar L", "pillar", f"{L}_left"), ("ball R", "loom", f"{L}_walk_right"),
        ("ball L", "loom", f"{L}_walk_left"), ("stand ball", "loom", f"{L}_stand_right")]
G = ["DNp01", "DNp02", "DNp11", "LPLC2", "LC4", "DNb05"]


def n(r, k):
    try:
        return int(float(r.get(k, 0) or 0))
    except ValueError:
        return 0


def f(r, k):
    try:
        return float(r.get(k, "nan"))
    except ValueError:
        return float("nan")


for name, scen, tag in RUNS:
    path = os.path.join(args.out, f"loop_{scen}_{tag}.csv")
    if not os.path.exists(path):
        print(f"\n### {name}: no log ({path})")
        continue
    rows = list(csv.DictReader(open(path)))
    dur = float(rows[-1]["t"])
    print(f"\n### {name}  ({dur:.1f} s)")
    tot = {g: (sum(n(r, g + ":right") for r in rows), sum(n(r, g + ":left") for r in rows)) for g in G}
    print("  totals R/L: " + "  ".join(f"{g} {a}/{b}" for g, (a, b) in tot.items()))
    # per second R/L of the escape groups and the gate
    print("  per second   " + "".join(f"{g + ' R/L':>12}" for g in G) + "   ball deg")
    for s0 in range(int(dur + 0.999)):
        sl = [r for r in rows if s0 < float(r["t"]) <= s0 + 1]
        if not sl:
            continue
        th = max(f(r, "theta_deg") for r in sl)
        print(f"  {s0}-{s0 + 1} s     " + "".join(
            f"{str(sum(n(r, g + ':right') for r in sl)) + '/' + str(sum(n(r, g + ':left') for r in sl)):>12}"
            for g in G) + (f"   {th:5.0f}" if th == th and th > 0 else ""))
    # escape onsets: escaping command starts (cmd_vx < 0 after not escaping)
    prev = False
    for i, r in enumerate(rows):
        esc = f(r, "cmd_vx") < 0
        if esc and not prev:
            t = float(r["t"])
            side = "right" if f(r, "cmd_yaw") > 0 else "left"     # yaw + (left) = escaping a RIGHT threat
            print(f"  ESCAPE at {t:.2f} s from a {side} threat; ball {f(r, 'theta_deg'):.0f} deg, "
                  f"time to impact {f(r, 'ttc_ms'):+.0f} ms. last 120 ms (R/L):")
            for rr in rows[max(0, i - 5):i + 1]:
                print(f"     t={float(rr['t']):.2f}  esc {n(rr, 'escape_R')}/{n(rr, 'escape_L')}  " + "  ".join(
                    f"{g} {n(rr, g + ':right')}/{n(rr, g + ':left')}" for g in G[:4])
                      + f"   ball {f(rr, 'theta_deg'):.0f} deg")
        prev = esc
