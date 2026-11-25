"""Loom-speed sweep: does the fly brain fire at a fixed angular size, like a real fly?

Real flies (and locusts) trigger looming escapes when the approaching object reaches
a roughly fixed angular size, so the warning time before collision grows in proportion
to r/v (object radius / approach speed). This script runs fly_loop.py for several r/v
values and checks that.

Default is --observe (the duck stays still, so every run sees the full approach and the
timings can be compared). Add --behave to let the duck actually escape.

Usage (duck environment, from ~/flyduck/microduck_rl):
    uv run python ~/flyduck/bridge/loom_sweep.py                      # r/v 20 40 80 160 ms
    uv run python ~/flyduck/bridge/loom_sweep.py --rv 40 80 --eye-gain 2
    uv run python ~/flyduck/bridge/loom_sweep.py --summary-only        # re-read old CSVs
Outputs (bridge folder): loop_loom_rv<NN>_g<G>.csv/.mp4 per run, loom_sweep_g<G>.csv,
loom_sweep_g<G>.png (if matplotlib is installed).
"""
import argparse
import csv
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--rv", type=float, nargs="+", default=[20, 40, 80, 160])
ap.add_argument("--eye-gain", type=float, default=1.0)
ap.add_argument("--threshold", type=int, default=3)
ap.add_argument("--step-ms", type=int, default=100, choices=[20, 40, 100])
ap.add_argument("--behave", action="store_true", help="let the duck escape (default: observe)")
ap.add_argument("--untrained-eyes", action="store_true")
ap.add_argument("--summary-only", action="store_true")
ap.add_argument("--out", default=HERE)
args, passthrough = ap.parse_known_args()      # other options go to fly_loop.py

GROUPS = ["DNp01", "DNp02", "DNp11", "LPLC2", "LC4"]
gtag = f"g{args.eye_gain:g}" + (f"_s{args.step_ms}" if args.step_ms != 100 else "")


def run_tag(rv):
    return f"rv{rv:g}_{gtag}" + ("_behave" if args.behave else "")


def run_one(rv):
    cmd = [sys.executable, os.path.join(HERE, "fly_loop.py"), "--scenario", "loom",
           "--rv", str(rv), "--eye-gain", str(args.eye_gain), "--threshold", str(args.threshold),
           "--tag", run_tag(rv), "--out", args.out, "--step-ms", str(args.step_ms)]
    if not args.behave:
        cmd.append("--observe")
    if args.untrained_eyes:
        cmd.append("--untrained-eyes")
    cmd += passthrough
    print(f"\n### r/v = {rv:g} ms", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"### r/v = {rv:g} ms done in {time.time() - t0:.0f} s", flush=True)


def summarise(rv):
    path = os.path.join(args.out, f"loop_loom_{run_tag(rv)}.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    f = lambda r, k: float(r[k])
    out = {"rv_ms": rv}
    hit = next((r for r in rows if max(f(r, "escape_R"), f(r, "escape_L")) >= args.threshold), None)
    out["escape_t"] = f(hit, "t") if hit else math.nan
    out["escape_theta_deg"] = f(hit, "theta_deg") if hit else math.nan
    out["escape_ttc_ms"] = f(hit, "ttc_ms") if hit else math.nan
    out["escape_side"] = ("right" if f(hit, "escape_R") >= f(hit, "escape_L") else "left") if hit else ""
    # first 100 ms bin in which each right-side group fires at all, and its angular size then
    for g in GROUPS:
        first = next((r for r in rows if f(r, f"{g}:right") > 0), None)
        out[f"{g}_first_theta"] = f(first, "theta_deg") if first else math.nan
        out[f"{g}_total_R"] = int(sum(f(r, f"{g}:right") for r in rows))
        out[f"{g}_total_L"] = int(sum(f(r, f"{g}:left") for r in rows))
    # eye drive: angular size at which right-eye T4+T5 drive reaches half its peak
    tt = [f(r, "eye_T4:right") + f(r, "eye_T5:right") for r in rows]
    half = max(tt) / 2 if tt and max(tt) > 0 else math.inf
    first = next((r for r, v in zip(rows, tt) if v >= half), None)
    out["eyeT4T5_half_theta"] = f(first, "theta_deg") if first else math.nan
    out["eyeT4T5_peak"] = round(max(tt), 0) if tt else math.nan
    out["_rows"] = rows
    return out


if not args.summary_only:
    for rv in args.rv:
        run_one(rv)

res = [s for s in (summarise(rv) for rv in args.rv) if s]
if not res:
    raise SystemExit("loom_sweep: no results found")
fields = [k for k in res[0] if not k.startswith("_")]
sum_path = os.path.join(args.out, f"loom_sweep_{gtag}" + ("_behave" if args.behave else "") + ".csv")
with open(sum_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows([{k: s[k] for k in fields} for s in res])

fmt = lambda x: "  -  " if x != x else f"{x:5.0f}"
print(f"\n=== loom sweep (eye gain {args.eye_gain:g}, threshold {args.threshold}, decide every {args.step_ms} ms) ===")
print(" r/v ms | escape: deg wide, ms before collision, side | first spike (deg wide): "
      "LPLC2   LC4  DNp02  GF | eye T4+T5 half-peak deg")
for s in res:
    print(f" {s['rv_ms']:6g} |        {fmt(s['escape_theta_deg'])}      {fmt(s['escape_ttc_ms'])}"
          f"     {s['escape_side'] or '-':>5} |                         "
          f"{fmt(s['LPLC2_first_theta'])} {fmt(s['LC4_first_theta'])} {fmt(s['DNp02_first_theta'])}"
          f" {fmt(s['DNp01_first_theta'])} | {fmt(s['eyeT4T5_half_theta'])}")
print("\nFly-like = the 'deg wide' at escape stays roughly the same across r/v, so the warning time")
print("(ms before collision) grows with r/v. If escape only ever happens near the end, compare the")
print("eye T4+T5 column with LPLC2: early eye drive but late LPLC2 = brain needs more drive (try")
print("--eye-gain); late eye drive too = the vision side is the bottleneck.")
print(f"summary: {sum_path}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("(matplotlib not installed: no plot)")
    raise SystemExit
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for s in res:
    rows = s["_rows"]
    ttc = [float(r["ttc_ms"]) for r in rows]
    hz = 1000.0 / args.step_ms                                         # counts per row -> Hz
    gf = [hz * float(r["DNp01:right"]) for r in rows]
    lplc2 = [hz * float(r["LPLC2:right"]) / 102 for r in rows]          # Hz per cell
    axs[0].plot(ttc, gf, label=f"r/v {s['rv_ms']:g} ms")
    axs[1].plot(ttc, lplc2, label=f"r/v {s['rv_ms']:g} ms")
for ax, title in [(axs[0], "right giant fiber (Hz)"), (axs[1], "right LPLC2 (Hz per cell)")]:
    ax.set_xlabel("time to collision (ms)")
    ax.set_xlim(max(10 * max(s["rv_ms"] for s in res), 1000), -300)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(title)
    ax.legend()
rv = [s["rv_ms"] for s in res]
axs[2].plot(rv, [s["escape_ttc_ms"] for s in res], "o-", label="escape: ms before collision")
axs[2].plot(rv, [s["escape_theta_deg"] for s in res], "s--", label="escape: angular size (deg)")
axs[2].set_xlabel("r/v (ms)")
axs[2].axhline(0, color="k", lw=0.8)
axs[2].legend()
axs[2].set_title("fly-like: angle flat, warning time rising")
fig.tight_layout()
png = sum_path.replace(".csv", ".png")
fig.savefig(png, dpi=110)
print(f"plot: {png}")
