"""Write the Milestone-2 results table into README.md from walk_trials_<label>_summary.json.

Usage (from the repository folder):  python3 tools/fill_readme.py --label final
Replaces everything between <!-- M2_RESULTS_START --> and <!-- M2_RESULTS_END -->.
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--label", default="final")
args = ap.parse_args()
d = json.load(open(os.path.join(ROOT, f"walk_trials_{args.label}_summary.json")))
T, rows = d["tally"], d["rows"]


def rng(v, fmt, unit=""):
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    lo, hi = min(v), max(v)
    return (format(lo, fmt) if lo == hi else f"{format(lo, fmt)} to {format(hi, fmt)}") + unit


def line(scene, what, result):
    return f"| {scene} | {what} | {result} |"


out = ["| Scene | Should | Result |", "|---|---|---|"]
if "open" in T:
    t = T["open"]
    out.append(line("Walking, nothing around", "no escape",
                    f"**{t['n'] - t['runs_with_escape']}/{t['n']}** runs without a false escape"))
for sc, label in (("pillar R", "Pillar 6 cm right of the path"), ("pillar L", "Pillar 6 cm left of the path")):
    if sc in T:
        t = T[sc]
        out.append(line(label, "get around it",
                        f"**cleared {t['ok']}/{t['n']}**, closest gap {rng([t['gap_min'], t['gap_max']], '+.1f', ' cm')}; "
                        f"escapes in {t['runs_with_escape']}/{t['n']} runs, first escape on the pillar's side "
                        f"{t['first_escape_correct_side']}/{t['runs_with_escape']}"))
for sc, label in (("ball R", "Ball thrown at the walking duck from the right"),
                  ("ball L", "Ball thrown at the walking duck from the left"),
                  ("stand ball", "Standing duck, ball from the right (Milestone 1 check)")):
    if sc in T:
        t = T[sc]
        out.append(line(label, "escape away, ball misses",
                        f"**missed {t['ok']}/{t['n']}**, gap {rng([t['gap_min'], t['gap_max']], '+.1f', ' cm')}; "
                        f"first escape on the ball's side {t['first_escape_correct_side']}/{t['runs_with_escape']}; "
                        f"ball {rng(t['deg'], '.0f', '°')} wide, {rng(t['warning_ms'], '.0f', ' ms')} before impact"))
falls = sum(t["falls"] for t in T.values())
out.append("")
out.append(f"{len(rows)} runs ({d['repeats']} per scene), falls: {falls}. "
           f"Escape gate: LPLC2 ≥ {d['gate']} spikes in 100 ms; ball r/v {d['rv']:g} ms.")
out.append("")
out.append("<details><summary>Every run</summary>\n")
out.append("| run | outcome | gap cm | escapes (time s, side) | drift cm | turned |")
out.append("|---|---|---|---|---|---|")
for r in rows:
    esc = ", ".join(f"{e[0]:.2f} {e[1]}" for e in r["escapes"]) or "-"
    gap = "-" if r["gap_cm"] is None else f"{r['gap_cm']:+.1f}"
    out.append(f"| {r['run']} | {r['outcome']} | {gap} | {esc} | {r['drift_cm']:+.1f} | {r['turned_deg']:+.0f}° |")
out.append("\n</details>")

p = os.path.join(ROOT, "README.md")
s = open(p).read()
a, b = "<!-- M2_RESULTS_START -->", "<!-- M2_RESULTS_END -->"
if a not in s or b not in s:
    raise SystemExit("README.md has no M2_RESULTS markers")
s = s[:s.index(a) + len(a)] + "\n" + "\n".join(out) + "\n" + s[s.index(b):]
open(p, "w").write(s)
print("\n".join(out[:8]))
print(f"\nREADME.md updated ({len(rows)} runs)")
