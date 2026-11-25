"""Which inputs made a neuron fire? Trace one cell type's drive in finished runs.

For every neuron that synapses onto the target, this counts how many spikes it sent
during the run and multiplies by its connection strength (synapse count x sign x
0.275 mV, the brain model's weight per synapse). Summing by input cell type shows what
actually pushed the target toward (or away from) firing. It also prints when the target
fired, in 100 ms bins.

Usage (brain environment, from ~/flyduck/bridge, after brain_bridge.py):
    python explain_neuron.py DNp01 --side right
    python explain_neuron.py DNp01 --side right --stimuli looming sideways
    python explain_neuron.py LPLC2 --side right --top 15

Then follow the trail: run it again on the top input type to see what drove *that*.
"""
import argparse
import glob
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("cell_type", help="FlyWire cell type to explain, e.g. DNp01 (giant fiber)")
ap.add_argument("--side", choices=["right", "left", "both"], default="right")
ap.add_argument("--stimuli", nargs="*", help="stimulus labels (default: every run in --results)")
ap.add_argument("--results", default="results")
ap.add_argument("--brain", default=os.path.expanduser("~/flyduck/Drosophila_brain_model"))
ap.add_argument("--annotations", default="data/Supplemental_file1_neuron_annotations.tsv")
ap.add_argument("--top", type=int, default=10, help="how many input types to list")
args = ap.parse_args()

W_SYN = 0.275  # mV per synapse (Shiu et al. default)

ann = pd.read_csv(args.annotations, sep="\t", low_memory=False,
                  usecols=["root_id", "cell_type", "side", "super_class"])
ann["label"] = ann.cell_type.fillna(ann.super_class.fillna("untyped") + " (untyped)")
ann = ann.set_index("root_id")

sel = ann.cell_type == args.cell_type
if args.side != "both":
    sel &= ann.side == args.side
targets = set(ann.index[sel])
if not targets:
    raise SystemExit(f"No FlyWire neurons of type {args.cell_type!r} on side {args.side}.")

con = pd.read_parquet(os.path.join(args.brain, "Connectivity_783.parquet"),
                      columns=["Presynaptic_ID", "Postsynaptic_ID", "Excitatory x Connectivity"])
inp = con[con.Postsynaptic_ID.isin(targets)]
w = inp.groupby("Presynaptic_ID")["Excitatory x Connectivity"].sum() * W_SYN   # mV per spike

if args.stimuli:
    files = [os.path.join(args.results, f"{s}_spikes.parquet") for s in args.stimuli]
else:
    files = sorted(glob.glob(os.path.join(args.results, "*_spikes.parquet")))

print(f"\n{args.cell_type} [{args.side}]: {len(targets)} neuron(s), "
      f"{len(w):,} input neurons, {int((w > 0).sum()):,} excitatory / "
      f"{int((w < 0).sum()):,} inhibitory")

for f in files:
    label = os.path.basename(f).replace("_spikes.parquet", "")
    s = pd.read_parquet(f)
    n_trials = s.trial.nunique() if "trial" in s else 1
    T = s.t.max()

    # target firing over time
    ts = s[s.flywire_id.isin(targets)]
    bins = np.arange(0, np.ceil(T * 10) / 10 + 0.1, 0.1)
    hist = np.histogram(ts.t, bins)[0] / n_trials / len(targets)
    print(f"\n=== {label} ===  {args.cell_type} spikes per neuron per trial: "
          f"{len(ts) / n_trials / len(targets):.1f}")
    print("  timeline (spikes per 100 ms):  " +
          " ".join("." if h == 0 else f"{h:.1f}" for h in hist) +
          f"   [0 … {bins[-1]:.1f} s]")

    # input drive by cell type
    counts = s[s.flywire_id.isin(w.index)].groupby("flywire_id").size() / n_trials
    puppet_ids = set(s.flywire_id[s.puppet]) if "puppet" in s else set()
    df = pd.DataFrame({"weight_mV": w, "spikes": counts.reindex(w.index).fillna(0)})
    df["drive_mV"] = df.weight_mV * df.spikes
    df["type"] = ann.label.reindex(df.index).fillna("unknown").values
    df["side"] = ann.side.reindex(df.index).fillna("?").values
    df["eye_driven"] = df.index.isin(puppet_ids)
    g = df.groupby(["type", "side"]).agg(
        neurons=("spikes", "size"), active=("spikes", lambda x: int((x > 0).sum())),
        spikes=("spikes", "sum"), drive_mV=("drive_mV", "sum"),
        eye_driven=("eye_driven", "any"))
    total_exc = g.drive_mV[g.drive_mV > 0].sum()
    total_inh = g.drive_mV[g.drive_mV < 0].sum()
    print(f"  total drive per trial: +{total_exc:.0f} mV excitatory, {total_inh:.0f} mV inhibitory")

    def show(part, title):
        print(f"  {title}")
        print(f"    {'input type':28s} {'side':6s} {'active/all':>10s} {'spikes':>8s} "
              f"{'drive mV':>9s} {'share':>6s}  source")
        for (typ, side), r in part.iterrows():
            share = r.drive_mV / (total_exc if r.drive_mV > 0 else total_inh) * 100
            src = "fly eye (flyvis)" if r.eye_driven else "brain"
            print(f"    {typ[:28]:28s} {side:6s} {int(r.active):>4d}/{int(r.neurons):<5d} "
                  f"{r.spikes:8.0f} {r.drive_mV:+9.1f} {share:5.0f}%  {src}")

    show(g[g.drive_mV > 0].sort_values("drive_mV", ascending=False).head(args.top),
         "top EXCITATORY inputs (pushing it to fire):")
    show(g[g.drive_mV < 0].sort_values("drive_mV").head(max(3, args.top // 2)),
         "top INHIBITORY inputs (holding it back):")
