"""Run the FlyWire whole-brain model with its optic lobe driven by flyvis.

Every FlyWire neuron listed in bridge_map.csv becomes a "puppet": its own inputs from
the brain are cut, and it spikes according to the firing rates in a rates file
(produced by fly_eye.py from the real flyvis eye, or by make_test_rates.py for an
idealised test). Everything downstream (LPLC2, LC4, the giant fiber, HS/VS cells,
steering neurons...) runs normally on FlyWire's own wiring.

Usage (inside the Drosophila_brain_model environment):
    python brain_bridge.py RATES.npz [--brain ~/flyduck/Drosophila_brain_model] [--out results/]

Rates file (npz):
    rates     float32 [T, N]  firing rate in Hz for each time bin and puppet neuron
    root_ids  int64   [N]     FlyWire root id of each puppet neuron
    dt        float           bin width in seconds (flyvis uses 0.02)
    label     str             a name for this stimulus
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("rates", nargs="+", help="one or more rates .npz files")
ap.add_argument("--brain", default=os.path.expanduser("~/flyduck/Drosophila_brain_model"))
ap.add_argument("--annotations", default="data/Supplemental_file1_neuron_annotations.tsv")
ap.add_argument("--out", default="results")
ap.add_argument("--trials", type=int, default=5,
                help="repeat each stimulus this many times (input spikes are random) and average")
ap.add_argument("--silence", nargs="*", default=[],
                help="cell types to silence, e.g. DNp70 or DNp70:right. Their outgoing synapses "
                     "are set to zero (like genetically blocking their output in a real fly)")
ap.add_argument("--kick_mV", type=float, default=20.0,
                help="voltage jump per input spike; 20 mV > the 7 mV to threshold, so each "
                     "input spike makes the puppet neuron fire once")
args = ap.parse_args()

sys.path.insert(0, args.brain)
from brian2 import (Network, PoissonGroup, SpikeMonitor, Synapses, TimedArray, Hz, ms, mV,
                    prefs, second)
from model import create_model, default_params

prefs.codegen.target = "numpy"   # avoids the Cython warning; switch to "cython" if installed

# Readout neurons: name -> FlyWire cell types
READOUT = {
    "LPLC2 (looming)": ["LPLC2"],
    "LC4 (looming)": ["LC4"],
    "Giant fiber DNp01 (escape)": ["DNp01"],
    "DNp02 (looming DN)": ["DNp02"],
    "DNp11 (looming DN)": ["DNp11"],
    "HS cells (horizontal motion)": ["HSE", "HSN", "HSS"],
    "DNa01 (steering)": ["DNa01"],
    "DNa02 (steering)": ["DNa02"],
    "MDN (backward walking)": ["MDN"],
}

ann = pd.read_csv(args.annotations, sep="\t", low_memory=False,
                  usecols=["root_id", "cell_type", "side"])
comp = pd.read_csv(os.path.join(args.brain, "Completeness_783.csv"), index_col=0)
flyid2i = {fid: i for i, fid in enumerate(comp.index)}
os.makedirs(args.out, exist_ok=True)

summary = []
for path in args.rates:
    d = np.load(path, allow_pickle=True)
    rate_arr, root_ids, bin_dt = d["rates"].astype(float), d["root_ids"], float(d["dt"])
    label = str(d["label"]) if "label" in d else os.path.basename(path)
    keep = np.array([r in flyid2i for r in root_ids])
    rate_arr, root_ids = rate_arr[:, keep], root_ids[keep]
    puppets = np.array([flyid2i[r] for r in root_ids])
    T = rate_arr.shape[0] * bin_dt
    if args.silence:
        label = label + "__silence_" + "_".join(x.replace(":", "-") for x in args.silence)
    print(f"\n=== {label}: {len(puppets):,} puppet neurons, {T:.2f} s ===")

    t0 = time.time()
    params = dict(default_params)
    neu, syn, spk_mon = create_model(os.path.join(args.brain, "Completeness_783.csv"),
                                     os.path.join(args.brain, "Connectivity_783.parquet"),
                                     params)
    # cut the brain's own inputs onto puppet neurons
    is_puppet = np.zeros(len(comp), bool)
    is_puppet[puppets] = True
    # (Brian2 needs integer indices here; a boolean mask is silently ignored)
    syn.w[np.flatnonzero(is_puppet[np.asarray(syn.j[:])])] = 0 * mV
    # silence chosen cell types: zero all their outgoing synapses
    for item in args.silence:
        typ, _, side = item.partition(":")
        m = ann.cell_type == typ
        if side:
            m &= ann.side == side
        ids = [flyid2i[r] for r in ann.root_id[m] if r in flyid2i]
        if not ids:
            raise SystemExit(f"--silence {item}: no FlyWire neurons found")
        is_sil = np.zeros(len(comp), bool)
        is_sil[ids] = True
        syn.w[np.flatnonzero(is_sil[np.asarray(syn.i[:])])] = 0 * mV
        print(f"silenced {item}: {len(ids)} neuron(s)")

    w_now = np.asarray(syn.w[:] / mV)
    assert np.abs(w_now[is_puppet[np.asarray(syn.j[:])]]).sum() == 0, "puppet input cut failed"

    # time-varying Poisson drive: one source per puppet, one-to-one kick
    rate_ta = TimedArray(rate_arr * Hz, dt=bin_dt * second)
    src = PoissonGroup(len(puppets), rates="rate_ta(t, i)",
                       namespace={"rate_ta": rate_ta}, name="flyvis_drive")
    drive = Synapses(src, neu, on_pre=f"v_post += {args.kick_mV}*mV", name="flyvis_kick")
    drive.connect(i=np.arange(len(puppets)), j=puppets)

    net = Network(neu, syn, spk_mon, src, drive)
    net.store("start")
    trial_spikes = []
    for trial in range(args.trials):
        net.restore("start")                 # same brain, fresh random input spikes
        net.run(T * second, namespace={})
        spk_i = np.asarray(spk_mon.i[:])
        trial_spikes.append(pd.DataFrame({
            "trial": trial, "t": np.asarray(spk_mon.t[:] / second),
            "flywire_id": comp.index.values[spk_i], "puppet": is_puppet[spk_i]}))
    print(f"simulated {args.trials} trial(s) in {time.time() - t0:.0f} s")
    spikes = pd.concat(trial_spikes, ignore_index=True)
    spikes.to_parquet(os.path.join(args.out, f"{label}_spikes.parquet"))

    row = {"stimulus": label}
    for name, types in READOUT.items():
        for side in ["right", "left"]:
            ids = set(ann[ann.cell_type.isin(types) & (ann.side == side)].root_id)
            n = int(spikes.flywire_id.isin(ids).sum())
            row[f"{name} [{side}]"] = n / max(len(ids), 1) / T / args.trials  # mean Hz/neuron
    gf = set(ann[(ann.cell_type == "DNp01") & (ann.side == "right")].root_id)
    per_trial = spikes[spikes.flywire_id.isin(gf)].groupby("trial").size() \
        .reindex(range(args.trials), fill_value=0).tolist()
    row["giant fiber [right] spikes per trial"] = str(per_trial)
    row["puppet spikes per trial"] = int(spikes.puppet.sum() / args.trials)
    summary.append(row)

tab = pd.DataFrame(summary).set_index("stimulus").T
pd.set_option("display.width", 160)
print(f"\nMean firing rate per neuron (Hz), averaged over {args.trials} trials:")
print(tab.to_string(float_format=lambda x: f"{x:.1f}"))
tab.to_csv(os.path.join(args.out, "summary.csv"))
