"""FlyWire brain as a running service for the closed loop (brain environment).

Started by fly_loop.py. Same brain set-up as brain_bridge.py (eye-driven "puppet"
neurons, their brain inputs cut, chosen cell types silenced), but the brain keeps
running between calls: each step simulates the next chunk of time.

Messages in -> out:
  {cmd: "init", root_ids: [N]}       FlyWire neurons the eyes will drive
      -> {groups: [G], sides: [G]}   the readout groups reported by every step
  {cmd: "step", rates: [k, N], dt}   firing rates (Hz) for the next k bins of dt seconds
      -> {counts: [G]}              spikes per readout group during those k bins
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ipc

msg_in, msg_out = ipc.protocol_streams()

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--brain", default=os.path.expanduser("~/flyduck/Drosophila_brain_model"))
ap.add_argument("--annotations", default=os.path.join(HERE, "data/Supplemental_file1_neuron_annotations.tsv"))
ap.add_argument("--silence", nargs="*", default=["DNp70", "Li33"])
ap.add_argument("--kick_mV", type=float, default=20.0)
args = ap.parse_args()

sys.path.insert(0, args.brain)
from brian2 import (Network, PoissonGroup, Synapses, network_operation, Hz, ms, mV, prefs,
                    second)
from model import create_model, default_params

prefs.codegen.target = "numpy"
BIN_DT = 0.02          # seconds per rate bin (the eyes' frame step)

READOUT = ["DNp01", "DNp02", "DNp11", "DNa01", "DNa02", "MDN", "LPLC2", "LC4", "HSE", "HSN",
           "HSS"]
ann = pd.read_csv(args.annotations, sep="\t", low_memory=False,
                  usecols=["root_id", "cell_type", "side"])
comp = pd.read_csv(os.path.join(args.brain, "Completeness_783.csv"), index_col=0)
flyid2i = {fid: i for i, fid in enumerate(comp.index)}

net = src = mon = None
group_idx, groups, sides = [], [], []
last_count = None
while True:
    try:
        m = ipc.recv(msg_in)
    except EOFError:
        break
    cmd = str(m["cmd"])
    if cmd == "init":
        t0 = time.time()
        root_ids = m["root_ids"]
        keep = np.array([r in flyid2i for r in root_ids])
        puppets = np.array([flyid2i[r] for r in root_ids[keep]])
        neu, syn, mon = create_model(os.path.join(args.brain, "Completeness_783.csv"),
                                     os.path.join(args.brain, "Connectivity_783.parquet"),
                                     dict(default_params))
        pre, post = np.asarray(syn.i[:]), np.asarray(syn.j[:])
        is_puppet = np.zeros(len(comp), bool)
        is_puppet[puppets] = True
        syn.w[np.flatnonzero(is_puppet[post])] = 0 * mV        # integer indices (see brain_bridge)
        for item in args.silence:
            typ, _, side = item.partition(":")
            sel = ann.cell_type == typ
            if side:
                sel &= ann.side == side
            ids = [flyid2i[r] for r in ann.root_id[sel] if r in flyid2i]
            if not ids:
                raise SystemExit(f"brain_worker: --silence {item}: no neurons found")
            is_sil = np.zeros(len(comp), bool)
            is_sil[ids] = True
            syn.w[np.flatnonzero(is_sil[pre])] = 0 * mV
            print(f"brain_worker: silenced {item} ({len(ids)} neurons)", file=sys.stderr)
        w_now = np.asarray(syn.w[:] / mV)
        assert np.abs(w_now[is_puppet[post]]).sum() == 0, "puppet input cut failed"

        # drive: one Poisson source per puppet; rates updated every bin by a network operation
        src = PoissonGroup(len(puppets), rates=0 * Hz, name="flyvis_drive")
        drive = Synapses(src, neu, on_pre=f"v_post += {args.kick_mV}*mV", name="flyvis_kick")
        drive.connect(i=np.arange(len(puppets)), j=puppets)
        pending = {"rates": np.zeros((1, len(puppets))), "k": 0}
        keep_cols = np.flatnonzero(keep)

        @network_operation(dt=BIN_DT * second, when="start", name="update_rates")
        def update_rates(t):
            r = pending["rates"]
            src.rates_ = r[min(pending["k"], len(r) - 1)]
            pending["k"] += 1

        net = Network(neu, syn, mon, src, drive, update_rates)
        for typ in READOUT:
            for side in ["right", "left"]:
                ids = [flyid2i[r] for r in ann.root_id[(ann.cell_type == typ) & (ann.side == side)]
                       if r in flyid2i]
                group_idx.append(np.array(ids, int))
                groups.append(typ)
                sides.append(side)
        last_count = np.zeros(len(comp), np.int64)
        ipc.send(msg_out, groups=np.array(groups), sides=np.array(sides))
        print(f"brain_worker: ready in {time.time() - t0:.0f} s, {len(puppets):,} eye-driven "
              f"neurons", file=sys.stderr)
    elif cmd == "step":
        rates = m["rates"][:, keep_cols].astype(float)
        assert abs(float(m["dt"]) - BIN_DT) < 1e-9, "brain_worker expects 20 ms bins"
        pending["rates"] = rates            # update_rates applies one row per 20 ms bin
        pending["k"] = 0
        net.run(rates.shape[0] * BIN_DT * second, namespace={})
        assert pending["k"] == rates.shape[0], "rate bins and simulated time got out of step"
        count = np.asarray(mon.count[:])
        new = count - last_count
        last_count = count.copy()
        ipc.send(msg_out, counts=np.array([int(new[g].sum()) for g in group_idx]))
    else:
        raise SystemExit(f"brain_worker: unknown command {cmd}")
