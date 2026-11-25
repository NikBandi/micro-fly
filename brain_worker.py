"""FlyWire brain as a running service for the closed loop (brain environment).

Started by fly_loop.py. Same brain set-up as brain_bridge.py (eye-driven "puppet"
neurons, their brain inputs cut, chosen cell types silenced), but the brain keeps
running between calls: each step simulates the next chunk of time.

Speed: the network is started ONCE (one long Brian2 run) and the message exchange happens
inside it, in a network operation at the start of every 20 ms bin. Calling net.run() for every
20 ms step costs ~0.3 s of Brian2 set-up each time, which used to be most of the run time.
Code is compiled with Cython when it is available (--codegen auto); results are identical to
the pure-numpy mode, only faster. The first run compiles for ~0.5-1 min (cached afterwards).

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
ap.add_argument("--codegen", choices=["auto", "cython", "numpy"], default="auto",
                help="auto = Cython (compiled C) if it works here, else numpy")
args = ap.parse_args()

sys.path.insert(0, args.brain)
from brian2 import (Network, PoissonGroup, Synapses, network_operation, Hz, ms, mV, prefs,
                    second)
from model import create_model, default_params

target = args.codegen
if target in ("auto", "cython"):
    try:
        from brian2.codegen.runtime.cython_rt import CythonCodeObject
        ok = CythonCodeObject.is_available()
    except Exception as e:
        print(f"brain_worker: Cython check failed ({e})", file=sys.stderr)
        ok = False
    if not ok and target == "cython":
        raise SystemExit("brain_worker: Cython code generation is not available "
                         "(pip install cython; needs a C compiler)")
    target = "cython" if ok else "numpy"
prefs.codegen.target = target
print(f"brain_worker: code generation = {target}", file=sys.stderr)
BIN_DT = 0.02          # seconds per rate bin (the eyes' frame step)

READOUT = ["DNp01", "DNp02", "DNp11", "DNa01", "DNa02", "MDN", "LPLC2", "LC4", "HSE", "HSN",
           "HSS",
           # steering candidates (Milestone 2): DNa03, DNb01, DNb05 descending neurons, and the
           # visual projection neurons that feed them most (LPLC4 -> DNb05, LT51 -| DNa01)
           "DNa03", "DNb01", "DNb05", "LPLC4", "LT51"]
ann = pd.read_csv(args.annotations, sep="\t", low_memory=False,
                  usecols=["root_id", "cell_type", "side"])
comp = pd.read_csv(os.path.join(args.brain, "Completeness_783.csv"), index_col=0)
flyid2i = {fid: i for i, fid in enumerate(comp.index)}

net = src = mon = None
group_idx, groups, sides = [], [], []
m = ipc.recv(msg_in)
if str(m["cmd"]) != "init":
    raise SystemExit(f"brain_worker: expected init, got {m['cmd']}")
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

# drive: one Poisson source per puppet; rates set every 20 ms bin by the network operation
src = PoissonGroup(len(puppets), rates=0 * Hz, name="flyvis_drive")
drive = Synapses(src, neu, on_pre=f"v_post += {args.kick_mV}*mV", name="flyvis_kick")
drive.connect(i=np.arange(len(puppets)), j=puppets)
keep_cols = np.flatnonzero(keep)
for typ in READOUT:
    for side in ["right", "left"]:
        ids = [flyid2i[r] for r in ann.root_id[(ann.cell_type == typ) & (ann.side == side)]
               if r in flyid2i]
        group_idx.append(np.array(ids, int))
        groups.append(typ)
        sides.append(side)
state = {"rates": None, "k": 0, "last": np.zeros(len(comp), np.int64), "done": False}


@network_operation(dt=BIN_DT * second, when="start", name="exchange")
def exchange(t):
    """start of every 20 ms bin: when the current chunk is used up, report its spikes and wait for
    the next chunk of eye rates (fly_loop sends one only after it got the counts)"""
    if state["done"]:
        src.rates_ = 0
        return
    if state["rates"] is not None and state["k"] >= len(state["rates"]):
        count = np.asarray(mon.count[:])
        new = count - state["last"]
        state["last"] = count.copy()
        ipc.send(msg_out, counts=np.array([int(new[g].sum()) for g in group_idx]))
        state["rates"] = None
    if state["rates"] is None:
        try:
            m = ipc.recv(msg_in)
        except EOFError:                      # fly_loop finished: stop at the end of this step
            state["done"] = True
            src.rates_ = 0
            net.stop()
            return
        if str(m["cmd"]) != "step":
            raise SystemExit(f"brain_worker: unknown command {m['cmd']}")
        assert abs(float(m["dt"]) - BIN_DT) < 1e-9, "brain_worker expects 20 ms bins"
        state["rates"] = m["rates"][:, keep_cols].astype(float)
        state["k"] = 0
    src.rates_ = state["rates"][state["k"]]
    state["k"] += 1


net = Network(neu, syn, mon, src, drive, exchange)
ipc.send(msg_out, groups=np.array(groups), sides=np.array(sides))
print(f"brain_worker: ready in {time.time() - t0:.0f} s, {len(puppets):,} eye-driven "
      f"neurons", file=sys.stderr)
net.run(24 * 3600 * second, namespace={})     # ends when fly_loop closes the pipe (net.stop)
