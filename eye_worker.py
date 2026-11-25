"""Fly eyes as a running service for the closed loop (vision environment).

Started by fly_loop.py. Keeps the flyvis network's state between calls, so the eyes
see one continuous movie, 5 frames at a time.

Messages in -> out:
  {cmd: "init", frames: [1, H, W]}  the empty scene (no objects) the duck sees at rest.
      The eyes adapt to it for 2 s; its steady response becomes the 0 Hz baseline.
      -> {root_ids: [N]}  FlyWire neurons that will be driven
  {cmd: "step", frames: [k, H, W]}  next k camera frames (50 frames/s)
      -> {rates: [k, N],  firing rate in Hz for each driven FlyWire neuron
          drive: [k, D]}  summed rate (Hz) of the looming-pathway input types per eye
                          (names in drive_names, sent with the init reply)
--gain G multiplies every cell's response before the 0..RMAX clip (default 1 = as calibrated).
Frames are left eye | right eye side by side, values 0..1, as made by fly_loop.py.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ipc

msg_in, msg_out = ipc.protocol_streams()

import numpy as np
import pandas as pd
import scipy.ndimage as nd
import torch
import flyvis
from flyvis import Network, NetworkView
from flyvis.datasets.rendering import BoxEye

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(flyvis.results_dir / "flow/0000/000"))
ap.add_argument("--untrained", action="store_true")
ap.add_argument("--gain", type=float, default=1.0, help="eye -> brain drive multiplier")
args = ap.parse_args()

DT, RMAX = 0.02, 150.0
meta = json.load(open(os.path.join(HERE, "bridge_meta.json")))
bridge = pd.read_csv(os.path.join(HERE, "bridge_map.csv"))
bridge = bridge[bridge.flyvis_index != -1]
dev = "cuda" if torch.cuda.is_available() else "cpu"

if args.untrained:
    net, tag = Network(), "untrained"
else:
    net = NetworkView(args.model).init_network()
    tag = os.path.basename(os.path.dirname(os.path.dirname(args.model))) + "_" + \
          os.path.basename(args.model)
net = net.to(dev).eval()
cal_path = os.path.join(HERE, f"calibration_v3_{tag}.npz")
if not os.path.exists(cal_path):
    raise SystemExit(f"eye_worker: {cal_path} not found. Run fly_eye.py once first "
                     f"(for example: python fly_eye.py --stimulus looming) to create it.")
scale = np.load(cal_path)["scale"]
fv_types = np.array(net.connectome.nodes.type[:]).astype(str)
eye = BoxEye(extent=15, kernel_size=13)

# which flyvis cell drives which FlyWire neuron (same order as fly_eye.py)
# 'pick' lists, in output order, how to get each column: (eye, flyvis indices or mask, pooled?)
pick, ids = [], []
for b_i, side in enumerate(["left", "right"]):
    s = bridge[bridge.side == side]
    single = s[s.flyvis_index >= 0]
    pick.append((b_i, single.flyvis_index.values, False))
    ids.append(single.flywire_root_id.values)
    for r in s[s.flyvis_index == -2].itertuples():          # pooled wide-field cells
        pick.append((b_i, fv_types == r.flyvis_type, True))
        ids.append(np.array([r.flywire_root_id]))
root_ids = np.concatenate(ids)

# summary of the drive into the looming pathway, per eye: T4/T5 feed LPLC2; T2, Tm4, TmY3
# feed LC4; TmY5a is the main inhibitory eye input onto LPLC2
DRIVE = {"T4": ["T4a", "T4b", "T4c", "T4d"], "T5": ["T5a", "T5b", "T5c", "T5d"],
         "T2": ["T2"], "Tm4": ["Tm4"], "TmY3": ["TmY3"], "TmY5a": ["TmY5a"]}
types_out, sides_out = [], []
for b_i, side in enumerate(["left", "right"]):
    s = bridge[bridge.side == side]
    types_out.append(s[s.flyvis_index >= 0].flyvis_type.values)
    sides_out.append(np.repeat(side, (s.flyvis_index >= 0).sum()))
    for r in s[s.flyvis_index == -2].itertuples():
        types_out.append(np.array([r.flyvis_type]))
        sides_out.append(np.array([side]))
types_out, sides_out = np.concatenate(types_out), np.concatenate(sides_out)
assert len(types_out) == len(root_ids)
drive_names, drive_cols = [], []
for side in ["left", "right"]:
    for name, members in DRIVE.items():
        drive_names.append(f"eye_{name}:{side}")
        drive_cols.append(np.flatnonzero(np.isin(types_out, members) & (sides_out == side)))


def to_eyes(frames):
    """[T,H,W] left|right frames -> tensor [2, T, 1, 721] (same transform as fly_eye.py)."""
    w = frames.shape[2]
    halves = {"left": frames[:, :, : w // 2], "right": frames[:, :, w - w // 2:]}
    out = []
    for side in ["left", "right"]:
        c = meta["camera"][f"{side}_eye"]
        x = halves[side]
        if c["mirror_horizontally_first"]:
            x = x[:, :, ::-1]
        x = nd.rotate(x, c["then_rotate_ccw_deg"], axes=(2, 1), reshape=False, order=1,
                      mode="nearest")
        out.append(eye(torch.tensor(np.ascontiguousarray(x))[None]).cpu())
    return torch.cat(out, 0)


state, v_base = None, None
while True:
    try:
        m = ipc.recv(msg_in)
    except EOFError:
        break
    cmd = str(m["cmd"])
    with torch.no_grad():
        if cmd == "init":
            x = to_eyes(m["frames"].astype(np.float32)).to(dev)            # [2,1,1,721]
            state = net.fade_in_state(t_fade_in=1.0, dt=DT, initial_frames=x[:, 0])
            settle = x.repeat(1, int(1.0 / DT), 1, 1)                     # 1 s of stillness
            states = net.simulate(settle, dt=DT, initial_state=state, as_states=True)
            state = states[-1]
            v_base = state.nodes.activity.cpu().numpy()                     # [2, 45669]
            ipc.send(msg_out, root_ids=root_ids, drive_names=np.array(drive_names))
            print(f"eye_worker: ready ({tag}, {dev}, gain {args.gain:g}), {len(root_ids):,} "
                  f"FlyWire neurons", file=sys.stderr)
        elif cmd == "step":
            x = to_eyes(m["frames"].astype(np.float32)).to(dev)
            states = net.simulate(x, dt=DT, initial_state=state, as_states=True)
            v = torch.stack([s.nodes.activity for s in states], 1).cpu().numpy()  # [2,k,45669]
            state = states[-1]
            rate = RMAX * np.clip(args.gain * (v - v_base[:, None]) / scale, 0, 1)
            cols = [rate[b][:, sel].mean(1, keepdims=True) if pooled else rate[b][:, sel]
                    for b, sel, pooled in pick]
            out = np.concatenate(cols, 1).astype(np.float32)
            assert out.shape[1] == len(root_ids)
            drive = np.stack([out[:, c].sum(1) for c in drive_cols], 1).astype(np.float32)
            ipc.send(msg_out, rates=out, drive=drive)
        else:
            raise SystemExit(f"eye_worker: unknown command {cmd}")
