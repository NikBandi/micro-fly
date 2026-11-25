"""Idealised T4/T5 responses for three stimuli on the RIGHT eye, without flyvis.

Tests the brain side of the bridge on its own. A dark disc's edge passes over each
column; motion detectors there fire in proportion to how well the local edge motion
matches their preferred direction (measured from FlyWire anatomy). OFF-edge cells
(T5) get full rate, ON-edge cells (T4) half.

  looming   : disc expands from the eye centre, accelerating (approaching object)
  receding  : the same movie played backwards (edges move inward)
  sideways  : a dark bar sweeps front-to-back across the eye (no expansion)

Writes rates_ideal_<name>.npz in the format brain_bridge.py reads.
"""
import json

import numpy as np
import pandas as pd

DT, T, RMAX = 0.02, 1.0, 150.0
b = pd.read_csv("bridge_map.csv")
b = b[(b.flyvis_index >= 0) & (b.side == "right")]
meta = json.load(open("bridge_meta.json"))
fw = meta["alignment"]["right"]["flywire_pd_angles_deg"]
centre = np.array(meta["centre_pq"]["right"], float)


def cart(p, q):
    return np.stack([(q - p) / 2.0, (p + q) * np.sqrt(3) / 2], -1)


XY = cart(b.p.values - centre[0], b.q.values - centre[1])
pd_vec = np.zeros_like(XY)
is_md = b.flyvis_type.str.match(r"^T[45][abcd]$").values
ang = np.radians([fw.get(t, 0.0) for t in b.flyvis_type])
pd_vec[is_md] = np.stack([np.cos(ang), np.sin(ang)], 1)[is_md]
gain = np.where(b.flyvis_type.str.startswith("T5"), 1.0, 0.5) * is_md

steps = int(T / DT)
t = (np.arange(steps) + 0.5) * DT
dist = np.linalg.norm(XY, axis=1)
radial = XY / np.maximum(dist, 1e-6)[:, None]


def rates_for(edge_pos, proj, edge_dir):
    """edge_pos(t) -> edge position along proj; edge_dir: per-neuron unit motion vector."""
    R = np.zeros((steps, len(b)), np.float32)
    match = np.clip((edge_dir * pd_vec).sum(1), 0, None) * gain
    for k, tk in enumerate(t):
        near = np.abs(edge_pos(tk) - proj) < 1.0
        R[k] = RMAX * match * near
    return R


# looming: radius r(t) = r0 / (1 - t/Tc), capped at the eye edge (classic l/v loom)
loom_r = lambda tk: min(1.0 / max(1 - tk / 1.05, 1e-3), 30.0)
loom = rates_for(loom_r, dist, radial)
# receding: same edge positions in reverse order, edges moving inward
reced = rates_for(lambda tk: loom_r(T - tk), dist, -radial)
# sideways: bar edge sweeping along front-to-back (the T4a/T5a direction) at constant speed
ftb = np.array([np.cos(np.radians(fw["T5a"])), np.sin(np.radians(fw["T5a"]))])
side = rates_for(lambda tk: -20 + 40 * tk, XY @ ftb, np.tile(ftb, (len(b), 1)))

for name, R in [("looming", loom), ("receding", reced), ("sideways", side)]:
    np.savez_compressed(f"rates_ideal_{name}.npz", rates=R, root_ids=b.flywire_root_id.values,
                        dt=DT, label=f"ideal_{name}")
    print(f"{name:9s}: {(R > 0).any(0).sum():5d} neurons active, mean {R.mean():.2f} Hz")
