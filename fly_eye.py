"""The fly's eyes: camera frames -> flyvis -> firing rates for the FlyWire brain.

Pipeline for each camera frame:
  1. split the image into left and right halves (one per compound eye)
  2. mirror/rotate each half so the world's "up" and "front-to-back" land where FlyWire's
     anatomy expects them (values from bridge_meta.json)
  3. BoxEye samples each half onto flyvis's 721 hexagonal columns
  4. the pretrained flyvis network simulates all 45,669 visual neurons (both eyes as a batch)
  5. each cell's voltage is converted to a firing rate: 0 Hz at baseline, rising to RMAX at
     that cell type's strong-response level (from calibration). Baseline = a grey screen
     for the built-in test movies, or, for --frames, the empty scene held still (subtracted frame by frame)
  6. rates are picked out for the FlyWire neurons listed in bridge_map.csv

Usage (inside ~/flyduck/.venv-vision):
    python fly_eye.py --stimulus looming          # built-in test movies: looming, receding,
                                                  #   sideways, grey
    python fly_eye.py --stimulus directions       # check each motion detector's direction
    python fly_eye.py --frames duck_camera.npz    # your own frames: 'frames' [T,H,W] in 0..1, 'dt'
Options:
    --model PATH      flyvis model folder (default: pretrained flow/0000/000)
    --untrained       use a randomly initialised network (pipeline testing only)
    --out FILE        output rates file (default rates_<name>.npz)

The first run also computes calibration_<model>.npz (grey baseline + per-type response
scale) from a set of standard movies, and saves eye_check.png to verify orientation.
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.ndimage as nd
import torch

import flyvis
from flyvis import Network, NetworkView
from flyvis.datasets.rendering import BoxEye

DT = 0.02          # flyvis frame / integration step (s)
RMAX = 150.0       # Hz at a cell type's strong-response level (Shiu's default drive rate)
H, W = 400, 800    # built-in stimulus size: two 400x400 halves
GREY = 0.5

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--stimulus", choices=["looming", "receding", "sideways", "grey", "directions"])
g.add_argument("--frames")
ap.add_argument("--model", default=str(flyvis.results_dir / "flow/0000/000"))
ap.add_argument("--untrained", action="store_true")
ap.add_argument("--out")
args = ap.parse_args()

meta = json.load(open("bridge_meta.json"))
bridge = pd.read_csv("bridge_map.csv")
bridge = bridge[bridge.flyvis_index != -1]
dev = "cuda" if torch.cuda.is_available() else "cpu"


# --- stimuli -----------------------------------------------------------------------
def disc_movie(radii):
    """Dark disc centred in the right half (right eye), grey everywhere else."""
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W * 3 / 4
    f = np.full((len(radii), H, W), GREY, np.float32)
    for k, r in enumerate(radii):
        f[k][(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 0.0
    return f


def hold_end(f, t_hold=0.3):
    """Append t_hold seconds of the last frame (late responses are not cut off)."""
    return np.concatenate([f, np.repeat(f[-1:], int(t_hold / DT), 0)], 0)


def builtin(name, T=1.0):
    return hold_end(_builtin(name, T))


def _builtin(name, T=1.0):
    n = int(T / DT)
    t = (np.arange(n) + 0.5) * DT
    loom = np.minimum(10.0 / np.maximum(1 - t / 1.05, 1e-3), 400.0)   # approaching object
    if name == "looming":
        return disc_movie(loom)
    if name == "receding":
        return disc_movie(loom[::-1])
    if name == "grey":
        return np.full((n, H, W), GREY, np.float32)
    if name == "sideways":            # dark bar sweeping front-to-back over the right eye
        f = np.full((n, H, W), GREY, np.float32)
        for k, tk in enumerate(t):
            x0 = int(W / 2 - 60 + 460 * tk)
            f[k, :, max(x0, W // 2):max(min(x0 + 60, W), W // 2)] = 0.0
        return f
    raise ValueError(name)


def gratings(T=1.0):
    """Calibration: square-wave gratings drifting in 4 directions, both halves."""
    n = int(T / DT)
    yy, xx = np.mgrid[0:H, 0:W]
    out = []
    for ang in [0, 90, 180, 270]:
        a = np.radians(ang)
        ph = (xx * np.cos(a) + yy * np.sin(a))[None] / 40.0 - np.arange(n)[:, None, None] * 0.25
        out.append(((np.sin(2 * np.pi * ph) > 0) * 1.0).astype(np.float32))
    return out


# --- frames -> two eyes on the hex lattice ----------------------------------------------
eye = BoxEye(extent=15, kernel_size=13)


def to_eyes(frames):
    """[T,H,W] camera frames -> tensor [2 (left,right), T, 1, 721]."""
    T_, h, w = frames.shape
    halves = {"left": frames[:, :, : w // 2], "right": frames[:, :, w - w // 2:]}
    out = []
    for side in ["left", "right"]:
        c = meta["camera"][f"{side}_eye"]
        x = halves[side]
        if c["mirror_horizontally_first"]:
            x = x[:, :, ::-1]
        x = nd.rotate(x, c["then_rotate_ccw_deg"], axes=(2, 1), reshape=False,
                      order=1, mode="nearest")
        out.append(eye(torch.tensor(np.ascontiguousarray(x))[None]))   # [1,T,1,721]
    return torch.cat(out, 0)


# --- network -------------------------------------------------------------------------
if args.untrained:
    net = Network()
    tag = "untrained"
else:
    net = NetworkView(args.model).init_network()
    tag = os.path.basename(os.path.dirname(os.path.dirname(args.model))) + "_" + \
          os.path.basename(args.model)
net = net.to(dev).eval()
fv_types = np.array(net.connectome.nodes.type[:]).astype(str)
assert len(fv_types) == 45669


@torch.no_grad()
def run(frames):
    x = to_eyes(frames).to(dev)
    # start from a state adapted to the first frame (contrast faded in over 1 s), so the
    # movie's first frame is not a sudden flash
    init = net.fade_in_state(t_fade_in=1.0, dt=DT, initial_frames=x[:, 0])
    return net.simulate(x, dt=DT, initial_state=init).cpu().numpy()     # [2, T, 45669]


# --- calibration: grey baseline + per-type scale ------------------------------------------
cal_path = f"calibration_v3_{tag}.npz"
if os.path.exists(cal_path):
    cal = np.load(cal_path)
    v_base, scale = cal["v_base"], cal["scale"]
else:
    print("calibrating (first run only)...")
    v_base = run(builtin("grey"))[:, -1].mean(0)                         # [45669]
    dev_pos = []
    for mv in gratings() + [builtin("looming"), builtin("sideways")]:
        dev_pos.append(np.clip(run(mv) - v_base, 0, None).reshape(-1, 45669))
    dev_pos = np.concatenate(dev_pos, 0)
    scale = np.zeros(45669, np.float32)
    for t in np.unique(fv_types):
        k = fv_types == t
        s = np.percentile(dev_pos[:, k], 99)
        scale[k] = s if s > 1e-6 else 1.0
    np.savez(cal_path, v_base=v_base, scale=scale)
    print(f"saved {cal_path}")

# --- direction check: which way does each motion detector prefer? ---------------------
if args.stimulus == "directions":
    yy, xx = np.mgrid[0:H, 0:W // 2]
    angles = np.arange(0, 360, 45)
    md = [f"T{k}{d}" for k in "45" for d in "abcd"]
    resp = {t: [] for t in md}
    n = int(1.0 / DT)
    for ang in angles:               # world direction: 0 = front-to-back, 90 = up
        a = np.radians(ang)
        ph = (xx * np.cos(a) - yy * np.sin(a))[None] / 50.0 - np.arange(n)[:, None, None] * 0.1
        right = ((np.sin(2 * np.pi * ph) > 0) * 0.8 + 0.1).astype(np.float32)   # 5 Hz drift
        f = np.full((n, H, W), GREY, np.float32)
        f[:, :, W // 2:] = right
        r = RMAX * np.clip((run(f) - v_base) / scale, 0, 1)[1, int(0.3 / DT):]  # right eye
        for t in md:
            resp[t].append(r[:, fv_types == t].mean())
    want = {"a": 0, "b": 180, "c": 90, "d": 270}
    print("\nMean rate (Hz) of right-eye motion detectors for gratings drifting in each "
          "world direction\n(0 = front-to-back, 90 = up, 180 = back-to-front, 270 = down)\n")
    print("type  " + "".join(f"{a:>6d}" for a in angles) + "   preferred  expected")
    ok = 0
    for t in md:
        v = np.array(resp[t])
        pref = np.degrees(np.angle((v * np.exp(1j * np.radians(angles))).sum())) % 360
        err = abs((pref - want[t[-1]] + 180) % 360 - 180)
        ok += err <= 45
        print(f"{t:5s} " + "".join(f"{x:6.1f}" for x in v) +
              f"   {pref:6.0f}°   {want[t[-1]]:5d}°  {'OK' if err <= 45 else 'off by %.0f°' % err}")
    print(f"\n{ok}/8 motion detector types prefer the expected direction (within 45°).")
    # Suggest an extra rotation (and mirror) of the camera half-image that would best line
    # up the measured preferences with the expected ones.
    prefs = {}
    for t in md:
        v = np.array(resp[t])
        prefs[t] = np.angle((v * np.exp(1j * np.radians(angles))).sum())
    best = None
    for mirror in (False, True):
        d = [(-p if mirror else p) - np.radians(want[t[-1]]) for t, p in prefs.items()]
        delta = np.angle(np.exp(1j * np.array(d)).mean())
        resid = np.degrees(np.abs(np.angle(np.exp(1j * (np.array(d) - delta))))).mean()
        if best is None or resid < best[2]:
            best = (mirror, np.degrees(delta), resid)
    print(f"Best fit: mirror={best[0]}, extra rotation {best[1]:.0f}°, "
          f"remaining average error {best[2]:.0f}°.")
    print("Check this table before changing anything.")
    raise SystemExit

# --- this movie ------------------------------------------------------------------------
if args.stimulus:
    frames, name, dt_in = builtin(args.stimulus), args.stimulus, DT
else:
    d = np.load(args.frames)
    frames, dt_in = d["frames"].astype(np.float32), float(d["dt"])
    background = d["background"].astype(np.float32) if "background" in d else frames[0]
    name = os.path.splitext(os.path.basename(args.frames))[0]
    if abs(dt_in - DT) > 1e-6:                  # resample to 50 Hz
        idx = np.round(np.arange(0, len(frames) * dt_in, DT) / dt_in).astype(int)
        frames = frames[np.clip(idx, 0, len(frames) - 1)]

v = run(frames)                                                     # [2, T, 45669]
if args.frames:
    # Camera scenes are not mid-grey, so the 0 Hz baseline is measured on the scene
    # itself: the empty environment (no objects; saved by duck_eyes.py as 'background',
    # otherwise the first frame) held still, run for the same time and subtracted frame
    # by frame. What remains is caused by objects and their motion.
    still = run(np.repeat(background[None], len(frames), 0))         # [2, T, 45669]
    rate = RMAX * np.clip((v - still) / scale, 0, 1)
else:
    rate = RMAX * np.clip((v - v_base) / scale, 0, 1)               # Hz

# --- pick out the FlyWire puppets ----------------------------------------------------------
cols, ids = [], []
for b_i, side in enumerate(["left", "right"]):
    s = bridge[bridge.side == side]
    single = s[s.flyvis_index >= 0]
    cols.append(rate[b_i][:, single.flyvis_index.values])
    ids.append(single.flywire_root_id.values)
    for r in s[s.flyvis_index == -2].itertuples():                  # pooled wide-field cells
        cols.append(rate[b_i][:, fv_types == r.flyvis_type].mean(1, keepdims=True))
        ids.append(np.array([r.flywire_root_id]))
rates = np.concatenate(cols, 1).astype(np.float32)
root_ids = np.concatenate(ids)
out = args.out or f"rates_{name}.npz"
np.savez_compressed(out, rates=rates, root_ids=root_ids, dt=DT, label=name)

# --- report + orientation check picture ----------------------------------------------------------
md = [f"T{k}{d}" for k in "45" for d in "abcd"]
print(f"\n{name}: {rates.shape[0]} frames, {rates.shape[1]:,} FlyWire neurons -> {out}")
print("mean rate (Hz) of motion detectors, right eye | left eye:")
for t in md:
    k = fv_types == t
    print(f"  {t}: {rate[1][:, k].mean():6.2f} | {rate[0][:, k].mean():6.2f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = to_eyes(frames).cpu().numpy()[:, :, 0]                              # [2, T, 721]
    u = np.array(net.connectome.nodes.u[:])[fv_types == "R1"]
    vv = np.array(net.connectome.nodes.v[:])[fv_types == "R1"]
    ks = [0, len(frames) // 2, len(frames) - 1]
    fig, ax = plt.subplots(3, len(ks), figsize=(3.2 * len(ks), 8.6))
    for j, k in enumerate(ks):
        ax[0, j].imshow(frames[k], cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(f"camera t={k * DT:.2f}s"); ax[0, j].axis("off")
        for i, side in enumerate(["left", "right"]):
            a = ax[1 + i, j]
            a.scatter(vv, -(u + vv / 2), c=x[i, k], cmap="gray", vmin=0, vmax=1, s=9, marker="h")
            a.set_title(f"{side} eye (flyvis input)"); a.set_aspect("equal"); a.axis("off")
    fig.tight_layout(); fig.savefig(f"eye_check_{name}.png", dpi=110)
    print(f"saved eye_check_{name}.png")
except ImportError:
    pass
