"""Stage 1: what do the duck's two fly eyes see when something comes at it?

The duck gets two eye cameras on its head, one facing each side, shaped like a fly's
eyes: each covers about 150 degrees (a real fly eye's width), with the two eyes
overlapping slightly in front. Each eye's view is converted to equal angles per pixel,
so the fly eye model receives the world at its true size, not magnified.

The duck stands still in its STAND pose; a dark ball moves along a path. The duck's own
body is hidden from the eye cameras. Frames are saved at 50 frames/s as one image per
time step with the LEFT eye in the left half and the RIGHT eye in the right half: the
format fly_eye.py reads with --frames.

Scenarios (all on the duck's RIGHT side, so only the right fly eye sees them):
  loom    : ball flies straight at the head from 60 deg to the right, growing to ~90 deg (approaching)
  recede  : the same path played backwards (moving away)
  pass    : ball moves front-to-back past the right side at a fixed 0.6 m distance

Usage (inside ~/flyduck/microduck_rl, duck environment):
    uv run python ~/flyduck/bridge/duck_eyes.py
Outputs (in the bridge folder): duck_<scenario>.npz and duck_<scenario>.png
"""
import argparse
import os

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--scenario", choices=["loom", "recede", "pass", "all"], default="all")
ap.add_argument("--repo", default=os.path.expanduser("~/flyduck/microduck_rl"))
ap.add_argument("--out", default=HERE)
args = ap.parse_args()

SCENE = os.path.join(args.repo, "src/mjlab_microduck/robot/microduck/scene.xml")
DT, T, HOLD = 0.02, 1.0, 0.3       # frame step, motion time, still frames at the end
BALL_R = 0.05                       # 10 cm ball
EYE_FOV = 150.0                     # degrees covered by each eye (fly eye ~150 deg)
EYE_YAW = 70.0                      # each eye's centre points 70 deg to its side
EYE_PX = 400                        # output pixels per eye (square)
RENDER_PX = 1200                    # pinhole render size used for the resampling
LOOM_AZ = 60.0                      # loom direction, degrees right of straight ahead


# --- scene: duck + movable ball + two eye cameras on the head ----------------------------
def build():
    spec = mujoco.MjSpec.from_file(SCENE)
    spec.visual.global_.offwidth = RENDER_PX
    spec.visual.global_.offheight = RENDER_PX
    spec.visual.map.znear = 0.0005      # draw objects even 1 cm from the eye (no clipping)
    ball = spec.worldbody.add_body(name="ball", mocap=True, pos=[0, 0, -5])
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0],
                  rgba=[0.02, 0.02, 0.02, 1], contype=0, conaffinity=0, group=0)
    m0 = spec.compile()
    d0 = mujoco.MjData(m0)
    k = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    assert k >= 0, "STAND keyframe not found in the duck model"
    mujoco.mj_resetDataKeyframe(m0, d0, k)
    mujoco.mj_forward(m0, d0)

    # head frame and the duck's forward direction in the STAND pose
    hc = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    assert hc >= 0, "head_camera not found in the duck model"
    head_id = int(m0.cam_bodyid[hc])
    R_head, p_head = d0.xmat[head_id].reshape(3, 3), d0.xpos[head_id]
    trunk = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    fwd = d0.xmat[trunk].reshape(3, 3)[:, 0].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    eye_pos = d0.cam_xpos[hc] + 0.04 * fwd          # just in front of the face

    head_body = spec.camera("head_camera").parent
    for name, sign in [("eye_left", -1), ("eye_right", +1)]:
        a = np.radians(sign * EYE_YAW)
        look = np.cos(a) * fwd + np.sin(a) * right    # eye's viewing direction
        cam_right = np.cross(look, up)
        R_w = np.stack([cam_right, up, -look], 1)     # MuJoCo camera: x right, y up, -z view
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (R_head.T @ R_w).flatten())
        head_body.add_camera(name=name, pos=R_head.T @ (eye_pos - p_head), quat=q,
                             fovy=EYE_FOV)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, k)
    mujoco.mj_forward(m, d)
    return m, d, fwd, right, up


model, data, FWD, RIGHT, UP = build()
cams = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in ("eye_left", "eye_right")]
eye_pos = data.cam_xpos[cams[1]].copy()
for c, sign in zip(cams, (-1, 1)):
    look = -data.cam_xmat[c].reshape(3, 3)[:, 2]
    az = np.degrees(np.arctan2(look @ RIGHT, look @ FWD))
    assert abs(az - sign * EYE_YAW) < 0.5 and abs(look[2]) < 0.01, "eye camera points the wrong way"
print(f"eyes at {np.round(eye_pos, 3)} m; left eye centred {-EYE_YAW:.0f} deg, right eye "
      f"+{EYE_YAW:.0f} deg; each covers {EYE_FOV:.0f} deg; overlap in front "
      f"{2 * (EYE_FOV / 2 - EYE_YAW):.0f} deg")

# --- resampling: pinhole render -> equal angles per pixel (what a fly eye samples) ---------
# Output pixel (row, col) of an eye sits at angles (ax, ay) from that eye's centre; the ray
# in that direction is found in the pinhole image at x = f*tan, giving a lookup table.
half = np.radians(EYE_FOV / 2)
g = (np.arange(EYE_PX) + 0.5) / EYE_PX * 2 - 1                   # -1..1
ax, ay = np.meshgrid(g * half, -g * half)                         # ay > 0 = up
r = np.hypot(ax, ay)
phi = np.arctan2(ay, ax)
dx, dy, dz = np.sin(r) * np.cos(phi), np.sin(r) * np.sin(phi), np.cos(r)
valid = dz > 0.05                                                  # in front of the lens
f_px = (RENDER_PX / 2) / np.tan(half)
px = np.where(valid, RENDER_PX / 2 + f_px * dx / np.where(valid, dz, 1), 0)
py = np.where(valid, RENDER_PX / 2 - f_px * dy / np.where(valid, dz, 1), 0)
valid &= (px >= 0) & (px <= RENDER_PX - 1) & (py >= 0) & (py <= RENDER_PX - 1)
x0, y0 = np.floor(px).astype(int).clip(0, RENDER_PX - 2), np.floor(py).astype(int).clip(0, RENDER_PX - 2)
wx, wy = px - x0, py - y0

renderer = mujoco.Renderer(model, RENDER_PX, RENDER_PX)
opt = mujoco.MjvOption()
opt.geomgroup[2] = 0      # hide the duck's own body (visual meshes)
opt.geomgroup[3] = 0      # and its collision shapes


def eye_image(cam_id):
    renderer.update_scene(data, camera=cam_id, scene_option=opt)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0   # hidden body casts no shadow
    img = renderer.render().astype(np.float32) / 255.0 @ np.array([0.299, 0.587, 0.114], np.float32)
    out = (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x0 + 1] * wx * (1 - wy) +
           img[y0 + 1, x0] * (1 - wx) * wy + img[y0 + 1, x0 + 1] * wx * wy)
    return np.where(valid, out, out[valid].mean()).astype(np.float32)


def frame(ball_pos):
    data.mocap_pos[0] = ball_pos
    mujoco.mj_forward(model, data)
    return np.concatenate([eye_image(cams[0]), eye_image(cams[1])], 1)   # [left | right]


# --- ball paths --------------------------------------------------------------------------
n = int(T / DT)
t = (np.arange(n) + 0.5) * DT


def direction(az_deg):
    a = np.radians(az_deg)
    return np.cos(a) * FWD + np.sin(a) * RIGHT


def path(name):
    if name in ("loom", "recede"):
        dist = 2.0 - (2.0 - 0.07) * (t / T)          # constant-speed approach, 2.0 m -> 0.07 m (ball ~90 deg wide)
        if name == "recede":
            dist = dist[::-1]
        return eye_pos + dist[:, None] * direction(LOOM_AZ)
    az = np.linspace(25, 125, n)                      # front-to-back along the right side
    return eye_pos + 0.6 * np.stack([direction(a) for a in az])


empty = frame([0, 0, -5])                             # the scene without the ball
scenarios = ["loom", "recede", "pass"] if args.scenario == "all" else [args.scenario]
for name in scenarios:
    pos = path(name)
    pos = np.concatenate([pos, np.repeat(pos[-1:], int(HOLD / DT), 0)])
    frames = np.stack([frame(p) for p in pos])
    out = os.path.join(args.out, f"duck_{name}.npz")
    # 'background' = the same view with no ball: fly_eye.py uses it as the 0 Hz baseline
    np.savez_compressed(out, frames=frames, dt=DT, background=empty)

    ball_px = np.abs(frames - empty) > 0.1
    L, R = ball_px[:, :, :EYE_PX].sum((1, 2)), ball_px[:, :, EYE_PX:].sum((1, 2))
    print(f"{name:6s}: {len(frames)} frames -> {out}")
    print(f"        ball pixels, right eye: first {R[0]:6d}  max {R.max():6d}  last {R[-1]:6d}"
          f"   (left eye max {L.max()})")
    assert R.max() > 50, f"{name}: the ball never appears in the right eye"
    assert L.max() < 0.02 * R.max(), f"{name}: the ball leaks into the left eye"
    if name == "loom":
        assert R[-1] == R.max(), "loom: the ball must be largest at the end (clipping?)"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = np.linspace(0, n - 1, 5).astype(int)
        fig, ax_ = plt.subplots(1, len(ks), figsize=(3.6 * len(ks), 2.0))
        for a_, k in zip(ax_, ks):
            a_.imshow(frames[k], cmap="gray", vmin=0, vmax=1)
            a_.axvline(EYE_PX, color="r", lw=0.6)
            a_.set_title(f"{name} t={k * DT:.2f}s   (left eye | right eye)", fontsize=7)
            a_.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"duck_{name}.png"), dpi=100)
        plt.close(fig)
    except ImportError:
        pass
