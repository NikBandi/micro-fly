"""Stage 2: the closed loop. The fly brain sees through the duck's eyes and moves its legs.

Every 100 ms of simulated time:
  1. the duck walks for 100 ms with its current command (Pollen's walking controller)
  2. its two fly-eye cameras record 5 frames
  3. the fly eyes (flyvis, eye_worker.py) turn them into firing rates
  4. the FlyWire brain (brain_worker.py) runs for the same 100 ms
  5. the escape neurons decide the next command:
       escape signal = spikes of giant fiber (DNp01) + DNp02 + DNp11 on each side
       if a side reaches --threshold: back away and turn away from that side for 1 s
       otherwise: stand still
The brain therefore reacts with a 100 ms delay, similar to a real fly's escape latency.

Scenarios (the ball is aimed at where the duck's head is at the start):
  loom  : ball flies at the duck from 60 deg right at constant speed, from where it looks
          --start-deg wide until it is 7 cm from the eye (starts at 0.5 s, then holds 0.3 s).
          Speed is set the way fly experiments do it, by r/v (ball radius / speed, --rv, ms):
          small r/v = fast. Default 26 ms (= the first milestone's 2.0 m -> 0.07 m in 1 s).
  pass  : ball moves front-to-back past the right side at 0.6 m (no threat)
  none  : no ball (does the duck stay put?)
--observe: the brain is read out but the duck never moves (clean timing measurements).
--escape away (default): back up and sidestep away from the threatened side, no turning.
--escape turn: the first milestone's escape (back up + turn away).
--fly-through: the loom ball does not stop 7 cm short; it flies on through where the head
  was, so a dodge can be scored: closest gap between the ball's surface and the duck's body
  (negative = the ball went into the duck = HIT).
--eye-gain: multiplies the eye -> brain drive (1 = as calibrated).
The log has the ball's angular size (theta_deg) and time to collision (ttc_ms) per step.

Usage (duck environment, from ~/flyduck/microduck_rl):
    uv run python ~/flyduck/bridge/fly_loop.py --scenario loom
Outputs in the bridge folder: loop_<scenario>.mp4 (video), loop_<scenario>.csv (log).
"""
import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ipc

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", choices=["loom", "pass", "none", "pillar", "corridor"], default="loom")
ap.add_argument("--walk", type=float, default=None,
                help="forward walking command (m/s) when not escaping; default 0.3 for pillar/corridor, else 0")
ap.add_argument("--hold-heading", type=float, default=None,
                help="while walking, turn back toward the start heading: yaw cmd = -K * heading error "
                     "(rad/s per rad; a stand-in for the fly's haltere/vestibular course stabilisation). "
                     "Default 2 when walking, 0 = off")
ap.add_argument("--steer", default=None, metavar="GROUP[,GROUP]",
                help="steer away from obstacles with these brain readout groups, e.g. DNb05: "
                     "yaw += gain * ((R - L) spikes/s over --steer-window - --steer-bias); "
                     "+yaw = left, so more spikes on the right turns the duck left")
ap.add_argument("--steer-gain", type=float, default=0.03, help="rad/s of yaw per spike/s of R-L")
ap.add_argument("--steer-window", type=int, default=300, help="ms of spikes averaged for steering")
ap.add_argument("--steer-bias", type=float, default=0.0,
                help="R-L spikes/s the steering groups show while walking with nothing around (subtracted)")
ap.add_argument("--escape-gate", default=None, metavar="GROUP:N",
                help="escape only if GROUP also fired >= N spikes on the threat side in the same 100 ms "
                     "window, e.g. LPLC2:2 (looming detector; silent during plain walking)")
ap.add_argument("--loom-start", type=float, default=0.5,
                help="loom: time (s) the ball starts moving. While walking, the ball aims at where the "
                     "head will be at impact (from the walking speed over the last 0.5 s)")
ap.add_argument("--home-deg", type=float, default=15.0,
                help="walking loom: the ball keeps re-aiming at the (moving) duck until it looks this many "
                     "degrees wide, then flies straight - like a predator committing to its strike")
ap.add_argument("--no-escape", action="store_true",
                help="read the escape signal out but keep walking (steering probes)")
ap.add_argument("--pillar-ahead", type=float, default=0.6, help="pillar: distance ahead of the eyes (m)")
ap.add_argument("--pillar-offset", type=float, default=0.06,
                help="pillar: sideways offset of its centre from the walking line, towards --side (m)")
ap.add_argument("--corridor-half", type=float, default=0.18, help="corridor: half width (m)")
ap.add_argument("--threshold", type=int, default=3,
                help="escape spikes (one side, per 100 ms) that trigger an escape")
ap.add_argument("--duration", type=float, default=None,
                help="seconds of simulated time (default: loom long enough to finish, else 2.5)")
ap.add_argument("--rv", type=float, default=26.0, help="loom r/v in ms (ball radius / speed)")
ap.add_argument("--start-deg", type=float, default=2.9, help="loom: angular size at the start")
ap.add_argument("--observe", action="store_true", help="read the brain out but never move")
ap.add_argument("--eye-gain", type=float, default=1.0, help="eye -> brain drive multiplier")
ap.add_argument("--tag", default="", help="added to the output file names")
ap.add_argument("--escape", choices=["away", "turn"], default="away",
                help="away: back + sidestep away from the threat; turn: back + turn away")
ap.add_argument("--escape-cmd", type=float, nargs=3, metavar=("VX", "VY", "YAW"), default=None,
                help="custom escape for a threat on the RIGHT (mirrored for the left); overrides --escape")
ap.add_argument("--side", choices=["right", "left"], default="right",
                help="which side the ball comes from (loom and pass are mirrored for left)")
ap.add_argument("--fly-through", action="store_true",
                help="loom ball keeps going through the duck's start position (hit/miss test)")
ap.add_argument("--step-ms", type=int, default=100, choices=[20, 40, 100],
                help="how often eyes+brain run and the escape is decided; the escape signal is "
                     "always counted over the last 100 ms (sliding window)")
ap.add_argument("--repo", default=os.path.expanduser("~/flyduck/microduck_rl"))
ap.add_argument("--walking", default=None, help="walking policy (default: repo/policies/alpha_walking.onnx)")
ap.add_argument("--vision-python", default=os.path.expanduser("~/flyduck/.venv-vision/bin/python"))
ap.add_argument("--brain-python", default=os.path.expanduser("~/flyduck/Drosophila_brain_model/.venv/bin/python"))
ap.add_argument("--brain", default=os.path.expanduser("~/flyduck/Drosophila_brain_model"))
ap.add_argument("--untrained-eyes", action="store_true", help="testing only")
ap.add_argument("--motion-test", action="store_true",
                help="no brain: check the duck stands, then backs away and turns as an escape would")
ap.add_argument("--motion-sweep", action="store_true",
                help="no brain: try several escape commands from a fresh stand, report how far each moves")
ap.add_argument("--out", default=HERE)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
walking = args.walking or os.path.join(args.repo, "policies/alpha_walking.onnx")
for p, what in [(walking, "walking policy"), (args.vision_python, "vision environment python"),
                (args.brain_python, "brain environment python")]:
    if not os.path.exists(p):
        raise SystemExit(f"fly_loop: {what} not found at {p}")

sys.path.insert(0, os.path.join(args.repo, "scripts"))
import infer_policy as ip                     # Pollen's controller code (BAM motors, policy)

SCENE = os.path.join(args.repo, "src/mjlab_microduck/robot/microduck/scene.xml")
DT = 0.02                 # control step = eye frame step (50 Hz)
DECIMATION = 4            # physics steps per control step (0.005 s each)
CHUNK = args.step_ms // 20   # control steps per brain decision
WINDOW = 100 // args.step_ms  # decisions in the 100 ms escape-count window
EYE_FOV, EYE_YAW, EYE_PX, RENDER_PX = 150.0, 70.0, 400, 900
BALL_R = 0.05
ESCAPE_TIME = 1.0         # seconds an escape command lasts
ESCAPE_GROUPS = ["DNp01", "DNp02", "DNp11"]

# loom geometry (distances from the right eye camera)
LOOM_T0, LOOM_END, LOOM_HOLD = args.loom_start, 0.07, 0.3
loom_v = BALL_R / (args.rv / 1000.0)                        # m/s
loom_d0 = BALL_R / math.sin(math.radians(args.start_deg) / 2)
loom_t_end = LOOM_T0 + (loom_d0 - LOOM_END) / loom_v         # ball stops 7 cm from the eye
loom_t_coll = LOOM_T0 + loom_d0 / loom_v                     # when it would reach the eye
THROUGH = 0.3                                                # fly-through: goes 0.3 m past
loom_t_hide = (LOOM_T0 + (loom_d0 + THROUGH) / loom_v) if args.fly_through else loom_t_end + LOOM_HOLD
if args.duration is None:
    args.duration = (max(2.5, loom_t_hide + 0.5) if args.scenario == "loom" else
                     6.0 if args.scenario in ("pillar", "corridor") else 2.5)
if args.walk is None:
    args.walk = 0.3 if args.scenario in ("pillar", "corridor") else 0.0
WALKING = args.scenario in ("pillar", "corridor") or args.walk != 0.0
if args.hold_heading is None:
    args.hold_heading = 2.0 if WALKING else 0.0
SIDE_SIGN = 1.0 if args.side == "right" else -1.0     # azimuth + = right of the duck
PILLAR_R, PILLAR_H = 0.03, 0.5
STRIPE = 0.04                                          # corridor wall stripe width (m)
CORRIDOR_LEN, WALL_H = 3.0, 0.3
suffix = f"_{args.tag}" if args.tag else ""


# --- duck model: Pollen's BAM motor set-up + our ball and two fly-eye cameras -------------
def build(eye_cams=None, origin=None, fwd_=None):
    bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, 0.0)
    kt, R = bam_model.kt.value, bam_model.R.value
    force_limit = bam_model.actuator.vin * kt / R
    spec = mujoco.MjSpec.from_file(SCENE)
    names = []
    for act in spec.actuators:                       # same edits as ip.load_mujoco_with_bam
        tgt = act.target
        tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
        if tgt_name.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for joint in spec.joints:
            if joint.name == tgt_name:
                joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0
                joint.solref_friction = ip.BAM_STIFF_SOLREF_FRICTION
                joint.solimp_friction = ip.BAM_STIFF_SOLIMP_FRICTION
                break
    spec.visual.global_.offwidth = RENDER_PX
    spec.visual.global_.offheight = RENDER_PX
    spec.visual.map.znear = 0.0005
    ball = spec.worldbody.add_body(name="ball", mocap=True, pos=[0, 0, -5])
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0],
                  rgba=[0.02, 0.02, 0.02, 1], contype=0, conaffinity=0, group=0)
    if origin is not None:
        add_obstacles(spec, origin, fwd_)
    if eye_cams:
        head_body = spec.camera("head_camera").parent
        for name, pos, quat in eye_cams:
            head_body.add_camera(name=name, pos=pos, quat=quat, fovy=EYE_FOV)
    model = spec.compile()
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    from bam.mujoco import MujocoController
    bam_ctrl = MujocoController(bam_model, names, model, data, vin_drop_gain=0.1,
                                vin_min=ip.BAM_VIN_MIN)
    return model, data, bam_ctrl


def add_obstacles(spec, origin, f):
    """Static obstacles fixed in the world, laid out from the eyes' start position (origin) and the
    duck's start heading (f). No collision (contype 0): touching is measured with mj_geomDistance."""
    f = np.array([f[0], f[1], 0.0]) / np.linalg.norm(f[:2])
    rt = np.array([f[1], -f[0], 0.0])                      # right of the duck
    yaw = math.atan2(f[1], f[0])
    quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    if args.scenario == "pillar":
        c = origin + args.pillar_ahead * f + SIDE_SIGN * args.pillar_offset * rt
        spec.worldbody.add_geom(name="obstacle_pillar", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                size=[PILLAR_R, PILLAR_H / 2, 0], pos=[c[0], c[1], PILLAR_H / 2],
                                rgba=[0.05, 0.05, 0.05, 1], contype=0, conaffinity=0, group=0)
    elif args.scenario == "corridor":
        n = int(CORRIDOR_LEN / STRIPE)
        for side, sgn in (("right", 1), ("left", -1)):
            y = sgn * (args.corridor_half + 0.005)
            for k in range(n):                             # vertical black/white stripes
                c = origin + (0.1 + (k + 0.5) * STRIPE) * f + y * rt
                shade = 0.1 if k % 2 == 0 else 0.85
                spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                        size=[STRIPE / 2, 0.005, WALL_H / 2],
                                        pos=[c[0], c[1], WALL_H / 2], quat=quat,
                                        rgba=[shade, shade, shade, 1], contype=0, conaffinity=0,
                                        group=0)
            c = origin + (0.1 + CORRIDOR_LEN / 2) * f + y * rt   # invisible proxy for the gap test
            spec.worldbody.add_geom(name=f"obstacle_wall_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                    size=[CORRIDOR_LEN / 2, 0.005, WALL_H / 2],
                                    pos=[c[0], c[1], WALL_H / 2], quat=quat, rgba=[0, 0, 0, 0],
                                    contype=0, conaffinity=0, group=5)


def start_pose(model, data, bam_ctrl, policy=None):
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = model.jnt_qposadr[fj]
    data.qpos[adr:adr + 3] = [0, 0, 0.125]
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    if policy is not None:
        for i, q in enumerate(policy.joint_qpos_indices):
            data.qpos[q] = policy.default_pose[i]
        bam_ctrl.reset(data.qpos)
        policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    return adr


def make_policy(model, data, bam_ctrl):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        # same settings as Pollen's infer_policy.py defaults (projected gravity is what the
        # policy was trained on; the raw accelerometer option is off by default there)
        pol = ip.PolicyInference(model, data, bam_ctrl=bam_ctrl, walking_onnx_path=walking,
                                 new_cmd_obs=True, use_projected_gravity=True)
        pol.vel_max_x, pol.vel_min_x = 0.3, -0.3
        pol.vel_max_y, pol.vel_min_y = 0.2, -0.2
        pol.vel_max_ang = 1.5
    return pol


def set_cmd(pol, vx, yaw, vy=0.0):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        pol.set_vel_cmd(vx, vy, yaw)


def escape_cmd(pol, threat_side):
    """(vx, vy, yaw) that moves away from a threat on threat_side. +vy = left, +yaw = left."""
    away = 1.0 if threat_side == "right" else -1.0
    if args.escape_cmd:
        vx, vy, wz = args.escape_cmd
        return vx, away * vy, away * wz
    if args.escape == "turn":
        return pol.vel_min_x, 0.0, away * pol.vel_max_ang
    return pol.vel_min_x, (pol.vel_max_y if away > 0 else pol.vel_min_y), 0.0


# pass 1: find where the head is in the standing pose, to place the eye cameras
model, data, bam_ctrl = build()
policy = make_policy(model, data, bam_ctrl)
start_pose(model, data, bam_ctrl, policy)
hc = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
head_id = int(model.cam_bodyid[hc])
R_head, p_head = data.xmat[head_id].reshape(3, 3).copy(), data.xpos[head_id].copy()
trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
fwd = data.xmat[trunk].reshape(3, 3)[:, 0].copy()
fwd[2] = 0
fwd /= np.linalg.norm(fwd)
up = np.array([0.0, 0.0, 1.0])
eye_world = data.cam_xpos[hc] + 0.04 * fwd
cams = []
for name, sign in [("eye_left", -1), ("eye_right", +1)]:
    a = np.radians(sign * EYE_YAW)
    look = np.cos(a) * fwd + np.sin(a) * np.cross(fwd, up)
    R_w = np.stack([np.cross(look, up), up, -look], 1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (R_head.T @ R_w).flatten())
    cams.append((name, R_head.T @ (eye_world - p_head), q))

# pass 2: the real model with eyes (and obstacles laid out from the start pose)
model, data, bam_ctrl = build(cams, eye_world, fwd)
policy = make_policy(model, data, bam_ctrl)
qadr = start_pose(model, data, bam_ctrl, policy)
eye_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in ("eye_left", "eye_right")]
ball_mocap = 0


def control_step():
    action = policy.infer()
    policy.apply_action(action)
    for _ in range(DECIMATION):
        bam_ctrl.update()
        mujoco.mj_step(model, data)


# --- eye rendering (same equal-angle resampling as duck_eyes.py) ---------------------------
half = np.radians(EYE_FOV / 2)
g = (np.arange(EYE_PX) + 0.5) / EYE_PX * 2 - 1
ax_, ay_ = np.meshgrid(g * half, -g * half)
r_, phi_ = np.hypot(ax_, ay_), np.arctan2(ay_, ax_)
dx, dy, dz = np.sin(r_) * np.cos(phi_), np.sin(r_) * np.sin(phi_), np.cos(r_)
valid = dz > 0.05
f_px = (RENDER_PX / 2) / np.tan(half)
px = np.where(valid, RENDER_PX / 2 + f_px * dx / np.where(valid, dz, 1), 0)
py = np.where(valid, RENDER_PX / 2 - f_px * dy / np.where(valid, dz, 1), 0)
valid &= (px >= 0) & (px <= RENDER_PX - 1) & (py >= 0) & (py <= RENDER_PX - 1)
x0 = np.floor(px).astype(int).clip(0, RENDER_PX - 2)
y0 = np.floor(py).astype(int).clip(0, RENDER_PX - 2)
wx, wy = px - x0, py - y0
eye_r = mujoco.Renderer(model, RENDER_PX, RENDER_PX)
opt_eye = mujoco.MjvOption()
opt_eye.geomgroup[2] = 0
opt_eye.geomgroup[3] = 0
GREY = np.array([0.299, 0.587, 0.114], np.float32)


def eyes_frame():
    out = []
    for c in eye_ids:
        eye_r.update_scene(data, camera=c, scene_option=opt_eye)
        eye_r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        img = eye_r.render().astype(np.float32) / 255.0 @ GREY
        e = (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x0 + 1] * wx * (1 - wy) +
             img[y0 + 1, x0] * (1 - wx) * wy + img[y0 + 1, x0 + 1] * wx * wy)
        out.append(np.where(valid, e, e[valid].mean()))
    return np.concatenate(out, 1).astype(np.float32)


# external view for the video
VW, VH = 640, 360
ext_r = mujoco.Renderer(model, VH, VW)
ext_cam = mujoco.MjvCamera()
ext_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
ext_cam.trackbodyid = trunk
ext_cam.distance, ext_cam.elevation, ext_cam.azimuth = 1.1, -25.0, -30.0


def ext_frame():
    ext_r.update_scene(data, camera=ext_cam)
    ext_r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0       # avoids shadow-map artefacts
    return ext_r.render()


# --- start the eye and brain workers --------------------------------------------------------
def start(cmd):
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)


if args.motion_sweep:
    a60 = np.radians(60)
    CMDS = [("back", -0.3, 0.0, 0.0), ("sidestep left", 0.0, 0.2, 0.0),
            ("sidestep right", 0.0, -0.2, 0.0), ("back + sidestep left", -0.3, 0.2, 0.0),
            ("back + half sidestep left", -0.3, 0.1, 0.0), ("back + turn left", -0.3, 0.0, 1.5),
            ("back + left + turn left", -0.3, 0.2, 1.5), ("forward", 0.3, 0.0, 0.0),
            ("forward + sidestep left", 0.3, 0.2, 0.0)]
    print("each command from a fresh 1 s stand; head displacement after 0.5 / 1.0 / 1.5 s.")
    print("'away' = head distance gained from a ball coming from 60 deg right; 'off path' = sideways off "
          "its line (need ~10 cm off path to miss)")
    for name, vx, vy, wz in CMDS:
        m_, d_, b_ = build(cams)                           # (no obstacles)
        pol_ = make_policy(m_, d_, b_)
        qa_ = start_pose(m_, d_, b_, pol_)
        cam_ = mujoco.mj_name2id(m_, mujoco.mjtObj.mjOBJ_CAMERA, "eye_right")
        tr_ = mujoco.mj_name2id(m_, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

        def step_():
            pol_.apply_action(pol_.infer())
            for _ in range(DECIMATION):
                b_.update()
                mujoco.mj_step(m_, d_)
        set_cmd(pol_, 0.0, 0.0)
        for _ in range(50):
            step_()
        mujoco.mj_forward(m_, d_)
        f_ = d_.xmat[tr_].reshape(3, 3)[:, 0].copy()
        f_[2] = 0
        f_ /= np.linalg.norm(f_)
        l_ = np.cross([0.0, 0.0, 1.0], f_)
        thr = np.cos(a60) * f_ - np.sin(a60) * l_              # direction to the ball (60 deg right)
        perp = np.cross([0.0, 0.0, 1.0], thr)
        h0 = d_.cam_xpos[cam_].copy()
        set_cmd(pol_, vx, wz, vy)
        out = []
        for k in range(75):
            step_()
            if (k + 1) % 25 == 0:
                dh = d_.cam_xpos[cam_] - h0
                out.append(f"{-(dh @ thr) * 100:+5.1f} away {abs(dh @ perp) * 100:4.1f} off")
        up_ = d_.xmat[tr_].reshape(3, 3)[2, 2]
        fell_ = d_.qpos[qa_ + 2] < 0.07 or up_ < 0.8
        print(f"  {name:<26} (vx {vx:+.1f} vy {vy:+.1f} yaw {wz:+.1f}): " + " | ".join(out)
              + ("   FELL" if fell_ else ""), flush=True)
    raise SystemExit

# settle the duck standing still for 1 s
set_cmd(policy, 0.0, 0.0)
for _ in range(50):
    control_step()

if args.motion_test:
    def pose():
        q = data.qpos[qadr + 3:qadr + 7]
        yaw = math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        return data.qpos[qadr:qadr + 3].copy(), yaw
    p0, y0_ = pose()
    print(f"standing: trunk height {p0[2] * 100:.1f} cm")
    for _ in range(50):                                   # 1 s more of standing
        control_step()
    p1, y1_ = pose()
    print(f"after 1 s standing still: drifted {np.linalg.norm(p1[:2] - p0[:2]) * 100:.1f} cm, "
          f"turned {y1_ - y0_:+.1f} deg, height {p1[2] * 100:.1f} cm")
    f0 = data.xmat[trunk].reshape(3, 3)[:, 0].copy()
    l0 = np.cross([0.0, 0.0, 1.0], f0)                    # left
    evx, evy, eyaw = escape_cmd(policy, "right")          # escape from a threat on the right
    set_cmd(policy, evx, eyaw, evy)
    up0 = data.xmat[trunk].reshape(3, 3)[2, 2]
    print(f"tilt while standing: {math.degrees(math.acos(min(1, up0))):.0f} deg")
    for _ in range(int(ESCAPE_TIME / DT)):
        control_step()
    p2, y2_ = pose()
    d = p2[:2] - p1[:2]
    print(f"escape '{args.escape}' from a threat on the right for {ESCAPE_TIME} s: moved "
          f"{d @ f0[:2] * 100:+.1f} cm forward (negative = backwards), {d @ l0[:2] * 100:+.1f} cm "
          f"sideways (positive = left), turned {(y2_ - y1_ + 180) % 360 - 180:+.1f} deg "
          f"(positive = left), height {p2[2] * 100:.1f} cm")
    # the threat came from 60 deg right: how far did the head get away from that line?
    a = np.radians(60)
    threat = np.cos(a) * f0[:2] - np.sin(a) * l0[:2]
    print(f"  distance gained away from a threat at 60 deg right: {-(d @ threat) * 100:+.1f} cm; "
          f"sideways off its path: {abs(d @ np.array([-threat[1], threat[0]])) * 100:.1f} cm")
    t_mark = data.time
    set_cmd(policy, 0.0, 0.0)
    for sec in (0.5, 1.0, 1.5, 2.0, 3.0):
        while data.time < t_mark + sec:
            control_step()
        p3, _ = pose()
        up_z = data.xmat[trunk].reshape(3, 3)[2, 2]          # 1 = upright
        print(f"  {sec:.1f} s after the escape: height {p3[2] * 100:.1f} cm, "
              f"tilt {math.degrees(math.acos(min(1, up_z))):.0f} deg")
    print("RESULT:", "still standing" if p3[2] > 0.09 and up_z > 0.9 else "FELL OVER / unstable")
    raise SystemExit

print("starting fly eyes and fly brain (about a minute)...")
eye_cmd = [args.vision_python, os.path.join(HERE, "eye_worker.py")]
if args.untrained_eyes:
    eye_cmd.append("--untrained")
eye_cmd += ["--gain", str(args.eye_gain)]
if args.scenario == "loom":
    print(f"loom: r/v {args.rv:g} ms = {loom_v:.2f} m/s, from {loom_d0:.2f} m ({args.start_deg:g} deg) "
          f"to {LOOM_END} m at t={loom_t_end:.2f} s; collision would be at t={loom_t_coll:.2f} s; "
          f"{args.duration:.1f} s simulated" + ("; OBSERVE only" if args.observe else ""))
eye_p = start(eye_cmd)
brain_p = start([args.brain_python, os.path.join(HERE, "brain_worker.py"), "--brain", args.brain])
ipc.send(eye_p.stdin, cmd="init", frames=eyes_frame()[None])
init_reply = ipc.recv(eye_p.stdout)
root_ids = init_reply["root_ids"]
drive_names = [str(x) for x in init_reply["drive_names"]]
ipc.send(brain_p.stdin, cmd="init", root_ids=root_ids)
b = ipc.recv(brain_p.stdout)
groups = [f"{gr}:{sd}" for gr, sd in zip(b["groups"], b["sides"])]
esc_idx = {s: [groups.index(f"{gr}:{s}") for gr in ESCAPE_GROUPS] for s in ("right", "left")}
gate_group, gate_n = (args.escape_gate.split(":")[0], int(args.escape_gate.split(":")[1])) \
    if args.escape_gate else (None, 0)
if gate_group and f"{gate_group}:right" not in groups:
    raise SystemExit(f"--escape-gate: {gate_group} is not a brain readout group")

# --- ball path, fixed in the world from the duck's start pose ------------------------------
fwd0 = data.xmat[trunk].reshape(3, 3)[:, 0].copy()
fwd0[2] = 0
fwd0 /= np.linalg.norm(fwd0)
right0 = np.cross(fwd0, up)
head0 = data.cam_xpos[eye_ids[1]].copy()
HIDDEN = np.array([0.0, 0.0, -5.0])
from collections import deque


_head_hist = deque()
_aim = {"origin": None}


def loom_origin(t):
    """where the ball is aimed: the start head position when standing; when walking, the head's
    predicted position at impact (current head + walking velocity x time left), frozen at launch"""
    if not WALKING:
        return head0
    if _aim["origin"] is not None:
        return _aim["origin"]
    h = data.cam_xpos[eye_ids[1]].copy()                # (frozen once the ball commits)
    _head_hist.append((t, h))
    while _head_hist and _head_hist[0][0] < t - 0.5:
        _head_hist.popleft()
    t_old, h_old = _head_hist[0]
    v = (h - h_old) / (t - t_old) if t - t_old > 0.25 else np.zeros(3)
    v[2] = 0.0
    o = h + v * max(loom_t_coll - t, 0.0)
    if t >= LOOM_T0:
        d = loom_d0 - loom_v * (t - LOOM_T0)
        dist = float(np.linalg.norm(o + d * _dir(SIDE_SIGN * 60) - h))
        size = math.degrees(2 * math.asin(min(1.0, BALL_R / max(dist, BALL_R))))
        if size >= args.home_deg:
            _aim["origin"] = o
            _aim["t"] = t
    return o


def ball_at(t):
    if args.scenario in ("none", "pillar", "corridor"):
        return HIDDEN
    if args.scenario == "loom":
        if t > loom_t_hide:
            return HIDDEN
        d = loom_d0 - loom_v * max(t - LOOM_T0, 0.0)
        if not args.fly_through:
            d = max(d, LOOM_END)
        return loom_origin(t) + d * _dir(SIDE_SIGN * 60)
    s = min(max((t - 0.5) / 1.0, 0.0), 1.0)      # 0 until 0.5 s, then 0 -> 1 over 1 s
    if t > 1.8:
        return HIDDEN
    return head0 + 0.6 * _dir(SIDE_SIGN * (25 + 100 * s))


def _dir(az):
    a = np.radians(az)
    return np.cos(a) * fwd0 + np.sin(a) * right0


def yaw_of():
    q = data.qpos[qadr + 3:qadr + 7]
    return math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


# --- video writer -----------------------------------------------------------------------
video_path = os.path.join(args.out, f"loop_{args.scenario}{suffix}.mp4")
FW, FH = VW, VH + 320
ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{FW}x{FH}", "-r", str(int(1 / DT)), "-i", "-",
                       "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23", video_path],
                      stdin=subprocess.PIPE)
try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None


def write_video(ext, eyes, status, escaping):
    e = (np.clip(eyes, 0, 1) * 255).astype(np.uint8)
    idx_r = (np.arange(320) * EYE_PX / 320).astype(int)
    idx_c = (np.arange(640) * 2 * EYE_PX / 640).astype(int)
    e = e[idx_r][:, idx_c]
    frame = np.zeros((FH, FW, 3), np.uint8)
    frame[:VH] = ext
    frame[VH:] = np.repeat(e[:, :, None], 3, 2)
    frame[VH:, FW // 2 - 1:FW // 2 + 1] = [255, 0, 0]
    if escaping:
        frame[:6] = frame[-6:] = [255, 40, 40]
        frame[:, :6] = frame[:, -6:] = [255, 40, 40]
    if Image is not None:
        im = Image.fromarray(frame)
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, FW, 20], fill=(0, 0, 0))
        dr.text((8, 4), status, fill=(255, 255, 255))
        dr.text((8, VH + 4), "left fly eye", fill=(255, 255, 0))
        dr.text((FW // 2 + 8, VH + 4), "right fly eye", fill=(255, 255, 0))
        frame = np.asarray(im)
    ff.stdin.write(frame.tobytes())


# --- the loop -------------------------------------------------------------------------------
log = []
from collections import deque
hist_r, hist_l = deque(maxlen=WINDOW), deque(maxlen=WINDOW)
gate_r, gate_l = deque(maxlen=WINDOW), deque(maxlen=WINDOW)
escapes = []                                    # every escape: (t, side)
cmd_vx, cmd_yaw, escape_until = 0.0, 0.0, -1.0
first_escape = None
start_xy = data.qpos[qadr:qadr + 2].copy()
start_yaw = yaw_of()
n_chunks = int(round(args.duration / (DT * CHUNK)))
t = 0.0
t_wall = time.time()
min_dist = 9.9
# hit test: closest gap between the ball's surface and any part of the duck (body meshes)
ball_geom = int(np.flatnonzero(model.geom_bodyid ==
                               mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball"))[0])
duck_geoms = [g for g in range(model.ngeom)
              if model.body_rootid[model.geom_bodyid[g]] == model.body_rootid[trunk]]
_fromto = np.zeros(6)


def ball_gap():
    return min(mujoco.mj_geomDistance(model, data, ball_geom, g, 1.0, _fromto) for g in duck_geoms)


obstacle_geoms = [g for g in range(model.ngeom)
                  if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("obstacle_")]


# duck surface points for the corridor test: mesh vertices (mesh frame) per duck mesh geom
_duck_pts = []
for g in duck_geoms:
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        m = model.geom_dataid[g]
        a0, n = model.mesh_vertadr[m], model.mesh_vertnum[m]
        _duck_pts.append((g, model.mesh_vert[a0:a0 + n].copy(), 0.0))
    else:                                    # primitive: centre + bounding radius (conservative)
        _duck_pts.append((g, np.zeros((1, 3)), float(model.geom_rbound[g])))
_side_dir = np.cross(fwd, up)                # right of the start heading


def obstacle_gap():
    """closest gap (m) between any obstacle and the duck's body; <= 0 means it walked into it"""
    if args.scenario == "corridor":
        # walls are planes at +-corridor_half from the start line (from 0.1 m ahead of the eyes):
        # measured from the duck's own vertices (mj_geomDistance box-vs-mesh was unreliable)
        best = 9.9
        for g, pts, rad in _duck_pts:
            w = data.geom_xpos[g] + pts @ data.geom_xmat[g].reshape(3, 3).T
            rel = w - eye_world
            ahead = rel @ fwd >= 0.1
            if not ahead.any():
                continue
            lat = rel[ahead] @ _side_dir
            best = min(best, args.corridor_half - (np.abs(lat).max() + rad))
        return best
    return min(mujoco.mj_geomDistance(model, data, o, g, 1.0, _fromto)
               for o in obstacle_geoms for g in duck_geoms)


def obstacle_view():
    """pillar: (distance from the eyes to its surface, bearing in deg, + = right of the heading);
    corridor: (sideways offset of the eyes from the centre line in m, + = right; heading error deg)"""
    h = data.cam_xpos[eye_ids[1]] * 0.5 + data.cam_xpos[eye_ids[0]] * 0.5
    f = data.xmat[trunk].reshape(3, 3)[:, 0]
    head_deg = math.degrees(math.atan2(f[1], f[0]))
    if args.scenario == "pillar":
        c = model.geom_pos[obstacle_geoms[0]]
        d = c[:2] - h[:2]
        bearing = (head_deg - math.degrees(math.atan2(d[1], d[0])) + 180) % 360 - 180
        return float(np.linalg.norm(d) - PILLAR_R), float(bearing)
    off = float((h - eye_world) @ np.cross(fwd, up))
    err = (head_deg - math.degrees(math.atan2(fwd[1], fwd[0])) + 180) % 360 - 180
    return off, -float(err)


min_gap, gap_now = 9.9, float("nan")
min_obs_gap = 9.9
totals = {}                                     # spikes per readout group over the whole run
WALK_CMD = (0.0, 0.0, 0.0) if args.observe else (args.walk, 0.0, 0.0)
steer_groups = args.steer.split(",") if args.steer else []
for gr in steer_groups:
    if f"{gr}:right" not in groups:
        raise SystemExit(f"--steer: {gr} is not a brain readout group ({sorted(set(g.split(':')[0] for g in groups))})")
steer_idx = {s: [groups.index(f"{gr}:{s}") for gr in steer_groups] for s in ("right", "left")}
steer_hist = deque(maxlen=max(1, args.steer_window // args.step_ms))
steer_sig, steer_yaw = 0.0, 0.0
cmd_vx, cmd_vy, cmd_yaw = WALK_CMD
cmd_vy = 0.0
status = "starting"
for c in range(n_chunks):
    frames = []
    for k in range(CHUNK):
        data.mocap_pos[ball_mocap] = ball_at(t)
        control_step()
        t += DT
        mujoco.mj_forward(model, data)
        frames.append(eyes_frame())
        ball = data.mocap_pos[ball_mocap]
        if ball[2] > -1:
            min_dist = min(min_dist, float(np.linalg.norm(ball - data.cam_xpos[eye_ids[1]])))
            gap_now = float(ball_gap())
            min_gap = min(min_gap, gap_now)
        else:
            gap_now = float("nan")
        if obstacle_geoms:
            gap_now = float(obstacle_gap())
            min_obs_gap = min(min_obs_gap, gap_now)
            if gap_now >= 9:                   # corridor: duck not between the walls yet
                gap_now = float("nan")
        write_video(ext_frame(), frames[-1], status, t < escape_until)
    frames = np.stack(frames)
    ipc.send(eye_p.stdin, cmd="step", frames=frames)
    eye_reply = ipc.recv(eye_p.stdout)
    rates = eye_reply["rates"]
    drive = eye_reply["drive"].mean(0)                    # mean summed rate over the 100 ms
    ipc.send(brain_p.stdin, cmd="step", rates=rates, dt=DT)
    counts = ipc.recv(brain_p.stdout)["counts"]
    hist_r.append(int(counts[esc_idx["right"]].sum()))
    hist_l.append(int(counts[esc_idx["left"]].sum()))
    e_r, e_l = sum(hist_r), sum(hist_l)            # escape spikes in the last 100 ms
    if gate_group:
        gate_r.append(int(counts[groups.index(f"{gate_group}:right")]))
        gate_l.append(int(counts[groups.index(f"{gate_group}:left")]))
        ok_r, ok_l = sum(gate_r) >= gate_n, sum(gate_l) >= gate_n
    else:
        ok_r = ok_l = True
    trig_r = e_r >= args.threshold and ok_r
    trig_l = e_l >= args.threshold and ok_l
    ball = data.mocap_pos[ball_mocap]
    dist = float(np.linalg.norm(ball - data.cam_xpos[eye_ids[1]])) if ball[2] > -1 else float("nan")
    theta = math.degrees(2 * math.asin(min(1.0, BALL_R / dist))) if dist == dist else 0.0
    ttc = (loom_t_coll - t) * 1000 if args.scenario == "loom" else float("nan")

    # decide the next 100 ms
    for gname, v in zip(groups, counts):
        totals[gname] = totals.get(gname, 0) + int(v)
    passive = args.observe or args.no_escape
    if steer_groups:
        steer_hist.append(int(counts[steer_idx["right"]].sum()) - int(counts[steer_idx["left"]].sum()))
        steer_sig = sum(steer_hist) / (len(steer_hist) * args.step_ms / 1000) - args.steer_bias
        steer_yaw = args.steer_gain * steer_sig           # more on the right -> turn left (+yaw)
    if (trig_r or trig_l) and t >= escape_until:
        escape_until = t + ESCAPE_TIME
        if trig_r and trig_l:
            if gate_group:      # the looming detector is cleanly lateralised; escape counts are not
                threat = "right" if sum(gate_r) >= sum(gate_l) else "left"
            else:
                threat = "right" if e_r >= e_l else "left"
        else:
            threat = "right" if trig_r else "left"
        escapes.append((round(t, 2), threat))
        cmd_vx, cmd_vy, cmd_yaw = WALK_CMD if passive else escape_cmd(policy, threat)
        if first_escape is None:
            first_escape = (t, dist, threat, theta, ttc)
    elif t >= escape_until:
        cmd_vx, cmd_vy, cmd_yaw = WALK_CMD
    if (passive or t >= escape_until) and WALK_CMD[0] and args.hold_heading:
        err = math.radians((yaw_of() - start_yaw + 180) % 360 - 180)      # + = turned left
        cmd_yaw = float(np.clip(-args.hold_heading * err + steer_yaw, -1.5, 1.5))
    set_cmd(policy, cmd_vx, cmd_yaw, cmd_vy)

    if t < escape_until:
        act = ("would escape (observe)" if args.observe else
               "would escape (keeps walking)" if args.no_escape else
               f"ESCAPE (vx {cmd_vx:+.1f} vy {cmd_vy:+.1f} yaw {cmd_yaw:+.1f})")
    else:
        act = (f"walk {cmd_vx:.1f} m/s yaw {cmd_yaw:+.2f}" if cmd_vx else "stand")
    if obstacle_geoms:
        o1, o2 = obstacle_view()
        where = (f"pillar {o1 * 100:3.0f} cm at {o2:+4.0f} deg" if args.scenario == "pillar"
                 else f"off centre {o1 * 100:+4.1f} cm, heading {o2:+4.0f} deg")
        steer = "  DNa01 R{} L{} DNa02 R{} L{}".format(
            *[int(counts[groups.index(f"{gr}:{sd}")]) if f"{gr}:{sd}" in groups else 0
              for gr in ("DNa01", "DNa02") for sd in ("right", "left")])
        if steer_groups:
            steer = f"  steer {'+'.join(steer_groups)} R-L {steer_sig:+5.0f}/s"
        status = f"t={t:4.2f}s  {where}  esc R {e_r} L {e_l}{steer}  ->  {act}"
    else:
        o1 = o2 = float("nan")
        status = (f"t={t:4.2f}s  ball {dist:4.2f} m {theta:3.0f} deg  escape spikes R {e_r} L {e_l}  ->  {act}"
                  if dist == dist else f"t={t:4.2f}s  escape spikes R {e_r} L {e_l}  ->  {act}")
    print(status, f"   [{time.time() - t_wall:.0f} s]", flush=True)
    row = {"t": round(t, 3), "ball_dist": dist, "theta_deg": round(theta, 2),
           "ttc_ms": round(ttc, 1), "escape_R": e_r, "escape_L": e_l,
           "cmd_vx": cmd_vx, "cmd_vy": cmd_vy, "cmd_yaw": cmd_yaw,
           "gap_cm": round(gap_now * 100, 2) if gap_now == gap_now else float("nan"),
           "x": float(data.qpos[qadr]), "y": float(data.qpos[qadr + 1]), "yaw_deg": yaw_of(),
           "trunk_z": float(data.qpos[qadr + 2]), "obs_a": o1, "obs_b": o2,
           "steer_sig": round(steer_sig, 1), "steer_yaw": round(steer_yaw, 3)}
    row.update({gname: int(v) for gname, v in zip(groups, counts)})
    row.update({dn: round(float(v), 1) for dn, v in zip(drive_names, drive)})
    log.append(row)

ff.stdin.close()
ff.wait()
for p in (eye_p, brain_p):
    p.stdin.close()
    p.wait()

import csv
csv_path = os.path.join(args.out, f"loop_{args.scenario}{suffix}.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(log[0].keys()))
    w.writeheader()
    w.writerows(log)

moved = float(np.linalg.norm(data.qpos[qadr:qadr + 2] - start_xy))
turned = (yaw_of() - start_yaw + 180) % 360 - 180
fell = data.qpos[qadr + 2] < 0.07
print("\n=== result:", args.scenario, "===")
if first_escape:
    print(f"escape {'signal (observe only)' if args.observe else 'triggered'} at t={first_escape[0]:.2f} s "
          f"on the {first_escape[2]} side"
          + (f", ball {first_escape[1]:.2f} m away, {first_escape[3]:.0f} deg wide"
             if first_escape[1] == first_escape[1] else ""))
    if args.scenario == "loom":
        print(f"  time to collision at that moment: {first_escape[4]:+.0f} ms "
              f"(negative = after the ball would have hit)")
else:
    print("no escape triggered")
print(f"duck moved {moved * 100:.0f} cm and turned {turned:+.0f} deg; "
      f"{'it FELL OVER' if fell else 'still standing'}")
if args.scenario == "loom":
    print(f"closest the ball came to the duck's eye: {min_dist * 100:.0f} cm")
    if min_gap < 9:
        print(f"closest gap between ball surface and duck body: {min_gap * 100:+.1f} cm  ->  "
              + ("HIT (the ball went into the duck)" if min_gap <= 0 else "MISSED - the duck dodged it"))
if obstacle_geoms:
    print(f"closest gap between the duck and the {args.scenario}: {min_obs_gap * 100:+.1f} cm  ->  "
          + ("WALKED INTO IT" if min_obs_gap <= 0 else "clear"))
    o1, o2 = obstacle_view()
    if args.scenario == "corridor":
        print(f"end: {o1 * 100:+.1f} cm off the centre line (+ = right), heading {o2:+.0f} deg")
    print("spikes over the run, right / left:")
    for gr in sorted({g.split(":")[0] for g in groups}):
        print(f"  {gr:<7} {totals.get(gr + ':right', 0):6d} / {totals.get(gr + ':left', 0):<6d}")
print(f"video: {video_path}\nlog:   {csv_path}")

# one-line machine-readable summary (used by dodge_trials.py)
import json
summary = {"scenario": args.scenario, "side": args.side, "rv": args.rv if args.scenario == "loom" else None,
           "escaped": bool(first_escape), "escape_side": first_escape[2] if first_escape else None,
           "escape_t": round(first_escape[0], 3) if first_escape else None,
           "escape_deg": round(first_escape[3], 1) if first_escape else None,
           "warning_ms": round(first_escape[4], 0) if first_escape and args.scenario == "loom" else None,
           "moved_cm": round(moved * 100, 1), "turned_deg": round(turned, 1), "fell": bool(fell),
           "gap_cm": round(min_gap * 100, 2) if (args.scenario == "loom" and min_gap < 9) else None}
if args.scenario == "loom":
    summary["outcome"] = (("HIT" if summary["gap_cm"] <= 0 else "MISSED") if summary["gap_cm"] is not None
                          else ("ESCAPED" if first_escape else "NO ESCAPE"))
elif obstacle_geoms:
    summary["outcome"] = "WALKED INTO IT" if min_obs_gap <= 0 else "CLEAR"
    summary["obstacle_gap_cm"] = round(min_obs_gap * 100, 2)
else:
    summary["outcome"] = "FALSE ALARM" if first_escape else "OK (no escape)"
summary["walk"] = args.walk
summary["n_escapes"] = len(escapes)
summary["escapes"] = escapes
summary["totals"] = totals
summary["end_x_cm"] = round(float((data.qpos[qadr:qadr + 2] - start_xy) @ fwd0[:2]) * 100, 1)
summary["end_side_cm"] = round(float((data.qpos[qadr:qadr + 2] - start_xy) @ right0[:2]) * 100, 1)
with open(os.path.join(args.out, f"result_{args.scenario}{suffix}.json"), "w") as fh:
    json.dump(summary, fh)
