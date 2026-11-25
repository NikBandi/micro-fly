# Fly-brain Microduck — progress notes (paused 2026-11-23)

## END GOAL (user, 2026-11-24): publish the finished project as a GitHub repo
- At the end: clean up scripts, README with setup (WSL2, the 3 envs: duck/uv, vision/flyvis, brain/Shiu), data
  download steps (column_assignment.csv.gz, annotations, Shiu parquet are NOT redistributed - link them),
  results/videos, licenses/credits (FlyWire, flyvis, Shiu et al., Pollen microduck_rl).

## Plan
Fly brain (FlyWire connectome) drives Pollen/Hugging Face Microduck (2-legged, $399) in MuJoCo.
Pipeline: duck head camera -> flyvis (connectome visual model) -> fly brain / descending-neuron readout
-> small trained translation layer -> Microduck walking policy (vx, vy, yaw-rate command) -> MuJoCo.
Brain wiring stays fixed; only the translation layer is trained.

## User's machine
Windows 11 laptop, i7-12700H (14 cores), 32 GB RAM, NVIDIA RTX A2000 Laptop 8 GB (CUDA).
Plan: WSL2 Ubuntu. MuJoCo route, not Isaac Lab (8 GB VRAM is tight for Isaac Sim).

## Verified facts
### microduck_rl (github.com/pollen-robotics/microduck_rl)
- Python >=3.12,<3.13, managed by uv; mjlab 1.3.0, warp-lang, torch 2.9.1, better-actuator-models (bam), onnxruntime.
- Training needs a CUDA GPU (MuJoCo Warp). Running an exported ONNX policy runs in CPU MuJoCo.
- Run: `uv run scripts/infer_policy.py --walking <file>.onnx` (Linux-only: uses termios; keys read in the terminal).
  Keys: up/down forward/back, left/right strafe, A/E turn, space stop. Limits: vx ±0.3 m/s, vy ±0.2, yaw ±1.5 rad/s.
- Obs = 3 ang_vel + 3 proj_grav + 14 joint_pos + 14 joint_vel + 14 last_action + cmd(3, or 13 with --new-cmd-obs).
- Control 50 Hz, sim dt 0.005, decimation 4. Reusable: load_bam_model, load_mujoco_with_bam, PolicyInference.
- Default scene: src/mjlab_microduck/robot/microduck/scene.xml (includes robot_groundcontact.xml; only feet collide).
- Head camera exists: `<camera name="head_camera">` in robot_groundcontact.xml (default fovy 45; can widen via model.cam_fovy).
- Official policies on HF: pollen-robotics/microduck-policies -> alpha_walking.onnx, alpha_stand.onnx, velstand.onnx, etc.

### flyvis 1.2.0 (TuragaLab)
- `pip install flyvis`; tested on Linux py3.9–3.12. Pretrained: `flyvis download-pretrained` (Google Drive).
- 65 optic-lobe cell types, 45,669 cells, 721 hex columns. Includes T4a-d / T5a-d (motion). No LPLC2 (build looming readout from T4/T5).
- BoxEye(extent=15, kernel_size=13): input (samples, frames, H, W) -> (samples, frames, 1, 721).
- NetworkView(results_dir/"flow/0000/000").init_network(); network.fade_in_state(1.0, dt, frames[[0]]);
  network.simulate(movie[None], dt, initial_state=...) -> (1, T, 45669); LayerActivity(resp, network.connectome, keepref=True)["T4c"].
- Trained at dt = 1/50 s (matches duck's 50 Hz control loop).

### Whole brain
- Shiu et al. LIF model: github.com/philshiu/Drosophila_brain_model (Brian2; ships Connectivity_783.parquet + Completeness_783.csv).
- FlyWire v783 data: codex.flywire.ai/api/download, Zenodo record 10676866.

## Project goal (confirmed by user 2026-11-23)
This is a REINFORCEMENT LEARNING project: train the fly brain to learn to move its two-legged
Microduck body and avoid obstacles using vision. Proposed staged plan (awaiting user OK):
- Stage 0: baseline — Pollen's alpha_walking.onnx (sanity check + comparison).
- Stage 1: connectome-structured policy (FlyGM-style, arXiv 2602.17997) learns to WALK via PPO in mjlab
  (reuse Mjlab-Velocity-Flat-MicroDuck rewards), replacing the default MLP. Likely needs a reduced
  sub-connectome (visual projection -> central complex -> descending neurons) to fit 8 GB VRAM.
- Stage 2: add vision — head camera -> flyvis -> brain; pillar/obstacle scenes; reward = forward
  progress - collisions - falls. Fewer parallel envs (~128-512) because of camera rendering + flyvis.
- Risk: compute. Fallback = HF Jobs cloud (`--hf-jobs`).

## DIRECTION CHANGE (2026-11-23, user decision)
Hand-wired, NOT trained: fly brain's own wiring makes decisions; no RL on brain/walking (RL role TBD, maybe
only a few gains). Pipeline: duck camera (wide, split L/R) -> flyvis x2 -> BRIDGE -> FlyWire LIF brain
(Shiu/Brian2, own optic-lobe neurons silenced) -> descending neurons (P9/BPN fwd, DNa01/02 steer, MDN back,
Giant Fiber escape) -> speed/turn cmds -> Pollen gait (nerve-cord stand-in) -> MuJoCo.
First milestone: looming escape. Obstacles wanted: pillars, ramps, maze, gaps/jumping (jump = body lacks skill).

## Bridge data check (verified in workspace)
- flywire_annotations Supplemental_file1_neuron_annotations.tsv (GitHub, 139,248 neurons): has root_id,
  cell_type, side, positions, top_nt. Found: T4a 1462, T5a 1484, LPLC2 210, LC4 104, HSE/HSN/VS1 2 each,
  DNa01 4, DNa02 2, MDN 4, DNp01 (Giant Fiber) 2 — all split left/right.
- Shiu repo Completeness_783.csv / Connectivity_783.parquet use the same v783 root_ids.
- Still needed: per-neuron eye-column assignment for optic-lobe cells (Matsliah et al. 2024; Codex
  download, may need free login). Fallback: derive retinotopy from pos_x/y/z.

## Column data (searched 2026-11-23)
- Official file: https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz
  (columns: root_id, p, q, ...; Zhao hex "eye coordinates"; both eyes per OpticLobe.jl comments).
  Blocked from the build workspace (proxy 403) -> user downloads it. Backup: Zenodo record 17030629.
- Codex: 23,452 columnar neurons of 31 types in 796 columns (right optic lobe). Non-columnar flyvis types
  need separate handling.
- GitHub hsseung/OpticLobe.jl (reachable): data/RightEyeGrid.csv (p,q -> column id, right eye) and
  data/columns_Mi1.csv (Mi1 root_id per column). Hex axes: +p anterodorsal, +q posterodorsal (right eye).

## Bridge step 1 DONE (2026-11-23) — the build workspace
- build_bridge.py -> bridge_map.csv (32,811 FlyWire neurons -> flyvis cell index), bridge_meta.json, bridge_report.txt
- validate_bridge.py -> leave-one-out check (connectivity placement within 1 column: 96-100%), 0 duplicate
  flyvis cells per eye, bridge_hexmap.png.
- Lattice alignment: flyvis (u,v) -> FlyWire (p,q) = swap (u->q, v->p), chosen by matching T4/T5 preferred
  directions (Mi9->Mi4, Tm9->Tm1/Tm2 vectors); mean cos 0.886 vs runner-up 0.80; same answer both eyes.
- FlyWire eye is tall oval; ~490 of 721 flyvis columns covered per eye. Camera: right half mirror + rotate
  -159 deg CCW, left half rotate -158 deg CCW, before BoxEye (axes ~22 deg from orthogonal).
- Not in FlyWire: Mi3, Mi11, Mi12, Tm5Y, Tm28, Tm30, TmY13. CT1/Am pooled (flyvis_index -2).
- STEP 2 DONE: fly_eye.py (vision env), brain_bridge.py + make_test_rates.py (brain env), README.md, zipped as fly_bridge.zip.
  Idealised test: LPLC2 R 0.7/0/0 Hz, GF R 2/0/0, HS R 18.7/27.7/40.7 (looming/receding/sideways).
  Untrained-flyvis pipeline test passed (orientation pictures correct). Awaiting user's run with pretrained flyvis.
- FIRST REAL-EYE RUN (user, 2026-11-24): NOT fly-like yet. Receding drove eyes/GF more than looming (onset flash of
  big disc from grey steady state); sideways front-to-back bar drove T4b/T4d/T5b most (direction mismatch?);
  single trials noisy. LC4/DNp02/DNa02 scale with total input.
- v2 fixes (fly_bridge_v2.zip): fade_in_state from first frame, 0.3 s hold at end, calibration_v2_*,
  `--stimulus directions` check (gratings, 8 dirs, suggests extra rotation/mirror), brain_bridge --trials 5
  with store/restore.
- DIRECTION CHECK (pretrained, user): 0/8 OK, best fit = ~180 deg rotation. ROOT CAUSE FOUND: flyvis edge offsets
  are (target - source); build_bridge.py read them as source position. Fixed (negated flyvis vectors). Alignment
  matrix now [[0,-1],[-1,0]] (u->-q, v->-p); camera: right mirror + rotate +20.8, left rotate +21.7.
  Predicted from user's table after fix: ~4/8 within 45 deg, rest 47-64 deg off, T4d weak/axis-only (model 000).
  fly_bridge_v3.zip, calibration_v3_*.
- REAL-EYE RESULT v3 (5 trials): direction fixed (sideways -> T4a/T5a). HS 14.8/35.8/76.6 (L/R/S) good;
  DNp02 14.2/11.2/5.4, DNp11 4.3/0.5/2.6, LC4 10.9/8.9/6.7 prefer looming (modest). BUT giant fiber
  7.1/12.5/33.8 (sideways highest), left GF fires too (26 Hz sideways) -> non-visual/central route; LPLC2 ~0.4.
  Real eye drives ~200k puppet spikes/trial vs ~9k idealised.
- explain_neuron.py: per-input-type drive (spikes x weight) + timeline. Ideal looming: GF driven by DNp70 44%,
  LPLC2 31%.
- Real-eye trace (user): looming GF ramps during expansion, driven by LC4 60% + LPLC2 (biologically right).
  Receding GF fires only at start (fast contraction; LC4 + DNp70). Sideways GF driven by DNp70 64%; left GF by
  left DNp70 76%. DNp70 <-> DNp01 mutual excitation loop, kicked off by AVLP498/AVLP396 (central).
- brain_bridge.py --silence TYPE[:side] (zero outgoing synapses; label suffix __silence_...). Tested.
  User ran --silence DNp70: NO change (GF sideways 34) -> found BUG: Brian2 ignores boolean-mask assignment
  on syn.w. Silencing never applied AND puppet input cut never applied (2.17M synapse rows into puppets were
  live in every run so far). Fixed with np.flatnonzero + assert. Idealised results unchanged after fix
  (GF 3.3/0/0, HS 19.6/28.3/40.7). Real-eye runs must be redone (normal + --silence DNp70).
- CLEAN REAL-EYE RESULTS (fixed code, 5 trials), GF right Hz looming/receding/sideways:
  baseline 9.5/14.9/34.5 (leak had little effect); --silence DNp70: 5.2/1.5/2.5 (looming now highest),
  left GF ~0.3 all. => DNp70<->GF loop caused the misfiring. HS 20/36/77, DNp02 14/11/5 unchanged.
  Next: decide DNp70 handling for duck (silence vs weaken loop), then duck camera + output mapping.
- (old) NEXT: step 2 = voltage->rate conversion (gray-screen baseline) + Brian2 injection (mapped neurons become
  rate-driven spike sources); then looming-disc bench test (LPLC2/LC4/GF).

## Environment setup status
- WSL2 Ubuntu (user is root), nvidia-smi OK (RTX A2000, CUDA 13.3), git/build-essential/ffmpeg/uv installed.
- ~/flyduck/microduck_rl cloned, `uv sync` done, `uv run list-envs` works (also has Unitree G1/Go1 tasks).
- DONE: alpha_walking.onnx downloaded to policies/; runs with `--new-cmd-obs` (policy expects 61-D obs).
- Fixed laggy viewer: WSLg was llvmpipe; set GALLIUM_DRIVER=d3d12 + MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA in ~/.bashrc.
- DONE: flyvis env (~/flyduck/.venv-vision), pretrained models downloaded, NetworkView loads.
- DONE on user's laptop: Shiu brain model installed (~/flyduck/Drosophila_brain_model/.venv); test_looming_pathway.py
  -> GF 158 spikes/s, 761 active neurons, 30 s wall time. Cython not active (numpy fallback) — optional speed-up later.
- Shiu brain model: install needs "setuptools<81" (brian2 2.5.1 imports pkg_resources).
  Verified in workspace: test_looming_pathway.py excites 102 right LPLC2 -> right giant fiber (DNp01)
  fires ~155 spikes/s; 1 simulated second ~ 50 s wall time (numpy fallback, 2 cores).

## Next steps (not started)
1. duck_sees.py: pillar corridor scene, walk forward, record head-camera frames (.npz).
2. fly_eye.py: frames -> BoxEye -> flyvis -> T4/T5 responses + looming plot.
3. Translation layer + training loop.

## Stage 1: duck camera -> fly eyes -> brain (2026-11-24)
- Decision: one duck camera split L/R (option B); two side cameras = later upgrade.
- duck_eyes.py (duck env, `uv run python ~/flyduck/bridge/duck_eyes.py`): robot's head_camera points backwards
  into the head shell in MuJoCo, so adds 'fly_cam' on the head body 4 cm in front, level, forward, fovy 45
  (79 deg wide at 800x400). STAND keyframe, duck held still, mocap dark ball r=5 cm. Scenarios loom (2.0->0.2 m,
  18 deg right), recede, pass (0.6 m, 8->36.5 deg). Asserts ball visible in right half only. Tested on
  MuJoCo 4.0 and 3.10 (identical frames).
- fly_eye.py: for --frames, baseline = same scene held still, subtracted frame by frame (loom first frames 0.00 Hz).
- End-to-end tested with untrained flyvis + brain (works).
- USER REAL RUN (silence DNp70, 5 trials) loom/recede/pass: GF R 0.8/0.8/2.2 (weak, unselective);
  LC4 3.6/25.1/3.0, DNp02 3.8/39.5/0.0, HS 11.6/73.8/27.4; puppet spikes 86k/240k/56k. Receding drives most.
- Diagnosis: (1) geometry: camera half = ~40 deg wide but BoxEye maps it onto the whole fly eye (31 columns x ~5 deg
  = ~150 deg) -> ~4x magnification of sizes AND speeds; (2) LC4/DNp02 respond to fast change regardless of
  approach vs retreat (also seen with built-in movies). Proposed: two wide cameras (~120-150 deg each, square)
  to match fly eye geometry, then retest.
- TWO-EYE VERSION of duck_eyes.py: cameras eye_left/eye_right on head, yaw -/+70 deg, 150 deg each (10 deg frontal
  overlap), pinhole 1200px resampled to equal-angle 400x400 per eye (no magnification), duck body hidden (geomgroup
  2,3 off), shadows off, znear 0.0005. loom from 60 deg right, 2.0 -> 0.07 m (~90 deg); pass 0.6 m, 25 -> 125 deg.
  Asserts: ball only in right eye; loom largest at end. Tested on MuJoCo 3.10; pipeline tested with untrained eye.
- USER RUN two-eye: still recede >> loom (LC4 2.5/19.0/0, DNp02 2.6/31.1/0, GF ~0.8/0.6/0, LPLC2 ~0).
  explain LC4: recede driven by T2 62% + Tm4 26%, sustained 2.2/100ms through static end-hold -> BASELINE ARTIFACT
  (first-frame baseline = big ball for recede; brightness cells report 'brighter' forever).
  FIX: duck_eyes saves 'background' (empty scene); fly_eye baseline = empty scene held still. Untrained test:
  loom hold 7.45 Hz / start 0.02; recede start 7.81 / hold 0.03 (symmetric). LPLC2 silent = separate issue.

## Stage 2: closed loop (2026-11-24) — built, plumbing tested, waiting for alpha_walking.onnx from user
- Option A chosen: escape signal = DNp01 + DNp02 + DNp11 spikes per side per 100 ms; >= threshold (default 3)
  -> back away (vx -0.3) + turn away (yaw +-1.5) for 0.6 s; else stand. Silenced DNp70 + Li33.
- Files: ipc.py (length-prefixed npz over pipes), eye_worker.py (vision env, flyvis state kept between chunks;
  baseline = steady response to empty scene after 1 s fade-in + 1 s settle), brain_worker.py (brain env,
  network_operation updates Poisson rates every 20 ms), fly_loop.py (duck env coordinator, BAM + Pollen
  policy, two fly-eye cameras, ball mocap, video loop_<scenario>.mp4 + csv). Scenarios loom/pass/none.
- Verified: flyvis chunked == full (exact); brain_worker chunked ~= brain_bridge totals; eye_worker == fly_eye
  per neuron (corr 1.000 after fixing a column-order bug); full loop runs with dummy controller + untrained eyes.
- Got alpha_walking.onnx. FOUND: PolicyInference default use_projected_gravity=False but Pollen's script uses
  True -> duck fell after a few seconds. Fixed (use_projected_gravity=True); all motion tests stable.
- Escape = vx -0.3 + yaw +-1.5 for 1.0 s: backs 7.8 cm, turns 47 deg away, stays upright. Yaw + = left (verified).
- Full loop with real controller + untrained eyes: runs 2.5 s sim in ~150 s (my CPU), duck stays up, video OK.
- Delivered fly_loop_stage2.zip (ipc, eye_worker, brain_worker, fly_loop). --motion-test option for quick check.

## MILESTONE 1 RESULT (user, 2026-11-24, trained eyes, threshold 3, DNp70+Li33 silenced)
- loom: 0 escape spikes until ball at 0.07 m (t=1.6 s), then R 7,10,3,4... L 0 -> escape on RIGHT side,
  duck turned +43 deg (left, away), moved 7 cm, stayed standing. Correct side/direction, but TOO LATE
  (no spikes at 30 cm / 11 cm; ball ~50 deg before response).
- pass: 0 spikes, no escape. none: 0 spikes, no escape. => escape is threat-specific.
- Bug found in pass scenario: ball sat at 2 m then teleported to 0.6 m at t=0.5 s. Fixed (starts at 0.6 m).
  Result still valid (no escape even with the jump).
- Next options: timing (slower approach; why LPLC2 silent / late response), or next behaviour (steering).

## Stage 3: loom timing (2026-11-24) — fly_loop_stage3.zip, waiting for user's run
- Connectome check (Shiu weights, right side, synapses per cell): LPLC2 gets ~390 excitatory from eye puppets
  (T4/T5 a-d, Tm20, Tm4, TmY3) but ~310 inhibitory (PVLP011 -47, LPi11 -34, TmY5a -29 [eye puppet], Tm5e -24).
  LIF threshold needs ~5,000 synapse*Hz net (7 mV / (0.275 mV * 5 ms)) -> a small loom covering a few columns of an
  LPLC2 arm can't drive it; explains LPLC2 ~silent until the ball is huge. GF (DNp01) has NO direct eye input:
  LPLC2 +622, LC4 +431, DNp70 +338 (silenced), lots of central inhibition (~2,500). LC4: T2 +222, TmY3 +194,
  Tm4 +141, Tm2 +123 (all eye puppets).
- Milestone-1 loom was very fast: r/v = 26 ms (2.0 -> 0.07 m in 1 s); escape at t=1.6 s was AFTER virtual
  collision (t~1.53).
- fly_loop.py: --rv (ms, loom speed as r/v; default 26 = old loom), --start-deg (2.9), loom stops 7 cm from eye
  then holds 0.3 s; duration auto-extends; --observe (brain read out, duck never moves); --eye-gain (passed to
  eye_worker); --tag (output names). CSV adds theta_deg, ttc_ms, eye_<T4|T5|T2|Tm4|TmY3|TmY5a>:<side> (summed Hz).
- eye_worker.py: --gain G (rate = RMAX*clip(G*dv/scale)), returns per-eye looming-pathway drive summary.
- loom_sweep.py: runs observe-mode looms for r/v 20/40/80/160 ms, prints angular size + time-to-collision at first
  escape signal, first-spike angle of LPLC2/LC4/DNp02/GF, eye T4+T5 half-peak angle; loom_sweep_g<G>.csv/.png.
  Fly-like signature = constant escape angle, warning time growing with r/v.
- inputs_of.py: the connectome input table above for any type.
- Tested in workspace with stand-ins (Pollen scene + stub policy/eyes/brain; flyvis/torch stubs): geometry,
  logging, gain, drive sums, sweep summary/plot all correct. Real flyvis/brain not runnable here (pretrained
  weights + column file blocked).
- Next after user's sweep: if eye drive early but LPLC2 late -> eye->brain gain question (re-run sweep with
  --eye-gain 2, 4 AND re-check pass/none for false alarms); if eye drive late too -> vision side.
- SWEEP RESULT gain 1 (user, 2026-11-24): escape always at the 7 cm stop (85-93 deg) for r/v 20/40/80/160 ->
  warning -110/-19/+62/+223 ms (growth only because the stimulus ends at a fixed size). Per 100 ms bins:
  LC4 R first spikes at ~15 deg every run, ~20-38 spikes (of 50 cells) by 35-56 deg, peak ~130 (~25 Hz/cell) = fly-like.
  LPLC2 R first at 47-57 deg but WEAK: peak 11-19 spikes/102 cells (~2 Hz/cell). GF (DNp01 R) nearly silent:
  <=2 spikes/100 ms ever; rv160 single spikes at 35, 57, 85 deg; rv80 none before end. Escape count is mostly
  DNp02 + DNp11 at the end. Eye T5 >> T4 (dark ball = OFF edges). Ball disappearance makes a big eye transient.
  => bottleneck = LPLC2 under-driven (and GF). Biology note: one GF spike triggers a real fly's short-mode takeoff,
  so a 'GF >= 1 spike' readout is defensible, but GF is too sparse/inconsistent now. Next: gain-2 sweep + pass/none.
- SWEEP gain 2 (user): first spike deg (LPLC2/LC4/GF) rv20 92/21/92, rv40 47/11/92, rv80 33/10/53, rv160 25/9/25.
  Escape: rv160 at 57 deg (+323 ms, BEFORE the stop - first real early escape), rv80 still at stop (93, +62; 2 spikes
  at 53 deg), rv20/40 at stop. pass + none at gain 2: ZERO escape spikes (no false alarms).
  Fast looms (20/40 ms) go 23->92 deg inside one 100 ms decision bin -> decision latency is a structural limit.
- fly_loop_stage3b: --step-ms 20|40|100 (eyes+brain+decision every step; escape count = sliding 100 ms window, so
  threshold meaning unchanged; 100 = old behaviour). loom_sweep --step-ms, tag g<G>_s<ms>, plot Hz uses step.
  Tested with stubs. Asked user: sweep gain 2 + step 20, sweep gain 3 (step 100) + pass/none gain 3.
- RESULTS (user, 2026-11-24): gain 2 + step 20: escape at 92/92/70/57 deg, warning -50/+21/+122/+323 ms for r/v
  20/40/80/160 (LPLC2 first 47/26/26/29 deg, GF first 92/21/47/33). Gain 3 + step 100: escapes same as gain 2 step 100
  (-110/-19/+62/+323); only first spikes earlier. pass g3: one 1-spike bin (below threshold), none g3: 0.
  => 20 ms decisions are the real win; gain 3 adds nothing. CHOSEN SETTINGS: --eye-gain 2 --step-ms 20.
  r/v 20 ms is physically out of reach (ball 33 -> 92 deg in 40 ms). Next: behaving runs (no --observe) loom rv 40/80/160
  + pass/none at gain 2 step 20 = Milestone 1 final.
- MILESTONE 1 FINAL (behaving, gain 2, step 20, user 2026-11-24): loom rv40 escape at 92 deg +21 ms, rv80 at 53 deg
  +161 ms, rv160 at 50 deg +363 ms; all right side -> back + turn left; duck moved 6-8 cm, turned +33..+40 deg, stayed
  standing. pass + none: 0 escape spikes, no movement. => NEURAL milestone met (threat-specific, correct side, before
  contact, warning grows with r/v). BUT closest approach still 7-8 cm: body too slow to clear the path (backing at
  <=0.3 m/s from a ball 60 deg right gains only ~cos60 of that; turning barely moves the head; gait needs ramp-up).
  Ball has no collision (contype 0) and stops 7 cm from start head pos, so 'closest' metric is crude.
  Ideas: escape straight away from threat (back + sidestep, no turn), ball continues through and collision/miss test.
  Videos loop_loom_final_rv{40,80,160}.mp4, loop_pass_final.mp4, loop_none_final.mp4 = demo clips.
- Stage 4 (dodge), fly_loop_stage4.zip: --escape away (default; vx -0.3 + vy +-0.2 away, yaw 0; +vy = left per Pollen
  arrow keys) | turn (old). --fly-through: loom ball continues 0.3 m past the start eye position. Hit test via
  mujoco.mj_geomDistance(ball, each of 81 duck mesh geoms) per frame -> min gap; log gap_cm; prints HIT/MISSED.
  Static check: ball 10 cm beside head camera -> gap +0.6 cm (barely misses), 5 cm -> -4.2 (hit). So dodge needs
  ~10 cm sideways at crossing. --motion-test now reports forward/sideways displacement and distance gained from a
  60-deg-right threat. Tested with stubs (observe -> HIT -4.2 cm; escaping stub -> MISSED).
- DODGE RUN 1 (user): motion-test 'away' (vx -0.3, vy +0.2, yaw 0) moved only -0.3 cm fwd / +0.6 cm left in 1 s (!)
  vs old back+turn 7.8 cm -> real policy barely executes combined back+sidestep. Results (fly-through): control
  (observe) HIT -4.5 cm; rv160 escape at 57 deg +323 ms -> HIT -3.4; rv80 60 deg +141 ms -> HIT -3.9; rv40 at contact
  -> HIT -4.2. Duck moved 0-1 cm. Brain timing fine; motor command was the failure.
- Geometry insight: to MISS, the head must get OFF the ball's line (~10 cm), not just away. For a ball from 60 deg
  right the perpendicular is forward-left (0.87 fwd, 0.5 left) -> forward+left may dodge best.
- fly_loop_stage4b: --motion-sweep (9 commands from fresh stand, 'away' + 'off path' cm at 0.5/1/1.5 s, FELL flag),
  --escape-cmd VX VY YAW (threat-right version; mirrored for left). Tested with stubs.
- MOTION SWEEP (real policy, user): head off-ball-line cm at 0.5/1.0/1.5 s: back 1.7/1.7/1.7 (stalls), sidestep L/R ~0,
  back+sidestep ~1, back+turnL 3.3/9.1/16.9, back+left+turnL 3.3/10.5/18.5 (best), forward 0.1/4.9/10.4,
  fwd+sidestepL 1.5/7.5/14.4. Only commands with yaw (or forward) make the duck walk; pure back/sideways stall.
  Compared our setup to Pollen's infer_policy main(): same BAM load (kp_fw 200, vin 7.4, max_current 0), same spec
  edits, timestep 0.005, decimation 4, action_scale 1, no delay, projected gravity, 13-D cmd, walking-only switch
  bypass -> no mismatch found. Likely the alpha_walking policy itself is weak at back/sideways. User can confirm in
  Pollen's viewer (down arrow = back).
- dodge2_rv160 with --escape-cmd 0.3 0.2 0: escape at 60 deg +303 ms, duck moved 16 cm total but HIT -6.1 cm.
  Side effect: ball passing through head -> left eye triggers a 2nd escape the other way.
- Warning-time math: escape at ~55 deg -> ttc ~ 1.9 x r/v (r/v 160 -> ~0.3 s). Best command needs ~1 s for 10 cm off
  line -> need ttc ~1 s: r/v ~500 ms, or earlier trigger (threshold 2; first escape spikes at ~25-35 deg -> ttc ~3.5 r/v).
- *** FIRST DODGES (user, 2026-11-24), escape-cmd -0.3 0.2 1.5 (back + sidestep + turn away), gain 2, step 20:
  r/v 500 (0.1 m/s), threshold 3: escape at 44 deg, +1339 ms -> MISSED by +7.3 cm (duck moved 43 cm, turned 58 deg).
  r/v 160, threshold 2: escape at 43 deg, +423 ms -> MISSED by +2.3 cm (moved 22 cm). pass, threshold 2: no escape
  (max 1 spike). => Milestone 1 achieved: fly connectome sees the ball and the duck dodges it.
  Caveats: single trials (Poisson noise), rv160 margin small, none@t2 not yet run, only right-side looms tested.
  CANDIDATE FINAL SETTINGS: --eye-gain 2 --step-ms 20 --threshold 2 --escape-cmd -0.3 0.2 1.5 --fly-through.
- *** MILESTONE 1 COMPLETE (user, 2026-11-24). Final settings: --eye-gain 2 --step-ms 20 --threshold 2
  --escape-cmd -0.3 0.2 1.5 --fly-through. r/v 160 x3: MISSED +2.3 / +5.1 / +2.3 cm (escape 43/25/43 deg,
  +423/+723/+423 ms). r/v 500 (threshold 3): MISSED +7.3. pass t2 + none t2: no escape. Run 1 and run 3 gave identical
  traces (randomness may repeat between runs - check seeding before claiming statistics). Demo videos:
  loop_loom_dodge_rv500.mp4, loop_loom_dodge_rv160_t2_run2.mp4, loop_pass_pass_t2.mp4, loop_none_none_t2.mp4.
  Known limits: right-side looms only; alpha_walking policy can't walk straight back/sideways (only with yaw).
- Run1 vs run3 'identical' explained: spike counts differ step by step (randomness fine), but both crossed threshold at
  the same 20 ms step -> same deterministic escape -> same gap. Outcome depends mainly on trigger time.
- fly_loop_stage5: --side right|left (mirrors loom + pass), writes result_<scenario>_<tag>.json summary.
  dodge_trials.py: N fresh-process loom trials per side (final settings baked in) + pass per side + none;
  prints table (outcome, gap, escape side, deg, warning, moved, fell) + dodged/false-alarm totals, dodge_trials.csv.
  Tested with stubs. User asked: step 1 (trials) then step 2 (GitHub packaging).
- STEP 2 (GitHub) draft in workspace the build workspace -> flyduck_repo_draft.zip: repo root = ~/flyduck/bridge
  (flat scripts), README (pipeline, findings, setup for 3 envs, data table, run commands, credits, limitations),
  MIT LICENSE (holder 'Sri Nikhil' - confirm), .gitignore (data/ + outputs excluded, media/ kept),
  requirements-vision/brain.txt, docs/development_log.md (= PROGRESS), tools/export_flyvis_inputs.py (UNTESTED:
  recreates data/flyvis_nodes.csv + fib25 json - compare with user's existing files). Results table to fill from
  dodge_trials. Open: public/private, demo gif from loop_loom_dodge_rv500.mp4.
- REPO NAME (user, 2026-11-24): micro-fly -> https://github.com/NikBandi/micro-fly (public, empty when checked).
  README title renamed; clone into ~/flyduck/bridge; *.zip added to .gitignore. micro-fly_repo.zip sent.
  Plan: git init inside the user's existing ~/flyduck/bridge, review `git status --short`, then first push.
- *** DODGE TRIALS (user, 2026-11-24, final settings, r/v 160, 3/side, fresh processes): dodged 6/6 (right gaps +2.3/+2.4/
  +4.3 cm, left +2.1/+1.4/+1.2), escape on threatened side 6/6, escape at 27-45 deg, warning +403..+663 ms, moved 22-25 cm,
  false alarms 0/3 (pass right, pass left, none), falls 0. Left-side margins smaller. README results table filled.
- REPO STAGED (user): git init -b main in ~/flyduck/bridge, `git status` = 23 clean files (scripts, docs, LICENSE,
  requirements, media/dodge_rv500.gif 4.9 MB from last 6 s of loop_loom_dodge_rv500.mp4). Not yet committed/pushed.
  User's data/ has only the annotations TSV: bridge_map.csv was built in the build workspace and copied over, so
  tools/export_flyvis_inputs.py + a full build_bridge rebuild (needs column_assignment.csv.gz) are still unverified.

## MILESTONE 2: steering around obstacles (started 2026-11-24)
- Connectome check (Shiu weights, signed input fraction from visual projection neurons per eye): DNa01/DNa02/DNa03
  get net INHIBITION from the same-side eye and weak excitation from the opposite eye (DNa01 R: right-eye VPN -0.047,
  left-eye +0.006 at 1 hop; biggest single input LT51 (same side) -320). If DNa01/02 = turn toward their own side
  (Rayshubskiy 2020), an object on the right should favour LEFT turning = avoidance. DNb05 is strongly driven by the
  same-side eye (+0.27; LPLC4 +2053, LLPC2 +666). DNa02's top inputs are central-complex (PFL3) - goal steering.
- Stage 1 (milestone2_stage1.zip, tested with stubs): fly_loop.py scenarios pillar (dark cylinder r 3 cm, 0.5 m tall,
  --pillar-ahead 0.6 m, --pillar-offset 0.06 m toward --side; no collision, gap via mj_geomDistance) and corridor
  (striped walls 4 cm stripes, half width --corridor-half 0.18 m, 3 m long, invisible proxy boxes for the gap);
  --walk VX (default 0.3 for these), --no-escape (log escape, keep walking); status line shows pillar distance/bearing
  + DNa01/DNa02 L/R; result json adds totals per group/side, obstacle_gap_cm, end_x_cm, end_side_cm.
  brain_worker READOUT + DNa03, DNb01, DNb05, LPLC4, LT51. steer_probe.py: open walk / pillar R / pillar L / corridor
  (7 s each, --no-escape), table of R/L spikes + pillar-side contrast + per-second R-L near the pillar.
  Waiting for user's probe run.
- PROBE RUN 1 (user, 2026-11-24, walk 0.3, no steering): the duck does NOT walk straight - identical body path in all
  4 runs: 64 cm forward, drifted +14.4 cm right, turned -26 deg (right) in 7 s (~0.09 m/s). So the pillar ended up on
  the duck's LEFT in both pillar runs (pillar R bearing +3 deg at 1 s, then -11..-137 deg; gap +0.5 cm, just missed;
  pillar L clear +11.3), corridor: walked into the right wall (-5.6 cm). Pillar-side contrast therefore confounded.
  Still informative: as the pillar nears on the left (4-6 s, 10-20 cm, -36..-98 deg) left LC4 (+500-1100/s), left
  LPLC4 (~100/s), left DNb05 (30-42/s), left DNp02, left DNp11 rise -> same-side visual pathway (as connectome said
  for DNb05). DNa01/DNa02/DNa03 show only small differences (<15/s), DNa02 R>L in every run incl. open (baseline).
  Open walk baseline asymmetries (veer-related or bridge): HSS L 507 vs R 5, HSN R 445 vs L 131, DNa02 R 199/L 66,
  LPLC4 R 271/L 24. ESCAPE SIGNAL crosses threshold 2 at ~0.3 s in EVERY walking run (DNp01 R ~12/s while walking)
  -> walking self-motion would trigger constant escapes; needs fixing before escape + walking are combined.
- Fix: --hold-heading K (default 2 when walking): yaw cmd = -K x heading error (stand-in for haltere/vestibular
  course control; steering from the brain will be added on top). Stub test with veer: -12 deg -> -2 deg.
  milestone2_stage1b.zip (fly_loop.py) -> rerun probe.
- PROBE RUN 2 (heading hold K=2, user 2026-11-24): straight walk (64 cm fwd, drift +1 cm, -1 deg). pillar R HIT -3.4,
  pillar L HIT -2.6, corridor gap 0.0 (odd - check CSV). Pillar-side contrast (+ = fires on the pillar's side):
  LC4 +4654, LPLC4 +745, LPLC2 +298, DNb05 +244, DNp02 +212, LT51 +166, HSN +413, HSE +225, HSS +146, DNp11 +91,
  DNa01 +60, DNa02 +60, DNb01 +9, DNa03 -4, DNp01 -40. Everything visual is IPSILATERAL to the pillar.
  DNb05 (a descending neuron) R-L per s, open baseline ~+5: pillar R +14/+13/+13/+53/+41 at 31/21/11/5/4 cm;
  pillar L +10/+3/-40/-63 at 32/22/13/3 cm -> clean but late (separates ~13-20 cm). LPLC4 (baseline +33/s): pillar R
  +78/+68/+61/+145 at 31/21/11/5 cm vs pillar L +61/+3/-132/-224 at 32/22/13/3 -> separates from ~21 cm.
  DNa01/DNa02 slightly ipsilateral (would turn TOWARD the pillar if they mean ipsilateral turning; cf. flies' dark
  stripe fixation) and small. Walking escape false alarm persists (threshold crossed at ~0.3-0.4 s every run).
- Stage 2 (milestone2_stage2.zip): --steer GROUP[,GROUP] --steer-gain G --steer-window 300 --steer-bias B:
  yaw = hold-heading + G * ((R-L spikes/s over window) - B), + = left -> turns away from the side that fires more.
  Sign bug caught in stub test (first version turned toward) and fixed. steer_probe --label NAME passes extra args.
  Plan: A = --steer DNb05 --steer-gain 0.04 --steer-bias 5; B = --steer LPLC4 --steer-gain 0.008 --steer-bias 33.
- *** FIRST FLY-BRAIN STEERING (user 2026-11-25): both pillar runs now CLEAR with either readout (without steering
  both were HITs -3.4/-2.6). A = DNb05 (gain 0.04, bias 5): pillar R +4.7 cm (drift -8.6 = moved left, away),
  pillar L +6.0 (drift +9.2 = right, away); open walk drift +2.7, -3 deg (bias OK). B = LPLC4 (gain 0.008, bias 33):
  pillar R +1.3, pillar L +7.1, open +2.0. Single runs. CHOSEN: A (DNb05 is a real descending neuron, better margin).
  Corridor gap 0.0 at t=0.5 s with the duck centred (off 0.9 cm) -> measurement bug: mj_geomDistance box(thin proxy)
  vs duck mesh returned 0 on the user's MuJoCo (worked on 4.0.1 here). FIX: corridor gap now computed from the duck's
  own mesh vertices vs the wall planes (+-corridor_half, from 0.1 m ahead). steerB corridor real collision likely
  (drift +26 cm). steer_probe --repeats N (extra pillar runs per side; prints cleared X/Y).
  milestone2_stage2b.zip. Next: DNb05 x3 per side + corridor with/without steering.
- *** PILLAR AVOIDANCE STATS (user 2026-11-25, DNb05 gain 0.04 bias 5): CLEARED 6/6 (R +3.8/+4.0/+4.5 cm, moved
  7.6-8.8 cm left; L +7.6/+11.0/+8.7, moved 11-15 cm right). Without steering both sides HIT. Open walk with steering
  drift +3.3 cm, -2 deg. Corridor with DNb05 steering (new vertex gap test): CLEAR +0.2 cm, drift +8.3 (barely).
  No-steer corridor run result pending. Escape signal still crosses threshold while walking (0.22-0.46 s).
- FAST BRAIN (fast_brain.zip = brain_worker.py): profiling showed ~0.3 s Brian2 set-up per net.run() call, i.e. most of
  the time with 20 ms steps. New worker: ONE long net.run; IPC happens inside a network_operation at the start of each
  20 ms bin (reports the finished chunk's counts, blocks for the next chunk; EOF -> net.stop). --codegen auto|cython|numpy
  (auto = Cython if CythonCodeObject.is_available()). Benchmark here (brian2 2.11, 2 cores, 1 s sim, random 10 Hz drive):
  many 20 ms runs numpy 38.6 s / cython 23.9 s; one long run numpy 29.4 s / cython 8.6 s; numpy and cython gave
  IDENTICAL spike totals (same seed). Full fly_loop + real brain_worker (cython) + stub eyes with real root ids: runs,
  exits cleanly. User needs Cython in the brain venv.
- Cython on user's machine: brain venv had Cython 3.3.1 -> Brian2 2.5.1 test compile fails (ImportError). Fixed with
  `uv pip install --python ~/flyduck/Drosophila_brain_model/.venv/bin/python "cython<3"` (0.29.37) -> C ready: True.
  Numpy-mode single-run worker: pillar run 207 s (was ~220-240). 7th pillar dodge: +8.3 cm. Cython timing pending.
- Stage 3 (milestone2_stage3.zip: fly_loop.py, walk_trials.py): --escape-gate GROUP:N (escape needs escape count >=
  threshold AND GROUP >= N spikes same side in the 100 ms window; LPLC2 was 0-5 spikes per 7 s of plain walking).
  --loom-start T (ball launch time); while walking the ball aims at the head's predicted impact position (head +
  velocity over last 0.5 s x time left, frozen at launch) - stub test: ball went through the eye (0 cm).
  escapes list + n_escapes in result json. walk_trials.py: open / pillar R / pillar L / ball R+L while walking
  (rv 160, start 6 deg, launch 2 s) / standing ball R; all with DNb05 steering + LPLC2:2 gate.
- C mode speed (from walk_trials run stamps, user): ~1.5-2 min per 7 s run with Cython vs ~3.5-4 min before.
  (The separate two-run timing test was never reported.)
- WALK TRIALS m2 (user 2026-11-25, C mode: ~1.5-2 min per run, batch ~10 min): open 1 false escape (left 2.18 s);
  pillar R CLEAR +26.1 but 3 escapes (first right 3.48 s, turned +104); pillar L CLEAR +42.5, 3 escapes (first left
  1.68 s at ~44 cm, early); ball R walking HIT -3.4 (escape right at 4.86 s, ~200 ms before impact - late);
  ball L walking MISSED +6.1 but first escape on the WRONG side (right 4.48 s); standing ball MISSED +2.6 (escape 6.34 s,
  ~490 ms warning, M1 intact). => LPLC2:2 gate cut false alarms from constant to ~1/run but walking still disturbs
  escape timing/side. escape_report.py (reads loop_*_m2_*.csv: totals, per-second R/L of DNp01/02/11, LPLC2, LC4,
  DNb05, and 120 ms before each escape) sent for diagnosis.
- escape_report on m2 (user): walking baseline escape counts DNp01 R ~13/s, DNp02 L ~9/s, DNp11 L ~10/s -> window sum
  sits at 2-3 constantly; LPLC2 is the only discriminator (open 0/3 in 7 s, pillar R 68/4, pillar L 0/16, stand ball
  24/7) and gate 2 was passed by 2 stray LPLC2 spikes (open false alarm). Standing loom escape at 37 deg had only
  LPLC2 2 in window. Walking-ball runs: aim frozen at launch (t=2) but the duck changed course (DNb05 R>L transient at
  2-3 s in ALL walking runs: 9-22 vs 4-8 -> left turn) -> ball R reached only 20 deg at nominal ttc 197 ms; ball L
  apparently crossed in front and came in from the RIGHT (LPLC2 R 22 in 4-5 s) -> 'wrong side' was test geometry.
- Stage 3b: --home-deg 15 (walking loom: ball re-aims at predicted impact point every frame until it is 15 deg wide,
  then flies straight; stub test: duck veering 93 deg still hit, 3 cm from eye); with a gate, escape SIDE = side with
  more gate-group (LPLC2) spikes; walk_trials default gate 3. milestone2_stage3b.zip.
- *** WALK + STEER + ESCAPE WORKS (user 2026-11-25, m2b: gate LPLC2>=3, LPLC2 picks side, homing ball 15 deg): 6/6 as
  wanted. open walk 0 escapes (drift +1.8, -6 deg); pillar R CLEAR +22.4 (3 escapes, all right side, first 3.72 s);
  pillar L CLEAR +28.7 (2 escapes, left, first 3.22 s); walking ball R MISSED +2.1 (escape right at 61 deg, +277 ms);
  walking ball L MISSED +6.0 (left, 44 deg, +417 ms); standing ball MISSED +2.4 (right, 41 deg, +443 ms; M1 intact).
  Near pillars the escape (not just steering) does the avoiding. Single runs; margins 2-6 cm on balls.
- WRAP-UP MILESTONE 2 (user 2026-11-25: "wrap up milestone 2 for today and update everything on github; README with
  stages + images/videos"). walk_trials --repeats N (tags _2,_3; per-scene tally; walk_trials_<label>_summary.json).
  tools/make_media.py (GIFs from loop_*.mp4: m1_escape_not_enough, m1_pass_no_escape, m2_pillar_no_steer_vs_steer
  (hstack probe_right vs dnb05_right), m2_walk_steer_escape_pillar (m2b), m2_walking_ball_dodge (m2b, hero); PNGs
  m1_loom_sweep, eye_check_duck_loom). tools/fill_readme.py fills README between M2_RESULTS markers. README rewritten
  into 10 stages with media. requirements-brain + cython<3. .gitignore + M2 outputs. Repo zip micro-fly_m2.zip.
- m2b REPEAT (user accidentally re-ran by pasting old output into the shell, 2026-11-25): again 6/6 as wanted: open 0
  escapes (+4.1, -5 deg); pillar R CLEAR +29.5 (3 escapes, right, first 3.06 s); pillar L CLEAR +28.2 (2, left, 3.50 s);
  ball R MISSED +3.4 (right, 58 deg, +277 ms); ball L MISSED +1.0 (left, 49 deg, +337 ms); stand ball MISSED +2.4
  (identical to before: same trigger step). This overwrote the m2b videos used for the README GIFs (still valid).
