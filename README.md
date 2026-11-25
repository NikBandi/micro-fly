# micro-fly: a real fly brain drives a walking robot duck

A simulated two-legged robot (Pollen Robotics' **Microduck**, in MuJoCo) sees the world through
**simulated fly eyes**, and the **wiring diagram of a real fruit fly brain** (FlyWire, ~139,000
neurons) decides what it does. Nothing in the brain is trained: its decisions come from the
connectome's own wiring.

**Milestone 1 - looming escape:** a ball flies at the duck. The fly's looming-detector neurons fire,
the giant-fiber escape pathway triggers on the correct side, and the duck steps out of the way.

![The duck dodges a ball that the fly brain saw coming](media/dodge_rv500.gif)

*Ball approaching at r/v 500 ms from 60° right. The fly brain escapes at 44°, 1.3 s before impact, and the ball misses by 7 cm.*

Results with the final settings (`dodge_trials.py`: 3 independent trials per side, ball at r/v 160 ms, about 0.3 m/s, from 60°):

| | Result |
|---|---|
| Looming balls dodged | **6 / 6** (3 from the right, 3 from the left) |
| Escape on the threatened side | **6 / 6** |
| False alarms (ball passing by on each side, empty scene) | **0 / 3** |
| Falls | 0 |
| Ball's angular size when the escape triggered | 27 - 45° |
| Warning before impact | 0.40 - 0.66 s |
| Closest gap between ball and duck | +1.2 to +4.3 cm (mean +2.3 cm) |

<details><summary>Per-trial table</summary>

| run | outcome | gap cm | escape side | deg | warning ms | moved cm |
|---|---|---|---|---|---|---|
| right 1 | missed | +2.3 | right | 43 | +423 | 22 |
| right 2 | missed | +2.4 | right | 41 | +443 | 22 |
| right 3 | missed | +4.3 | right | 27 | +663 | 25 |
| left 1 | missed | +2.1 | left | 39 | +463 | 24 |
| left 2 | missed | +1.4 | left | 42 | +423 | 23 |
| left 3 | missed | +1.2 | left | 45 | +403 | 23 |
| pass right | no escape | - | - | - | - | 0 |
| pass left | no escape | - | - | - | - | 0 |
| empty scene | no escape | - | - | - | - | 0 |

</details>

## How it works

```
 MuJoCo Microduck ──► two 150° head cameras ──► flyvis (connectome-constrained fly eye, 45,669 cells)
        ▲                                                   │  voltages -> firing rates
        │                                                   ▼
 Pollen walking policy ◄── escape command ◄── FlyWire LIF brain (Shiu et al.): eye-driven "puppet"
 (vx, vy, yaw)               (side of the        neurons -> LC4 / LPLC2 -> giant fiber (DNp01),
                              stronger signal)    DNp02, DNp11 = escape spikes, left vs right
```

1. **Eyes.** Each 150° camera image is resampled to equal-angle pixels, rotated to match the fly
   eye's lattice, and fed to the pretrained [flyvis](https://github.com/TuragaLab/flyvis) network.
2. **Bridge** (`build_bridge.py`). 32,811 FlyWire optic-lobe neurons are matched to the flyvis cell
   of the same type in the same eye column. The two hexagonal lattices are aligned using the
   anatomy of the motion detectors (T4/T5 preferred directions). Each matched FlyWire neuron
   becomes a *puppet*: its brain inputs are cut and it fires at the rate its flyvis twin dictates.
3. **Brain** ([Shiu et al. leaky integrate-and-fire model](https://github.com/philshiu/Drosophila_brain_model),
   FlyWire v783, Brian2). Runs in 20 ms steps alongside the simulation.
4. **Decision.** Escape spikes (DNp01 + DNp02 + DNp11) are counted per side over the last 100 ms.
   At 2 or more, the duck backs away, sidesteps and turns away from that side for 1 s
   (Pollen's walking policy stands in for the fly's nerve cord).

Two brain neurons are silenced (DNp70, Li33): DNp70 and the giant fiber excite each other in a
loop that made the giant fiber fire for sideways motion (see `docs/development_log.md`).

## What we learned (short version)

- The fly circuit behaves like a fly's: LC4 responds early (~10-15°) and LPLC2 later (~25-30°),
  and slower approaches give more warning before impact.
- LPLC2 needs strong input. Each cell gets ~390 excitatory synapses from eye motion cells but
  ~310 inhibitory ones (`inputs_of.py`). The eye-to-brain gain had to be doubled for LPLC2 to
  respond before the ball is huge.
- Decision latency matters: deciding every 100 ms was too slow for fast balls; 20 ms fixed it.
- The body is the bottleneck: the `alpha_walking` policy walks only slowly backwards or sideways
  unless it is also turning (`fly_loop.py --motion-sweep`). A real fly jumps.

## Setup (WSL2 Ubuntu or Linux; tested on an RTX A2000 laptop)

The scripts expect this layout, so clone this repository into `~/flyduck/bridge`:
```
~/flyduck/
  bridge/                    <- this repository (git clone https://github.com/NikBandi/micro-fly ~/flyduck/bridge)
  microduck_rl/              <- github.com/pollen-robotics/microduck_rl  (uv sync)
  Drosophila_brain_model/    <- github.com/philshiu/Drosophila_brain_model (+ .venv)
  .venv-vision/              <- Python venv with flyvis
```

1. **Duck environment.** Clone `microduck_rl` and run `uv sync`. Download `alpha_walking.onnx` from the
   Hugging Face repo `pollen-robotics/microduck-policies` into `microduck_rl/policies/`. Install `ffmpeg`.
2. **Vision environment.** `python -m venv ~/flyduck/.venv-vision`, then
   `pip install -r requirements-vision.txt`, then `flyvis download-pretrained`.
3. **Brain environment.** Clone `Drosophila_brain_model`, create its venv, install its requirements,
   then `pip install -r requirements-brain.txt`.

### Data (downloaded, not redistributed here)

Put these in `bridge/data/`:

| File | Source |
|---|---|
| `Supplemental_file1_neuron_annotations.tsv` | [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) (`supplemental_files/`) |
| `column_assignment.csv.gz` | FlyWire Codex download, v783 (Matsliah et al. 2024) |
| `Connectivity_783.parquet` | copy from `Drosophila_brain_model/` |
| `flyvis_nodes.csv`, `fib25-fib19_v2.2.json` | `python tools/export_flyvis_inputs.py` (vision env) |

### Build and run

```bash
cd ~/flyduck/bridge
python build_bridge.py && python validate_bridge.py                 # any env with pandas+pyarrow
~/flyduck/.venv-vision/bin/python fly_eye.py --stimulus looming      # first run calibrates the eyes

cd ~/flyduck/microduck_rl
uv run python ~/flyduck/bridge/fly_loop.py --motion-test             # duck stands, escape moves it
uv run python ~/flyduck/bridge/fly_loop.py --scenario loom --rv 160 --fly-through \
    --eye-gain 2 --step-ms 20 --threshold 2 --escape-cmd -0.3 0.2 1.5 # one dodge, writes a video
uv run python ~/flyduck/bridge/dodge_trials.py                       # full statistics (~40 min)
```

## Files

| File | What it does |
|---|---|
| `build_bridge.py`, `validate_bridge.py` | FlyWire to flyvis neuron mapping and lattice alignment |
| `fly_eye.py` | offline: movies to flyvis to firing rates (+ calibration, direction check) |
| `brain_bridge.py`, `make_test_rates.py`, `explain_neuron.py`, `inputs_of.py` | offline brain runs and circuit analysis |
| `duck_eyes.py` | renders the duck's two fly-eye cameras |
| `fly_loop.py`, `eye_worker.py`, `brain_worker.py`, `ipc.py` | the closed loop (three Python environments talking over pipes) |
| `loom_sweep.py`, `dodge_trials.py` | experiments: loom-speed sweep and dodge statistics |
| `docs/development_log.md` | the full lab notebook: every result, bug and decision |

## Limitations

- Pollen's walking policy stands in for the ventral nerve cord, and the escape command is a fixed
  mapping from escape-neuron spikes to (vx, vy, yaw).
- The eye-to-brain rate conversion (gain, calibration) is a modelling choice, not measured biology.
- Dodge margins are small (1-4 cm) and tested on 3 trials per side at one speed (r/v 160 ms).
  Faster balls (r/v 40-80 ms) are detected but the duck cannot get out of the way in time.
- About 490 of flyvis's 721 columns per eye map onto FlyWire neurons. Seven flyvis cell types have
  no FlyWire match.

## Credits

This project builds on:
- **FlyWire connectome:** Dorkenwald et al. 2024, Schlegel et al. 2024 (Nature); column assignment
  Matsliah et al. 2024 (Nature).
- **flyvis:** Lappalainen et al. 2024 (Nature), TuragaLab.
- **Whole-brain LIF model:** Shiu et al. 2024 (Nature), philshiu/Drosophila_brain_model.
- **Microduck robot, simulation and walking policies:** Pollen Robotics (microduck_rl).

Please cite the original works if you use this. Code in this repository: MIT (see `LICENSE`).
