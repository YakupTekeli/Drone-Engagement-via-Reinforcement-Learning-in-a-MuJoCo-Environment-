# Getting Started From Scratch

*Türkçe sürüm: [KURULUM.md](KURULUM.md)*

This document walks someone who has just cloned the repository through
every step needed to run the project on their own machine. The steps were
executed end to end in a clean virtual environment on 22 October 2026; the
expected output of each step is given next to it.

For what the project does, why decisions were made and where things stand,
the [README](../README.md), the [progress log](PROGRESS_LOG.md) and the
[ADRs](decisions/) are the primary sources. This document only answers "how
do I run it".

## 1. Requirements

| Requirement | Note |
|---|---|
| Operating system | All development and every measurement were done on macOS 15 (Apple Silicon M3). Linux is expected to work but was not measured; see §8 for the known differences. |
| Python | **3.13.x** is required (`requires-python >= 3.13, < 3.14`). The python.org installer is recommended; check with `python3.13 --version`. |
| Git | Only to clone the repository. The code itself does not need Git. |
| ffmpeg | Only to render the demo video (`brew install ffmpeg`). |
| Disk | Virtual environment ~1.3 GB. Training outputs (`runs/`) on top of that: visual dataset 1.9 GB, YOLO dataset 0.9 GB, ~20 MB per policy run. |
| Internet | Only for `pip` and a one-time download of the YOLO11n pretrained weights. Training and evaluation run offline. |

No GPU is required. Detector and policy training were measured on Apple MPS;
CPU works too, only slower.

## 2. Clone the repository

```bash
git clone https://github.com/YakupTekeli/Drone-Engagement-via-Reinforcement-Learning-in-a-MuJoCo-Environment-.git ebabil-mujoco
cd ebabil-mujoco
```

A path without spaces is easiest. Paths with spaces work as well; the only
difference is that macOS viewer commands go through the
`scripts/run_mjpython.sh` wrapper (§6).

## 3. Virtual environment and dependencies

The project runs only from the `.venv` in its own root. Nothing is installed
into the system Python and `sudo pip` is never used.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-build-isolation -e .
```

`requirements.lock` pins the core dependencies (MuJoCo 3.13, Gymnasium 1.3,
Stable-Baselines3 2.9, PyTorch 2.14), the detector chain (Ultralytics 8.4,
Pillow, OpenCV) and the development tools (pytest, ruff, mypy) together, so
no separate "extras" install is needed. The last line installs the project in
editable mode; `setuptools` is in the lock file, which is why
`--no-build-isolation` is safe.

Verify the install with one line:

```bash
.venv/bin/python -c "import ebabil, mujoco, torch; print('ok', mujoco.__version__, torch.__version__)"
# ok 3.13.0 2.14.0
```

Invoking everything as `.venv/bin/python ...` is the safest form; after
`source .venv/bin/activate` a plain `python` works as well.

## 4. Tests and quality gates

There are four canonical commands and all four must pass on a clean install:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
.venv/bin/mypy
```

Expected: `382 passed, 117 subtests passed`, "All checks passed!" and
"125 files already formatted" from ruff, "Success: no issues found in 61
source files" from mypy. The test suite is headless and needs neither a
graphics context nor any training output; it takes about one minute on an
M3.

## 5. First run (no training outputs needed)

The following commands use only the code in the repository and take seconds.

```bash
# Print the quadrotor model (mass, inertia, rotor positions)
.venv/bin/python scripts/inspect_model.py

# Headless hover with the classical cascaded controller; prints thrust/torque summary
.venv/bin/python scripts/run_hover_demo.py

# One episode in the legacy navigation environment (expected: termination_reason=goal)
.venv/bin/python scripts/smoke_navigation_env.py --mode clear

# Short smoke training of the PPO pipeline (~3 s, no quality claim)
.venv/bin/python scripts/train_ppo.py --smoke --device cpu --run-name ppo-smoke
```

The last command writes `final_model.zip`, `vecnormalize.pkl`,
`checkpoints/`, `evaluation/`, `logs/` and `metadata.json` under
`runs/ppo-smoke/`. Every training run uses this same layout.

## 6. Visualisation (MuJoCo viewer)

Training and batch evaluation are always headless; the viewer is only for
inspecting selected recordings. On macOS the MuJoCo viewer needs `mjpython`,
which the project launches through a wrapper:

```bash
scripts/run_mjpython.sh scripts/view_hover_demo.py --loop
scripts/run_mjpython.sh scripts/view_static_scene.py --mode single_blocker --seed 20260910
scripts/run_mjpython.sh scripts/view_target_camera.py --pose hover --camera onboard
```

To watch a policy replay, first record an `.npz` with
`record_vision_replay.py` (§7.3), then:

```bash
scripts/run_mjpython.sh scripts/view_policy_replay.py runs/<recording>.npz --loop
```

## 7. Training outputs: what is and is not in the repository

Training outputs under `runs/` are excluded from Git as a rule (`runs/*`,
`*.pt`, `*.zip`). As the one exception, **the approved YOLO11n detector chain
and the best PPO policy are committed**, so a fresh clone contains everything
needed for the demo and for evaluation. Datasets, intermediate checkpoints
and the other seeds' runs are not in the repository; §7.2 reproduces them.

### 7.1 Path A: use the trained models that ship with the clone

Model files that come with the clone (~7.8 MB in total):

| File | Size | SHA-256 (first 16) |
|---|---:|---|
| `runs/detector_audit_v1_20260929/approval_manifest.json` | 2.7 KB | `5b9d92e05c088253` |
| `runs/detector_audit_v1_20260929/audit_report.json` | 4.2 KB | `ba85938821a83635` |
| `runs/detector_quality_v1_20260929/quality_report.json` | 6.7 KB | `51f9bcbd89d517d0` |
| `runs/detector_quality_v1_20260929/seed_20260911/weights/last.pt` | 5.4 MB | `8a9b537d863cee9a` |
| `runs/detector_cone_d1a_seed20260912_20260929/final_model.zip` | 2.2 MB | `a9a25ebbd12a337d` |

Small record files come with them: detector training and quality metrics
(`seed_20260911/*.json`, `audit_consumed.json`), the policy run's
`metadata.json` (config, seed, package versions), `vecnormalize.pkl` (reward
normalisation for continuation training) and the final-protocol measurement
`evaluation_heldout_final_det_assist_farloss_governor.json`.

The approval manifest binds the other three detector files by relative path
and SHA-256; if any of them is moved or modified the loader stops with
"Onay SHA-256 eşleşmiyor" (approval hash mismatch). This is fail-closed by
design. `.gitattributes` disables line-ending conversion for these files so
the hashes also survive a Windows checkout. Do not edit the files by hand and
do not change the layout under `runs/`.

The four commands below run with the committed model right after cloning
and installing (all were executed with exactly these paths on 22 October
2026). The first three are headless.

**a) Environment and detector smoke** (check_env, random rollout, short
PPO, short evaluation; ~20 s). Confirms everything loads and the detector
sees the target:

```bash
.venv/bin/python scripts/smoke_detection_vision_env.py \
  runs/detector_audit_v1_20260929/approval_manifest.json \
  --scene-config src/ebabil/configs/environment/static_scene_d1a.toml \
  --output-dir runs/detection_vision_smoke
```

**b) Evaluate the policy.** Visual domain never seen in training
(`validation`), random obstacles and random start/target placement,
deterministic policy, terminal assist on; 40 episodes take ~5 min:

```bash
.venv/bin/python scripts/evaluate_vision_policy.py \
  runs/detector_cone_d1a_seed20260912_20260929/final_model.zip \
  runs/detector_audit_v1_20260929/approval_manifest.json \
  runs/eval_seed12_heldout.json \
  --scene-config src/ebabil/configs/environment/static_scene_d1a.toml \
  --scene-mode random_obstacles --endpoint-mode random_both \
  --episodes 40 --max-steps 1200 --env-version v2 \
  --visual-domain-split validation --deterministic --terminal-assist on
```

The script prints `success_rate`, `arrival_rate`, `collision_rate` and the
termination reasons; details go to the JSON. The author's measurement with
this exact command: success `0.875`, arrival `0.875`, collision `0.050`. For a
quick check pass `--episodes 4`; on 22 October 2026 that gave
`4/4 hover_success` with `0` collisions. `--terminal-assist off` measures what
the policy does without the assist (hover success `0`, ADR-025).

**c) Watch an episode in the MuJoCo viewer.** First record a `500 Hz` replay
(~20 s), then open the viewer:

```bash
.venv/bin/python scripts/record_vision_replay.py \
  runs/detector_cone_d1a_seed20260912_20260929/final_model.zip \
  runs/detector_audit_v1_20260929/approval_manifest.json \
  runs/replays/seed12_ep310004.npz \
  --seed 310004 --env-version v2 --terminal-assist on \
  --scene-config src/ebabil/configs/environment/static_scene_d1a.toml \
  --scene-mode random_obstacles --endpoint-mode random_both \
  --visual-domain-split development_ood
scripts/run_mjpython.sh scripts/view_policy_replay.py runs/replays/seed12_ep310004.npz --loop
```

`--seed 310004` with `development_ood` is the very episode shown in the
README demo (`hover_success` at step 389). Change the seed for another
episode; use `--visual-domain-split validation` for the dark held-out domain.

**d) Three-panel demo video** (drone camera with YOLO boxes, chase camera,
top-down map; needs ffmpeg, ~30 s):

```bash
.venv/bin/python scripts/render_demo_video.py \
  runs/detector_cone_d1a_seed20260912_20260929/final_model.zip \
  runs/detector_audit_v1_20260929/approval_manifest.json \
  runs/demo/bright_310004.mp4 \
  --episode-seed 310004 \
  --scene-config src/ebabil/configs/environment/static_scene_d1a.toml \
  --visual-domain-split development_ood
```

To continue training the committed policy (extra steps), see the `--resume`
note in §7.2 step 7; `vecnormalize.pkl` is in the repository for that
purpose.

### 7.2 Path B: reproduce everything from scratch

The full chain runs in the order below. Durations are M3 (MPS) measurements.

**1. Visual dataset (~7 min, 1.9 GB).** 6528 frames in the 40-obstacle D1
scene; train / validation / development_ood / audit splits are separated by
seed.

```bash
.venv/bin/python scripts/generate_vision_dataset.py runs/vision_dataset_d1 \
  --config src/ebabil/configs/training/vision_dataset_d1.toml \
  --scene-config src/ebabil/configs/environment/static_scene_d1.toml
```

**2. YOLO box labels (~1 min, 0.9 GB)** and a visual check on random frames:

```bash
.venv/bin/python scripts/export_yolo_dataset.py runs/vision_dataset_d1 runs/yolo_dataset_d1
.venv/bin/python scripts/review_yolo_labels.py runs/yolo_dataset_d1 --count 8 --output-dir runs/yolo_dataset_d1/review
```

**3. YOLO11n pretrained weights (once, needs internet).** Training starts
from the COCO checkpoint; the file comes from the Ultralytics `v8.4.0`
release assets and its SHA-256 must match the value in `detector.toml`:

```bash
mkdir -p runs/weights
curl -L -o runs/weights/yolo11n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
shasum -a 256 runs/weights/yolo11n.pt
# 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

Every later Ultralytics call is offline (the code sets `YOLO_OFFLINE=1`); no
account or login is needed.

**4. Detector quality run (~76 min per seed; the config asks for three).**

```bash
.venv/bin/python scripts/run_detector_quality.py runs/yolo_dataset_d1 runs/detector_quality
```

If one seed is judged sufficient, stop the run and close it with the seeds
that completed; the decision text is written into the report:

```bash
.venv/bin/python scripts/finalize_detector_quality.py runs/yolo_dataset_d1 runs/detector_quality \
  --seeds 20260911 --decision "Single-seed decision: ..."
```

**5. Audit and approval.** The audit split contains only ellipsoid obstacles
and is consumed once. With this dataset config the audit is expected to fail
its obstacle gates (REV-050); the detector is then approved by user decision,
restricted to the obstacle shapes seen in training:

```bash
.venv/bin/python scripts/run_detector_audit.py runs/yolo_dataset_d1 \
  runs/detector_quality/quality_report.json runs/detector_audit
# if the audit failed:
.venv/bin/python scripts/approve_detector_by_decision.py \
  runs/detector_quality/quality_report.json runs/detector_audit/audit_report.json \
  --finding REV-050 --geometries box cylinder capsule sphere \
  --decision "Approval restricted to training shapes: ..."
```

The manifest is written to `runs/detector_audit/approval_manifest.json`. The
`approval_manifest` field in
`src/ebabil/configs/training/ppo_detector_d1a.toml` must point at this path
(the default is the author's run name; update it to your directory name).

**6. Environment smoke** (the `smoke_detection_vision_env.py` command in
§7.1).

**7. Policy training (~3 hours per seed, 4 parallel environments).**

```bash
.venv/bin/python scripts/train_ppo.py \
  --config src/ebabil/configs/training/ppo_detector_d1a.toml \
  --seed 20260912 --run-name detector_cone_d1a_seed20260912
```

Progress is in `runs/<run-name>/logs/progress.csv` and TensorBoard
(`.venv/bin/tensorboard --logdir runs/<run-name>/logs`). Continuation
training uses `--resume` and `--resume-normalization`; in that case
`--total-timesteps` is the number of **additional** steps, not the total.

**8. Held-out evaluation (40 episodes, ~5 min).** Visual domain never seen in
training (`validation`), deterministic policy, terminal assist on:

```bash
.venv/bin/python scripts/evaluate_vision_policy.py \
  runs/detector_cone_d1a_seed20260912/final_model.zip \
  runs/detector_audit/approval_manifest.json \
  runs/detector_cone_d1a_seed20260912/evaluation_heldout.json \
  --scene-config src/ebabil/configs/environment/static_scene_d1a.toml \
  --scene-mode random_obstacles --endpoint-mode random_both \
  --episodes 40 --max-steps 1200 --env-version v2 \
  --visual-domain-split validation --deterministic --terminal-assist on
```

The output JSON contains `success_rate`, `arrival_rate`, `collision_rate`
and per-episode results. The author's three-seed measurement (30 September
2026):

| Seed | Hover success | Arrival | Collision |
|---|---:|---:|---:|
| 20260912 | 87.5% | 87.5% | 5.0% |
| 20260913 | 77.5% | 77.5% | 22.5% |
| 20260914 | 87.5% | 87.5% | 12.5% |
| Mean | 84.2% | 84.2% | 13.3% |

Seed-to-seed spread is large and training on MPS is not bit-reproducible;
your own run is expected to land in this range, not to reproduce these exact
numbers.

### 7.3 Replay and demo with your own policy

In commands c and d of §7.1, replace the checkpoint path with your own run's
`final_model.zip` and the manifest path with your own
`approval_manifest.json`; every other option stays the same.

## 8. Other platforms (Linux, CUDA, CPU only)

These are untested; the known differences are:

- **Device selection.** The detector device is hard-set to `mps` in two
  places: `src/ebabil/configs/training/detector.toml` (`[detector] device`)
  and `src/ebabil/configs/environment/vision_navigation_v4.toml`
  (`[detections] device`). Allowed values are `mps`, `cpu`, `cuda`. On a
  non-Apple machine set both to `cpu` or `cuda`. For PPO,
  `train_ppo.py --device auto|cpu|cuda` is enough.
- **Headless rendering.** On Linux without a display, set `MUJOCO_GL=egl`
  (or `osmesa`).
- **Viewer.** `mjpython` is macOS-only; on Linux run the viewer scripts
  directly with `.venv/bin/python`.
- **The approved detector bundle** is platform-independent; the `.pt`
  weights and the `.zip` policy load anywhere with the same package
  versions.

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `python3.13: command not found` | Python 3.13 is not installed or not on PATH. The python.org installer puts it at `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13`. |
| Viewer fails with `bad interpreter` | The project path contains a space; use `scripts/run_mjpython.sh` instead of `.venv/bin/mjpython`. |
| `Onay SHA-256 eşleşmiyor: ...` | Detector files under `runs/` were moved or modified. Keep the layout in §7.1; never edit those files by hand. |
| `Onay artefact kök dışına çıkıyor` | The quality and audit directories must sit under the same `runs/` root. |
| `detections.device mps/cpu/cuda olmalı` or MPS not found | Change the device in the two configs listed in §8. |
| Ultralytics tries to reach the network | It should not; the code sets `YOLO_OFFLINE=1` and a local `YOLO_CONFIG_DIR`. Only the download in §7.2 step 3 needs internet. |
| Training hangs at startup (0% CPU) | Observed once (30 September 2026); killing the process and relaunching the same command was enough. |
| Training with `--resume` runs longer than expected | With resume, `--total-timesteps` is the number of extra steps; to go from 600k to 1.2M pass `600000`. |

## 10. Project layout

```text
src/ebabil/
  assets/mjcf/        quadrotor.xml (MuJoCo model)
  configs/            vehicle, controller, environment, training, evaluation TOML files
  controllers/        cascaded classical flight controller (ADR-004, ADR-011)
  dynamics/           rotors, mixer, drag
  scenes/, sensors/   scene generation, camera, rangefinder (legacy)
  envs/               Gymnasium environments; detection_vision.py = EbabilVisionNavigation-v2
  perception/         dataset generation, YOLO export, detector training/quality/audit, box slot encoder
  missions/           hover state machine (ADR-017), terminal assist (ADR-025)
  rewards/            terminal hover reward (18 components)
  training/, evaluation/, experiments/   PPO runner, evaluation, replay
scripts/              command-line tools (every command in this document)
tests/                382 tests; headless
docs/                 design, plan, progress log, decisions/ (ADRs), reports/, media/
runs/                 training outputs; only the approved detector + policy are in Git, the rest is not
```

## 11. Reading order

1. [AGENTS.md](../AGENTS.md): working rules (venv, tests, coordinate and
   naming conventions, safety). Turkish.
2. [docs/PROGRESS_LOG.md](PROGRESS_LOG.md): the "Güncel durum" (current
   status) section and the newest entries at the top. Turkish.
3. [docs/decisions/](decisions/): ADR-011 (architecture), ADR-017 (hover
   state machine), ADR-023 (detector + policy), ADR-024 (motion
   constraint), ADR-025 (terminal assist). Turkish.
4. [docs/reports/2026-09-30-rl-sistemi-raporu.docx](reports/2026-09-30-rl-sistemi-raporu.docx):
   observation, action and reward details (Turkish);
   [system architecture diagram](reports/2026-10-01-sistem-mimarisi.png).

Most project documentation is written in Turkish; this guide and the code
comments' identifiers are the English entry points.

Simulation results are not a claim about real-flight safety (AGENTS.md §8).
