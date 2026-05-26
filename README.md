# M20 RL Training and Deploy

This repository is an M20-focused reinforcement learning workspace built on Isaac Lab. It contains:

- M20 locomotion tasks for flat, rough, and wheelie training
- RSL-RL training and playback scripts
- ONNX export utilities
- `sdk_deploy` for ROS 2 based sim-to-sim and sim-to-real deployment

The repo is tailored to the current M20 implementation in this workspace rather than the original multi-robot upstream layout.

## What Is Here

- `source/rl_training`
  M20 environment configs, rewards, commands, and deploy config export.
- `scripts/rsl_rl/train.py`
  Main training entrypoint.
- `scripts/rsl_rl/play.py`
  Checkpoint playback in Isaac Sim and export of `policy.onnx` / `policy.pt`.
- `scripts/tools/export_onnx_fast.py`
  Fast ONNX export from a `.pt` checkpoint without Isaac Sim.
- `sdk_deploy`
  ROS 2 workspace for `m20_sdk_deploy`, `drdds`, and MuJoCo ROS bridge.
- `simulate.sh`
  Starts the MuJoCo ROS 2 simulator side.
- `deploy.sh`
  Starts the M20 SDK deploy node.

## Requirements

- Ubuntu 22.04
- Isaac Sim / Isaac Lab installed and working
- Python environment with Isaac Lab and `rsl-rl-lib`
- ROS 2 Humble for `sdk_deploy`
- `mujoco` and `numpy < 2.0` for MuJoCo sim-to-sim

## Installation

Clone the repository outside your Isaac Lab checkout:

```bash
git clone https://github.com/admonakhov/m20_rl.git m20_rl
cd m20_rl
```

Install the extension package into the Python environment that already has Isaac Lab:

```bash
python -m pip install -e source/rl_training
```

Optional sanity check:

```bash
python scripts/tools/list_envs.py
```

## Available M20 Tasks

- `Flat-Deeprobotics-M20-v0`
- `Rough-Deeprobotics-M20-v0`
- `Wheelie-Deeprobotics-M20-v0`

The wheelie task is rear-wheel focused and uses its own reduced observation/action space and deploy config.

## Training

Examples:

```bash
# Flat terrain
python scripts/rsl_rl/train.py --task=Flat-Deeprobotics-M20-v0 --headless

# Rough terrain
python scripts/rsl_rl/train.py --task=Rough-Deeprobotics-M20-v0 --headless

# Wheelie
python scripts/rsl_rl/train.py --task=Wheelie-Deeprobotics-M20-v0 --headless
```

Logs are written under:

```text
logs/rsl_rl/<experiment_name>/<timestamp>/
```

During training, the repo now automatically saves:

- `params/env.yaml`
- `params/agent.yaml`
- `params/deploy.yaml`

That `deploy.yaml` is the runtime contract for `sdk_deploy`: joint order, action scales, stiffness, damping, default joint positions, observation layout, and command limits.

## Playback in Isaac Sim

Run a trained checkpoint in Isaac Sim:

```bash
python scripts/rsl_rl/play.py --task=Flat-Deeprobotics-M20-v0 --num_envs=10
```

Useful options:

- `--keyboard`
  Run a single robot and drive commands from the keyboard.
- `--load_run <run_folder>`
  Select a specific log directory.
- `--checkpoint <path-or-file>`
  Load an explicit checkpoint.
- `--video --video_length 200`
  Record playback video.

Keyboard mode is intended for a single robot:

```bash
python scripts/rsl_rl/play.py --task=Flat-Deeprobotics-M20-v0 --keyboard
```

## Exporting a Policy

### Option 1: export from `play.py`

When you run `scripts/rsl_rl/play.py`, it exports:

- `exported/policy.onnx`
- `exported/policy.pt`

inside the selected run directory.

### Option 2: fast export without Isaac Sim

```bash
python scripts/tools/export_onnx_fast.py \
    --checkpoint_path logs/rsl_rl/deeprobotics_m20_flat/<run>/model_XXXX.pt \
    --robot m20 \
    --output_path exported/m20_policy.onnx
```

This path is useful when you only need ONNX and do not want to launch Isaac Sim.

## Deploy Config Workflow

For deployment, you typically need both:

- `policy.onnx`
- matching `deploy.yaml`

Recommended manual flow:

1. Train a policy.
2. Export or obtain `policy.onnx`.
3. Copy the matching `params/deploy.yaml` into:

```text
sdk_deploy/src/M20_sdk_deploy/policy/deploy.yaml
```

4. Copy the ONNX model into:

```text
sdk_deploy/src/M20_sdk_deploy/policy/policy.onnx
```

`sdk_deploy` now reads `deploy.yaml` at runtime and adapts observation size, action size, active joints, stiffness, damping, default joint positions, and command scaling from that file.

## Sim-to-Sim with `sdk_deploy`

Install MuJoCo-side Python packages if needed:

```bash
python -m pip install "numpy<2.0" mujoco
```

Build the ROS 2 workspace:

```bash
cd sdk_deploy
source /opt/ros/humble/setup.bash
colcon build --packages-up-to m20_sdk_deploy --cmake-args -DBUILD_PLATFORM=x86
```

Important:

- Build `sdk_deploy` outside `conda` when possible.
- If `drdds` fails with `ModuleNotFoundError: No module named 'em'`, ROS is using your `conda` Python instead of `/usr/bin/python3`.
- A reliable fix is:

```bash
conda deactivate
unset PYTHONPATH
source /opt/ros/humble/setup.bash
cd sdk_deploy
colcon build --packages-up-to m20_sdk_deploy --cmake-args -DBUILD_PLATFORM=x86
```

### Run sim-to-sim

Open two terminals from the repo root.

Terminal 1:

```bash
./deploy.sh
```

Terminal 2:

```bash
./simulate.sh
```

What these scripts currently do:

- `deploy.sh`
  - `source /opt/ros/humble/setup.bash`
  - `source sdk_deploy/install/setup.bash`
  - `ros2 run m20_sdk_deploy rl_deploy`
- `simulate.sh`
  - `source sdk_deploy/install/setup.bash`
  - `python3 sdk_deploy/src/M20_sdk_deploy/interface/robot/simulation/mujoco_simulation_ros2.py`

Both scripts set `ROS_DOMAIN_ID=1`.

## Sim-to-Real Notes

The same `m20_sdk_deploy` package is also the deployment side for the real robot. In practice:

- build `sdk_deploy` for the target platform
- provide the correct `policy.onnx`
- provide the matching `deploy.yaml`
- run `ros2 run m20_sdk_deploy rl_deploy`

The exact networking, authorization, and hardware-side SDK enabling steps depend on your robot setup and are not fully automated by this repository.

## Common Commands

Train flat:

```bash
python scripts/rsl_rl/train.py --task=Flat-Deeprobotics-M20-v0 --headless
```

Train wheelie:

```bash
python scripts/rsl_rl/train.py --task=Wheelie-Deeprobotics-M20-v0 --headless
```

Play a trained run:

```bash
python scripts/rsl_rl/play.py --task=Flat-Deeprobotics-M20-v0 --load_run <run_folder>
```

Compare two runs:

```bash
python scripts/tools/compare_runs.py \
    logs/rsl_rl/deeprobotics_m20_flat/<run1> \
    logs/rsl_rl/deeprobotics_m20_flat/<run2>
```

## Troubleshooting

### `ament_cmake` not found during `colcon build`

ROS 2 environment is not sourced:

```bash
source /opt/ros/humble/setup.bash
```

### `ModuleNotFoundError: No module named 'em'`

`colcon` is picking up `conda` Python. Use system Python / ROS shell instead of `conda`.

### ONNX input mismatch like `Got: 57 Expected: 31`

`policy.onnx` and `deploy.yaml` do not match. Replace both files as a pair from the same training run.

### `sdk_deploy` build depends on `drdds`

If `m20_sdk_deploy` says `install/drdds/share/drdds/package.sh` is missing, `drdds` did not build successfully first.

## Notes on Current Implementation

- The current scripts use `scripts/rsl_rl/...`, not `scripts/reinforcement_learning/...`.
- `train.py` exports `deploy.yaml` automatically for supported manager-based RL environments.
- `play.py` exports ONNX into the selected run directory.
- `sdk_deploy` reads `deploy.yaml` dynamically and supports reduced-dimension policies such as the wheelie task.
