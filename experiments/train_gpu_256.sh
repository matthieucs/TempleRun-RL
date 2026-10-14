#!/usr/bin/env bash
# GPU FROM-SCRATCH 256-net train (run on the GPU machine). The narrows campaign showed a 128-net can't
# hold reach-center-narrows AND centered-crevasses (they interfere). A 256-net trained FROM SCRATCH on
# the full scenario stack (not warm-started -> no drift) is the clean shot at holding ALL skills at once.
# SB3 PPO auto-uses cuda when torch.cuda.is_available(). Longer run (6M) since it's from scratch + wide.
#
# SETUP on the GPU box:
#   git clone <this repo>; cd TempleRun-RL
#   python -m venv .venv && .venv/bin/pip install -U pip
#   .venv/bin/pip install "stable-baselines3[extra]" gymnasium "torch --index-url https://download.pytorch.org/whl/cu121"
#   .venv/bin/pip install -e .   # or add src/ to PYTHONPATH
#   .venv/bin/python -c "import torch; print('CUDA', torch.cuda.is_available())"   # must be True
#   bash experiments/train_gpu_256.sh
# ⚠️ HONEST NOTE ON THE GPU: this PPO trains a TINY 256x256 MLP over a PYTHON sim env. The bottleneck is
# CPU-bound ENV STEPPING (SubprocVecEnv), NOT the network — so the GPU gives only marginal speedup. The
# real lever is PARALLEL ENVS: set --num-envs to ~(CPU cores) on the GPU box (16 below; raise to 32+ if
# it has the cores). The GPU still auto-accelerates the net (SB3 device='auto'), just don't expect a big
# win from it alone. If the box has many cores, this is where the speedup comes from.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/train/train_sim.py \
  --timesteps 6000000 --num-envs 16 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --turn-linger 3 --ent-coef 0.02 --learning-rate 3e-4 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0}' \
  --out runs/ppo_gpu256
echo "GPU256 DONE"
# Bring runs/ppo_gpu256_*.zip back here, then: device eval via scripts/device/capture_calib.py runs/ppo_gpu256_best.zip
