#!/usr/bin/env bash
# v25 attempt 3: GENTLE warm-start (LR 5e-5) on crevasse-only + jump latency. The collapse at LR 3e-4 (both
# ent 0.04 and 0.01) = warm-start overshooting v14's fragile peak. Low LR should adopt the new obs
# (predicted_jump pre-aim) without knocking off the peak. Watch DEATHS drop (median is low-speed-polluted).
set -euo pipefail
cd /home/matthieu/Documents/robotics/TempleRun-RL
LAT='"jump_exec_lat_frames":2,"jump_exec_lat_jitter":1'
.venv/bin/python scripts/train_sim.py \
  --init-from runs/ppo_v14_obs30_leveldir.zip \
  --timesteps 1500000 \
  --net-arch 128,128 --action-history-len 4 --frame-stack 1 \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --no-domain-rand \
  --sim-config "{\"turn_prob\":0.0,\"turn_prob_late\":0.0,\"obstacle_prob\":0.0,\"zone_change_every_cells\":6,\"curriculum_max_start_dist\":8000,\"curriculum_deep_start_frac\":0.6,\"curriculum_dist_power\":1.0,$LAT}" \
  --out runs/ppo_v25c_lowlr
echo "LOWLR DONE"
