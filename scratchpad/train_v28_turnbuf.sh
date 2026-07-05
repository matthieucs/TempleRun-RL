#!/usr/bin/env bash
# v28 = v27 (calibrated apex + measured jump latency + LR 5e-5, single-phase full task) PLUS:
# - hook_turn_buffer + turn_approach_buffer: the sim finally models the device swipe-queue (hook 16o+)
#   where an EARLY swipe is SAFE (game buffers to the corner). v27's sim punished early swipes -> taught
#   the last-minute turns that missed at high speed (user-observed; 4 missed-junction deaths).
# - high-ground cliff edge = DEATH (plvl>=3, user-observed): teaches edge fear at altitude.
set -euo pipefail
cd /home/matthieu/Documents/robotics/TempleRun-RL
.venv/bin/python scripts/train_sim.py \
  --init-from runs/ppo_v14_obs30_leveldir.zip \
  --timesteps 2000000 \
  --net-arch 128,128 --action-history-len 4 --frame-stack 1 \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true}' \
  --out runs/ppo_v28_turnbuf
echo "V28 DONE"
