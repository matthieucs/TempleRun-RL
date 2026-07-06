#!/usr/bin/env bash
# v30 = v29 recipe + VELOCITY-DEPENDENT arrival gate (turn_queue_lat_frames=2, sim_env default).
# v29 post-mortem: the constant cp<=0.50 gate NEVER BOUND (policies swipe p90 0.40 in-sim) -> zero
# timing gradient, swipe distribution identical to v27. Device truth (v27_deviceeval, 4/4 missed-turn
# deaths): the byte ARRIVES at cp + 2*vel*dt/60; past the corner (1.0) = miss. Safe swipe cp <=0.67
# @v100 but <=0.30 @v210 where a junction frame may not even exist -> approach-cell pre-swipe (latched,
# 7/7 pass @v210) is the only robust strategy. Curriculum starts up to 5000m ensure v210 turn practice.
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
  --out runs/ppo_v30_arrival
echo "V30 DONE"
