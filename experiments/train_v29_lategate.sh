#!/usr/bin/env bash
# v29 = v28 recipe + the LATE-QUEUE MISS gate (turn_buffer_late_cp=0.50, sim_env default).
# v28 post-mortem: hook_turn_buffer WITHOUT a late gate made junction turns timing-free in-sim ->
# ZERO gradient to swipe early (v28@600k swipe-cp distribution identical to v27: med 0.30 / p90 0.43;
# the device tail p90 0.59 arrives past the corner = missed turn = "stops turning at speed").
# The gate: junction swipe queued past cp 0.50 = silent MISS -> run off the edge. Verified 2026-07-05:
# swipe>=0.20 -> 12/12 junctions pass; >=0.55 -> 0/12 (all die). Early swipes now the ONLY safe policy.
set -euo pipefail
cd /home/matthieu/Documents/robotics/TempleRun-RL
.venv/bin/python scripts/train/train_sim.py \
  --init-from runs/ppo_v14_obs30_leveldir.zip \
  --timesteps 2000000 \
  --net-arch 128,128 --action-history-len 4 --frame-stack 1 \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true}' \
  --out runs/ppo_v29_lategate
echo "V29 DONE"
