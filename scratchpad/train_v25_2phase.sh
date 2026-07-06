#!/usr/bin/env bash
# v25 = 2-phase crevasse-focused training on obs30 (direction + predicted_jump), warm-start v14.
# KEY FIX vs the collapsed 1st attempt: (1) jump_exec_lat_frames=2 jitter=1 makes the sim's crevasses
# device-faithfully HARD (v14 hi_med 2246->130; a pre-aiming oracle survives) -> real learning signal for
# predicted_jump; (2) ent 0.04->0.01 stops the warm-start collapse. Mirrors narrow-only->v7 isolation.
set -euo pipefail
cd /home/matthieu/Documents/robotics/TempleRun-RL
LAT='"jump_exec_lat_frames":2,"jump_exec_lat_jitter":1'

# ---- Phase A: CREVASSE-ONLY isolation + jump latency (learn robust PRE-AIM via predicted_jump) ----
.venv/bin/python scripts/train_sim.py \
  --init-from runs/ppo_v14_obs30_leveldir.zip \
  --timesteps 1500000 \
  --net-arch 128,128 --action-history-len 4 --frame-stack 1 \
  --turn-linger 3 --ent-coef 0.01 \
  --no-domain-rand \
  --sim-config "{\"turn_prob\":0.0,\"turn_prob_late\":0.0,\"obstacle_prob\":0.0,\"zone_change_every_cells\":6,\"curriculum_max_start_dist\":8000,\"curriculum_deep_start_frac\":0.6,\"curriculum_dist_power\":1.0,$LAT}" \
  --out runs/ppo_v25a_crevonly
echo "PHASE A DONE"
