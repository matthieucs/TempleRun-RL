#!/usr/bin/env bash
# v26 = the v25 fixes applied: (1) DEVICE-MEASURED jump latency (probe_jump_exec 25/25: exactly 1 frame,
# not the guessed 2) + jitter 1 for the TCP obs-staleness variance (turn analysis: ~1 frame, variable) —
# v25's 2±1 over-aimed by a full frame -> jumped too early on device; (2) SINGLE-PHASE full task — the
# crevasse-only isolation atrophied device turns (6/8 missed-turn deaths); turns must stay practiced.
# Low LR 5e-5 (the warm-start-collapse fix, validated in v25c). Warm-start v14+obs30.
set -euo pipefail
cd /home/matthieu/Documents/robotics/TempleRun-RL
.venv/bin/python scripts/train/train_sim.py \
  --init-from runs/ppo_v14_obs30_leveldir.zip \
  --timesteps 2000000 \
  --net-arch 128,128 --action-history-len 4 --frame-stack 1 \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1}' \
  --out runs/ppo_v26_cal
echo "V26 DONE"
