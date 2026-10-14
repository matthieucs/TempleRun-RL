#!/usr/bin/env bash
# v25 Phase B: integrate. Warm-start the CREVASSE-MASTER (v25c 500k: crevasse-under-latency hiMed 2244 vs
# v14's 130) on the FULL task + jump latency, LOW LR (no collapse) -> restore turns/narrows (eroded to
# ~1146 by the crevasse-only isolation) while KEEPING the crevasse pre-aim skill. Mirrors narrow-only->v7.
set -euo pipefail
cd /home/matthieu/Documents/robotics/TempleRun-RL
.venv/bin/python scripts/train/train_sim.py \
  --init-from runs/ppo_v25c_lowlr_500000_steps.zip \
  --timesteps 2000000 \
  --net-arch 128,128 --action-history-len 4 --frame-stack 1 \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 \
  --sim-config '{"jump_exec_lat_frames":2,"jump_exec_lat_jitter":1}' \
  --out runs/ppo_v25d_full
echo "PHASE B DONE"
