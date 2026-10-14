#!/usr/bin/env bash
# v36 = v34 (the deploy: turns/crevasses/jumps/cliff/narrows all solid, fresh-phone median 2863)
# WARM-STARTED FROM ITSELF + the STACKED-obstacle blindspot (compound_prob 0.12): narrow(A)->barrier->
# narrow(B opposite), needs a MID-AIR/pre-swing A->B. v34 had ZERO exposure (sim never stacked obstacles).
# Warm-start from v34 (not v14) to preserve its full skillset; LR 5e-5, keep BEST ckpt (drift past ~1M).
# Offline gate: compound survival up vs v34 AND turns/crevasses/narrows not regressed, then device eval.
set -euo pipefail
cd /home/matthieu/Documents/robotics/TempleRun-RL
.venv/bin/python scripts/train/train_sim.py \
  --init-from runs/ppo_v34_carryfix_2000000_steps.zip \
  --timesteps 2000000 \
  --net-arch 128,128 --action-history-len 4 --frame-stack 1 \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12}' \
  --out runs/ppo_v36_compound
echo "V36 DONE"
