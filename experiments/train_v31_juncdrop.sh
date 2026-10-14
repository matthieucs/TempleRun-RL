#!/usr/bin/env bash
# v31 = v27/v30 recipe + TWO device-measured fixes (2026-07-05 evening):
# 1. JUNCTION LEVEL DROPOUT: the hook's neighbor-chain walk breaks through turns -> on-junction frames
#    stream next_path_level=next2_path_level=-1 (178/202 in v27_deviceeval), approach frames (lvl,-1).
#    Sim now reproduces it. THIS was the retrain curse: obs30 finetunes got a junction-frame input
#    spike never seen in training -> 1-frame swipe hesitation on device -> missed turns (v26/v27/v30);
#    v14's zero-init level cols were immune.
# 2. Late-queue gate CONSTANT 0.56 (CanTurn window end, velocity-INDEPENDENT: 205 device first-swipes,
#    fired max 0.55-0.56 in every band incl v207, missed min 0.59). Replaces the wrong lag-2 arrival
#    model (v30) and the never-binding 0.50 (v29... which was in fact nearly right; the real v27 killer
#    was the dropout above making the policy swipe on the SECOND junction frame ~0.6 > 0.56).
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
  --out runs/ppo_v31_juncdrop
echo "V31 DONE"
