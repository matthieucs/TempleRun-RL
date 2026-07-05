#!/usr/bin/env bash
# v35 = v34 skills (surgical warm-start ppo_v34_obs32_look2, 0/300 equivalence) + NEXT2-OBSTACLE OBS
# (OBS_LEN 30->32: next2_must_jump/next2_must_slide from hook-16v 2-ahead walk, through-turn dropout).
# Target: the chained-jump blindness @v200 (jump spans ~2.75 cells; 1-cell lookahead lands blind into
# TreeJump — killed v14's 4633m and 6399m record runs). Full rule stack stays default (jump refusals,
# gap lip, cliff edge, dropout, 0.56 gate, buffered carry). LR 5e-5, single-phase, curriculum 5000.
set -euo pipefail
cd /home/matthieu/Documents/robotics/TempleRun-RL
.venv/bin/python scripts/train_sim.py \
  --init-from runs/ppo_v34_obs32_look2.zip \
  --timesteps 2000000 \
  --net-arch 128,128 --action-history-len 4 --frame-stack 1 \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1}' \
  --out runs/ppo_v35_look2
echo "V35 DONE"
