#!/usr/bin/env bash
# v34 = v32 + cliff edge-riding death + BUFFERED-TURN CARRY FIX (turn_lane_carry_buffered 1.0:
# native swipe-queue landings are forgiving — v14 swiped at lane<=4.6 on 212/212 junctions, 0 landing
# deaths; carry 3.0 on buffered turns taught the fatal recenter-then-swipe hesitation = v32's 5/104
# junction deaths at off-center late onsets). v33 killed mid-train (same flaw baked in). Base v33 = v32 + CLIFF EDGE-RIDING death (device-measured: sustained |lane|>=6.3 on plvl>=3 for
# ~6 frames = Fall; the old high-ground check was DEAD CODE — the tilt glide clamps at the edge so
# |lane|>edge never fired; v31/v32 ride |lane| 7.0 at altitude and die on device). Also v32's base:
# v32 = v31 (junction dropout + 0.56 gate) + DEVICE-MEASURED JUMP RULES (2026-07-06, 3 fresh captures):
# - jump_refuse_strafing_prob 0.6 (sim default): Jump() eaten 56% while the lane is gliding (>0.8/frame)
#   vs 4% level. Policies must LEVEL before jumping — kills v31's strafe-into-crevasse habit (3/8 deaths).
# - gap_jump_lip_cp 0.2 (sim default): Jump() ignored past the wide-gap lip (all grounded-at-gap deaths).
#   The survivable takeoff is on the APPROACH cell — predicted_jump already cues it.
# - jump_drop_prob 0.1 (training knob): small unconditional byte-drop -> closed-loop retry habit.
# Fresh-phone baselines to beat: v14 median 2010 best 5462 | v31 1331/2442 (turns fixed: 1 death/91
# junctions) | v27 1266/3864.
set -euo pipefail
cd /home/matthieu/Documents/robotics/TempleRun-RL
.venv/bin/python scripts/train/train_sim.py \
  --init-from runs/ppo_v14_obs30_leveldir.zip \
  --timesteps 2000000 \
  --net-arch 128,128 --action-history-len 4 --frame-stack 1 \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1}' \
  --out runs/ppo_v34_carryfix
echo "V34 DONE"
