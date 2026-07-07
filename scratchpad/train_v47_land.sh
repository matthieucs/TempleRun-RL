#!/usr/bin/env bash
# v47 = v43 recipe (v36_256 warm-start + clip-band, 2M @5e-5) PLUS the device-calibrated WIDE-GAP
# LANDING MODEL (wide_gap_landing_model, grace 0.35). Closes the biggest sim-to-real gap: 2-cell
# HoleLongs (pt12/13/15/16) had NO landing check -> the sim let a short jump arc "clear" while the
# device clips the far wall (dt4 Ledge, 20% of device 2-cell-hole jumps, concentrated at low speed).
# Measured span = 0.0204*velocity - 0.94 cells (jump_arc.py, 1600 jumps / 8 captures). Equivalence
# GREEN 1820/1820 with the model on. Grace leaves the good jumper (v36) ~untouched, punishes jump-spam.
# Warm-start v36_256; LR 5e-5; keep BEST ckpt. Gate: eval_ckpt_torch @1M/2M, then SAME-SESSION phone
# eval vs v36 (does closing the hole gap finally let a 256-net NOT device-regress?).
# NOTE (Windows GPU box): G:/python312/python.exe, train_fast.py (GPU-resident PPO), bs512+cuda-graph.
set -euo pipefail
G:/python312/python.exe scripts/train_fast.py \
  --init-from runs/ppo_v36_256.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35}' \
  --out runs/ppo_v47_land
echo "V47 DONE"
