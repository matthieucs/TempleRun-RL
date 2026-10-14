#!/usr/bin/env bash
# v48 = v47 recipe, but the wide-gap landing model is now HOLES-ONLY (obst==OBST_HOLE). v47 device
# eval showed the hole fix TRANSFERRED (median ~4800 > v36 3112, best 6783) BUT the landing model was
# bleeding onto LevelChange crevasses (in _MUST_JUMP + wide_gap), rewarding EARLY crevasse jumps ->
# device-measured crevasse hits at takeoff cp 0.33 (v47) vs 0.63 (v36), 25% vs 5-16% hit-rate.
# Restricting to holes reverts crevasses to v36's regime. Same recipe otherwise; warm-start v36_256.
set -euo pipefail
G:/python312/python.exe scripts/train/train_fast.py \
  --init-from runs/ppo_v36_256.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_apex_all":true}' \
  --out runs/ppo_v48_land
echo "V48 DONE"
