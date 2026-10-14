#!/usr/bin/env bash
# v37 = v36 (deploy, record 7040m: compound + all rules) + NARROWS-AT-SPEED fix. v36 device dies on
# narrows @v200+ (reaches lane ~2, short of device [3,7] band, inner-edge fall). Two device-grounded
# sim changes (TRAINING ONLY; defaults keep the measured avg for scorecard/replay):
#   strafe_exec_lat_frames=1  (control latency on the tilt byte, was UNMODELED -> sim never reproduced it)
#   narrow_lane_center=5/tol=2 = band [3,7] (device STRICT case; teaches reach-CENTER, safe on all device
#   bands [1-3,7]). Reproduces v36's death 45/60 @v210 -> the retrain learns to strafe EARLIER + reach 5.
# Keep compound_prob 0.12 + all v36 skills; warm-start v36; LR 5e-5; keep BEST ckpt.
set -euo pipefail
cd /home/matthieu/Documents/robotics/TempleRun-RL
.venv/bin/python scripts/train/train_sim.py \
  --init-from runs/ppo_v36_compound_2000000_steps.zip \
  --timesteps 2000000 \
  --net-arch 128,128 --action-history-len 4 --frame-stack 1 \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0}' \
  --out runs/ppo_v37_narrowfix
echo "V37 DONE"
