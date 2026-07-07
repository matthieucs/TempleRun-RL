#!/usr/bin/env bash
# v50 = WARM-START v47 (device champion) + ALL THREE device-grounded fidelity fixes:
#   wide_gap_landing_model (holes, device-verified via v47), crevasse_apex_all (device-verified via v48),
#   turn_send_lat_frames 0.4 (NEW — USER-observed "action fired but never launched"; the on-junction turn
#   byte arrives ~1 frame late and at high speed dead-letters past the 0.56 consume deadline -> off-track.
#   The latency-robust play is a PRE-SWIPE on the approach cell (turn_approach_buffer, immune); this knob
#   makes the sim penalise on-junction late swipes so the policy LEARNS to pre-swipe like v14 did).
# Profile mix gives the practice: 20% junction-dense (pre-swipe reps), 20% crevasse-dense, 15% narrow.
# lat 0.4 keeps v47 trainable (median 2252 vs 4189) while creating the pressure. LR 5e-5, keep BEST ckpt.
# Gate: eval_ckpt_torch, then FRESH-phone eval vs v47 (junction/off death rate + median).
set -euo pipefail
G:/python312/python.exe scripts/train_fast.py \
  --init-from runs/ppo_v47_land_1703936_steps.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_apex_all":true,"turn_send_lat_frames":0.4,"episode_profiles":[{"weight":0.45,"overrides":{}},{"weight":0.20,"overrides":{"turn_prob":0.32,"turn_prob_late":0.28}},{"weight":0.20,"overrides":{"zone_after_distance":100.0,"zone_change_every_cells":12}},{"weight":0.15,"overrides":{"narrows_after_distance":300.0}}]}' \
  --out runs/ppo_v50_turnlat
echo "V50 DONE"
