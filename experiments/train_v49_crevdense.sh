#!/usr/bin/env bash
# v49 = WARM-START from the device champion v47 (holes mastered) + crevasse_apex_all (the step-down/
# high-ground apex model) + a CREVASSE-DENSE episode_profiles mix. v48 proved the crevasse fix works
# (device hit-rate 19%->15%) but the stumble-penalty gradient was WEAK (crevasses too sparse in standard
# training). This profile mix gives 30% crevasse-dense episodes (zone_after 100 + zone_every 12) so the
# policy gets enough exposure to LEARN the late-jump timing, + 15% narrow-dense / 10% hole-rhythm to
# also practice the other fragile mechanics that inflate the sim-vs-device absolute gap. Keeps the hole
# landing model + all v47 skills. LR 5e-5, keep BEST ckpt. Gate: eval_ckpt_torch then FRESH-phone eval
# vs v47 (crevasse hit-rate + overall median).
set -euo pipefail
G:/python312/python.exe scripts/train/train_fast.py \
  --init-from runs/ppo_v47_land_1703936_steps.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_apex_all":true,"episode_profiles":[{"weight":0.45,"overrides":{}},{"weight":0.30,"overrides":{"zone_after_distance":100.0,"zone_change_every_cells":12}},{"weight":0.15,"overrides":{"narrows_after_distance":300.0}},{"weight":0.10,"overrides":{"wide_gap_prob":0.7,"min_space_between_obstacles":4}}]}' \
  --out runs/ppo_v49_crevdense
echo "V49 DONE"
