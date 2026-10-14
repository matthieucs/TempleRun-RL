#!/usr/bin/env bash
# v72 = v70@20M surgery-widened to obs64 (predicted_turn, zero col @35 = v70-exact) + 10M finetune
# with the turn latency BINDING (lat 0.7 + drop/debounce): the cue makes the junction deadline
# VISIBLE, the latency makes it BIND, the drop makes the approach pre-swipe explorable, the
# debounce keeps spam priced. WAVE-11 design: the dead-letter fix as an OBSERVATION (wave-10 proved
# pressure alone plateaus at +40% and a 20M from-scratch-mix run eroded crevasse skill — hence:
# init FROM v70 itself, SHORTER run, warm-start-safe LR top 5e-5, mix EXACTLY v70's).
# Gate: scripts/gate_battery.py (MANDATORY; retro-rejects v71c) -> cooled device A/B vs v70.
set -euo pipefail
python scripts/train_fast.py \
  --init-from runs/ppo_v70_obs64_pturn.zip \
  --timesteps 10000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 \
  --learning-rate 5e-5 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"turn_send_lat_frames":0.7,"turn_offjunction_drop":true,"turn_drop_debounce_s":0.3,"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_graze_worldy":true,"crevasse_cap_arc_offset":0.4,"crevasse_takeoff_std":0.2,"jumpslide_prob":0.12,"jumpslide_after_distance":300,"jumpslide_gap_cells":1,"jumpnarrow_prob":0.12,"jumpnarrow_after_distance":300,"jumpnarrow_gap_cells":1,"jumpnarrow_strip_cells":2,"jumpgap_prob":0.15,"jumpgap_after_distance":300,"jumpgap_chain_max":3,"turnobstacle_prob":0.08,"slidegap_prob":0.12,"slidegap_after_distance":300,"max_back_to_back_obstacles":6,"episode_profiles":[{"weight":45,"overrides":{}},{"weight":15,"overrides":{"obstacle_prob":0.9,"min_space_between_obstacles":2}},{"weight":12,"overrides":{"zone_change_every_cells":9,"obstacle_prob":0.55}},{"weight":12,"overrides":{"wide_gap_prob":0.9,"obstacle_prob":0.7,"min_space_between_obstacles":3}},{"weight":10,"overrides":{"compound_prob":0.4,"obstacle_prob":0.6,"min_space_between_obstacles":3}},{"weight":6,"overrides":{"narrows_after_distance":0.0,"obstacle_prob":0.7,"min_space_between_obstacles":3}}]}' \
  --ckpt-every 1000000 \
  --out runs/ppo_v72_pturn
echo "V72 DONE"
