#!/usr/bin/env bash
# v56 = the CHALLENGE-CURRICULUM retrain (USER insight: the mechanics are MODELLED but the policy TRAINED
# on the sparse standard distribution — obstacles spaced min_space 6, and max_back_to_back_obstacles=2
# CAPS chains at 2, so the sim literally never made a GAUNTLET. A hard device run is a continuous sequence
# the policy never trained on). Raise max_back_to_back to 6 (harmless to standard: min_space 6 keeps it
# sparse; the cap only bites when a profile lowers min_space) + an episode_profiles MIX (torch-only, sampled
# per reset): 45% standard / 15% dense-gauntlet / 12% crevasse-field / 12% long-hole-rhythm / 10% compound-
# dense / 6% narrow-heavy. Compounds (jumpslide/jumpnarrow) ON in the base so they appear everywhere.
# Smoke-test: v54 median 2531 (std) -> 1719 (challenge mix) = the gauntlets bite. Warm-start v54 (most
# well-rounded, has both compounds) so v56 ACCUMULATES + adds gauntlet-robustness. 45% standard weight
# guards the base distribution (the v48-v52 over-specialisation was too many changes + a dense grind w/o a
# standard anchor). Gate HARD: offline (standard NOT collapsed? challenge-cfg improved?) then device.
set -euo pipefail
python scripts/train_fast.py \
  --init-from runs/ppo_v54_jumpnarrow_best.zip \
  --timesteps 2500000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"jumpslide_prob":0.12,"jumpslide_after_distance":300,"jumpslide_gap_cells":1,"jumpnarrow_prob":0.12,"jumpnarrow_after_distance":300,"jumpnarrow_gap_cells":1,"jumpnarrow_strip_cells":2,"max_back_to_back_obstacles":6,"episode_profiles":[{"weight":45,"overrides":{}},{"weight":15,"overrides":{"obstacle_prob":0.9,"min_space_between_obstacles":2}},{"weight":12,"overrides":{"zone_change_every_cells":9,"obstacle_prob":0.55}},{"weight":12,"overrides":{"wide_gap_prob":0.9,"obstacle_prob":0.7,"min_space_between_obstacles":3}},{"weight":10,"overrides":{"compound_prob":0.4,"obstacle_prob":0.6,"min_space_between_obstacles":3}},{"weight":6,"overrides":{"narrows_after_distance":0.0,"obstacle_prob":0.7,"min_space_between_obstacles":3}}]}' \
  --out runs/ppo_v56_challenge
echo "V56 DONE"
