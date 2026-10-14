#!/usr/bin/env bash
# v78 = GUIDED EXPLORATION (WAVE 13). The turn dead-letter is a HARD SAMPLING DEADLOCK, not a reward
# bug alone: measured P(turn|approach)=0.000000 for v74/v75 AND v76 (bonus only) AND v77 (ent 0.05) —
# the policy assigns ~0 probability to the approach-cell turn, so it's NEVER sampled, so turn_buffer_bonus
# never fires. Pure reward (v76) and pure entropy (v77) BOTH failed to lift it off zero (global entropy
# spends its budget among safe actions, never the collapsed approach-turn).
#
# FIX (still 100% RL — exploring-starts): --turn-explore-prob 0.5 FORCES the correct-direction pre-swipe
# on approach cells during rollout (venv.approach_turn_hint), so turn_buffer_bonus IS sampled and its
# positive advantage (survive + bonus) propagates -> pi(approach-turn) rises off 0. --anneal-frac 0.5
# linearly decays the forcing to 0 by 10M, so the last 10M is PURE on-policy RL where the policy must
# SUSTAIN the pre-swipe itself (proves it learned it, not that it's being forced). Reward does ALL the
# learning; injection only guarantees sampling. No sim/reward change vs v76 => equivalence unaffected.
#
# EARLY-READABLE (preswipe_fraction.py, leading indicator P(turn|approach)): with FORCED sampling this
# should lift off 0.000000 by ~1-2M. If it rises -> pure RL cracks the deadlock -> run to 20M -> gate
# battery -> device A/B vs v74. Watch the anneal region (~8-12M): PRE% must HOLD as forcing decays, else
# the policy only pre-swipes when forced (not learned). If P stays 0 even under forcing -> debug/DAPG.
set -euo pipefail
python scripts/train_fast.py \
  --init-from runs/ppo_v66_obs64_base.zip \
  --timesteps 20000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 \
  --learning-rate 8e-5 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --turn-buffer-bonus 0.5 --turn-explore-prob 0.5 --turn-explore-anneal-frac 0.5 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"post_land_gap_rejump_prob":0.46,"turn_send_lat_frames":0.7,"turn_offjunction_drop":true,"turn_drop_debounce_s":0.3,"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_graze_worldy":true,"crevasse_cap_arc_offset":0.4,"crevasse_takeoff_std":0.2,"jumpslide_prob":0.12,"jumpslide_after_distance":300,"jumpslide_gap_cells":1,"jumpnarrow_prob":0.12,"jumpnarrow_after_distance":300,"jumpnarrow_gap_cells":1,"jumpnarrow_strip_cells":2,"jumpgap_prob":0.15,"jumpgap_after_distance":300,"jumpgap_chain_max":3,"turnobstacle_prob":0.08,"slidegap_prob":0.12,"slidegap_after_distance":300,"max_back_to_back_obstacles":6,"episode_profiles":[{"weight":45,"overrides":{}},{"weight":15,"overrides":{"obstacle_prob":0.9,"min_space_between_obstacles":2}},{"weight":12,"overrides":{"zone_change_every_cells":9,"obstacle_prob":0.55}},{"weight":12,"overrides":{"wide_gap_prob":0.9,"obstacle_prob":0.7,"min_space_between_obstacles":3}},{"weight":10,"overrides":{"compound_prob":0.4,"obstacle_prob":0.6,"min_space_between_obstacles":3}},{"weight":6,"overrides":{"narrows_after_distance":0.0,"obstacle_prob":0.7,"min_space_between_obstacles":3}}]}' \
  --ckpt-every 1000000 \
  --out runs/ppo_v78_guided
echo "V76 DONE"
