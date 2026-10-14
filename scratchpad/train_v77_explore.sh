#!/usr/bin/env bash
# v76 = v75's EXACT recipe (v66-obs64 base, 20M cosine 8e-5->5e-6, lat0.7 + drop + debounce +
# predicted_turn cue) + ONE new variable: --turn-buffer-bonus 0.5 (WAVE 13 Lever A).
#
# WHY: five arms (v71a/b/c/d, v72, v75) plateaued the turn dead-letter at ~1050-1200 isolated. Reading
# reward.py:124 revealed the cause is NOT "on-policy RL can't discover the pre-swipe" — it's that the
# dense shaping (reaction_bonus turn branch) REWARDS the REACTIVE junction swipe (the behavior that
# dead-letters under latency), while the latency-robust APPROACH-cell pre-swipe earned ZERO shaping.
# The policy correctly learned what it was paid to do. turn_buffer_bonus aligns the dense reward WITH
# survival: it credits a valid-direction pre-swipe on the approach cell (un-gameable: spam is eaten by
# debounce, a reactive junction swipe doesn't latch). Same shape as the jump/slide reaction_bonus that
# works. Everything else IDENTICAL to v75 => a clean A/B (v75 = bonus OFF plateau ~1050) at matched steps.
#
# CHEAP-VALIDATION PROTOCOL: same 20M cosine schedule so LR(step) == v75's at every step (only var =
# the bonus). EARLY-READABLE: eval ckpts at 4/6/8/10M under the isolated turn-cfg + preswipe_fraction.py;
# if v76 clearly beats v75's plateau by 10M -> let it run to 20M -> gate battery -> device A/B vs v74.
# If it does NOT move the pre-swipe fraction -> kill it; the mechanism is deeper than reward alignment.
set -euo pipefail
python scripts/train_fast.py \
  --init-from runs/ppo_v66_obs64_base.zip \
  --timesteps 20000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.05 \
  --learning-rate 8e-5 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --turn-buffer-bonus 0.5 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"post_land_gap_rejump_prob":0.46,"turn_send_lat_frames":0.7,"turn_offjunction_drop":true,"turn_drop_debounce_s":0.3,"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_graze_worldy":true,"crevasse_cap_arc_offset":0.4,"crevasse_takeoff_std":0.2,"jumpslide_prob":0.12,"jumpslide_after_distance":300,"jumpslide_gap_cells":1,"jumpnarrow_prob":0.12,"jumpnarrow_after_distance":300,"jumpnarrow_gap_cells":1,"jumpnarrow_strip_cells":2,"jumpgap_prob":0.15,"jumpgap_after_distance":300,"jumpgap_chain_max":3,"turnobstacle_prob":0.08,"slidegap_prob":0.12,"slidegap_after_distance":300,"max_back_to_back_obstacles":6,"episode_profiles":[{"weight":45,"overrides":{}},{"weight":15,"overrides":{"obstacle_prob":0.9,"min_space_between_obstacles":2}},{"weight":12,"overrides":{"zone_change_every_cells":9,"obstacle_prob":0.55}},{"weight":12,"overrides":{"wide_gap_prob":0.9,"obstacle_prob":0.7,"min_space_between_obstacles":3}},{"weight":10,"overrides":{"compound_prob":0.4,"obstacle_prob":0.6,"min_space_between_obstacles":3}},{"weight":6,"overrides":{"narrows_after_distance":0.0,"obstacle_prob":0.7,"min_space_between_obstacles":3}}]}' \
  --ckpt-every 1000000 \
  --out runs/ppo_v77_explore
echo "V76 DONE"
