#!/usr/bin/env bash
# v85 = THE MID->HIGH CREVASSE AT DIFFERENT SPEEDS (the USER's ask, verbatim: 'maybe we need to
# craft some special episodes at different speeds to make the model nail this'). v78's proven spine,
# ONE mechanism: crevasse-dense episodes at THREE speed bands + a wider curriculum start (5000 ->
# 9000) so caps are practised slow / mid / fast rather than only where a run happens to be.
#
# WHY CLEAN THIS TIME: the first attempt (v81) bundled this with a tight_pair dose that turned out
# ~6x too harsh AND sat on the v79 spine that had just device-regressed -- unattributable, killed.
# This arm changes ONLY the episode mix + curriculum range against the device-proven v78 recipe.
#
# NOTE it deliberately does NOT include min_clear_after_turn 3 (that is v83's isolated knob) so the
# two effects stay separable; combine only if both earn it.
set -euo pipefail
python scripts/train/train_fast.py \
  --init-from runs/ppo_v66_obs64_base.zip \
  --timesteps 20000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 \
  --learning-rate 8e-5 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --turn-buffer-bonus 0.5 --turn-explore-prob 0.5 --turn-explore-anneal-frac 0.5 \
  --curriculum-max-start-dist 9000 --turn-approach-buffer \
  --sim-config '{"post_land_gap_rejump_prob": 0.46, "turn_send_lat_frames": 0.7, "turn_offjunction_drop": true, "turn_drop_debounce_s": 0.3, "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": true, "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1, "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "stepup_clip_band": 0.25, "wide_gap_landing_model": true, "wide_gap_land_grace": 0.35, "crevasse_graze_worldy": true, "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2, "jumpslide_prob": 0.12, "jumpslide_after_distance": 300, "jumpslide_gap_cells": 1, "jumpnarrow_prob": 0.12, "jumpnarrow_after_distance": 300, "jumpnarrow_gap_cells": 1, "jumpnarrow_strip_cells": 2, "jumpgap_prob": 0.15, "jumpgap_after_distance": 300, "jumpgap_chain_max": 3, "turnobstacle_prob": 0.08, "slidegap_prob": 0.12, "slidegap_after_distance": 300, "max_back_to_back_obstacles": 6, "episode_profiles": [{"weight": 40, "overrides": {}}, {"weight": 13, "overrides": {"obstacle_prob": 0.9, "min_space_between_obstacles": 2}}, {"weight": 11, "overrides": {"zone_change_every_cells": 8, "obstacle_prob": 0.5, "zone_after_distance": 0.0}}, {"weight": 11, "overrides": {"zone_change_every_cells": 8, "obstacle_prob": 0.5, "zone_after_distance": 1500.0}}, {"weight": 11, "overrides": {"zone_change_every_cells": 8, "obstacle_prob": 0.5, "zone_after_distance": 3500.0}}, {"weight": 8, "overrides": {"wide_gap_prob": 0.9, "obstacle_prob": 0.7, "min_space_between_obstacles": 3}}, {"weight": 6, "overrides": {"compound_prob": 0.4, "obstacle_prob": 0.6, "min_space_between_obstacles": 3}}]}' \
  --ckpt-every 1000000 \
  --out runs/ppo_v85_crevspeed
echo "V85 DONE"
