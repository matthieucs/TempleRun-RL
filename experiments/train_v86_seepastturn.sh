#!/usr/bin/env bash
# v86_seepastturn = WAVE 15 PERCEPTION RUN. v78's device-proven spine + the turn-aware lookahead.
#
# THE FIX (device-probed 2026-07-26, hook 17g probe -> hook 17h): the hook's PathElement neighbour
# walk used ONE fixed heading offset for EVERY step, so it read NULL the moment the track turned.
# 72% of device frames on/approaching a junction streamed next/next2_path_level = -1 -- NOTHING could
# be seen past a corner. A heading-aware walk resolves the full 6-cell horizon on 100% of the frames
# where the old walk fell short (56.8% of all frames, mean +3.46 cells). Shipped in hook 17h and
# mirrored in BOTH sims as `turn_aware_lookahead`; junction blindness on device measured 72.1% -> 27.4%,
# the residual being genuine T-junctions where the walk deliberately REFUSES to guess.
#
# WHY IT MATTERS: 61-70% of every policy's WASTED jumps (arc lands short of the next must-jump =
# cleared nothing) sit within 3 cells of a junction, and they are ~7% lethal on device. The HUMAN,
# who can see the screen, hedges post-turn 2.4x MORE and dies from it 0/65 times. The blindness --
# not the hedging -- is the cause of the user's long-reported "random jump after a turn".
# CRUCIALLY the behaviour does NOT exist in the sim (0 wasted jumps in 74 km of rollout vs 6.5-7.3%
# of takeoffs on device), which is why ~20 models of training/generation tuning never touched it.
# This is a PERCEPTION fix; v78 and earlier were trained blind and cannot use the new information.
#
# BUNDLED (one-change rule relaxed by the user for coordinated fixes -- stated explicitly):
#   turn_aware_lookahead  = the fix itself
#   min_clear_after_turn 3 = device-measured post-turn clearance (device +2 is 0.7% occupied, the sim
#                            had 14.4%). Same corner region; verified NOT a difficulty distortion.
#
# ⚠️ DEPLOY: needs hook >= 17h with TURN_AWARE_LOOKAHEAD = true. Running this policy on 17f (or v78 on
# 17h) is an obs mismatch. Device-test against v78's stored baseline (median 4425), no alternating A/B.
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
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"post_land_gap_rejump_prob": 0.46, "turn_send_lat_frames": 0.7, "turn_offjunction_drop": true, "turn_drop_debounce_s": 0.3, "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": true, "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1, "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "stepup_clip_band": 0.25, "wide_gap_landing_model": true, "wide_gap_land_grace": 0.35, "crevasse_graze_worldy": true, "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2, "jumpslide_prob": 0.12, "jumpslide_after_distance": 300, "jumpslide_gap_cells": 1, "jumpnarrow_prob": 0.12, "jumpnarrow_after_distance": 300, "jumpnarrow_gap_cells": 1, "jumpnarrow_strip_cells": 2, "jumpgap_prob": 0.15, "jumpgap_after_distance": 300, "jumpgap_chain_max": 3, "turnobstacle_prob": 0.08, "slidegap_prob": 0.12, "slidegap_after_distance": 300, "max_back_to_back_obstacles": 6, "episode_profiles": [{"weight": 45, "overrides": {}}, {"weight": 15, "overrides": {"obstacle_prob": 0.9, "min_space_between_obstacles": 2}}, {"weight": 12, "overrides": {"zone_change_every_cells": 9, "obstacle_prob": 0.55}}, {"weight": 12, "overrides": {"wide_gap_prob": 0.9, "obstacle_prob": 0.7, "min_space_between_obstacles": 3}}, {"weight": 10, "overrides": {"compound_prob": 0.4, "obstacle_prob": 0.6, "min_space_between_obstacles": 3}}, {"weight": 6, "overrides": {"narrows_after_distance": 0.0, "obstacle_prob": 0.7, "min_space_between_obstacles": 3}}], "turn_aware_lookahead": true, "min_clear_after_turn": 3}' \
  --ckpt-every 1000000 \
  --out runs/ppo_v86_seepastturn
echo "V86_SEEPASTTURN DONE"
