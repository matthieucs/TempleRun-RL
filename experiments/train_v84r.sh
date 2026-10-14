#!/usr/bin/env bash
# RESUMED v84r from runs/ppo_v84_both_4001792_steps.zip at the point the cosine schedule had reached
# (LR 7.28e-5 -> 5e-6 over the remaining 16M). Restarting at 8e-5 would
# have undone the annealing. Original header follows.
# v84 = v83's post-turn fix + a DIFFICULTY-CALIBRATED tight_pair. v78 spine.
#
# CALIBRATION CORRECTION (my error, caught by dose-response): I first set tight_pair 0.55/0.65 by
# matching ONE statistic -- the device's 6.2% of must-jump pairs within the jump arc. That blew up
# TOTAL obstacle density ~6x. Measured on v78 (n=64, 8000-step cap):
#     tight_pair  0.0 -> median 6786 |  0.15 -> 3188 |  0.35 -> 1205 |  0.55 -> 772
#     v78 ACTUAL DEVICE median = 4425
# So the device-matching dose is ~0.08-0.10, not 0.55. v81/v82 were training against a sim ~6x
# harder than the real game and were killed.
#
# WHY KEEP IT AT ALL: with tight_pair 0 the sim gives v78 6786 vs 4425 on device -- the sim is ~1.5x
# TOO EASY, which is the chronic offline->device correlation failure this project keeps hitting. A
# 0.10 dose brings overall difficulty onto the device, and it is independently justified: the device
# really does place 19% of consecutive must-jump pairs closer than the sim's hard min_space floor.
#
# min_clear_after_turn 3 = the post-turn fix (device +2 is 0.7% occupied, the sim had 14.4%).
# Pairs with v83 (min_clear 3 alone) to isolate what the tight_pair dose adds.
set -euo pipefail
python scripts/train/train_fast.py \
  --init-from runs/ppo_v84_both_4001792_steps.zip \
  --timesteps 16000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 \
  --learning-rate 7.28e-5 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --turn-buffer-bonus 0.5 --turn-explore-prob 0.5 --turn-explore-anneal-frac 0.5 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"post_land_gap_rejump_prob": 0.46, "turn_send_lat_frames": 0.7, "turn_offjunction_drop": true, "turn_drop_debounce_s": 0.3, "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": true, "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1, "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "stepup_clip_band": 0.25, "wide_gap_landing_model": true, "wide_gap_land_grace": 0.35, "crevasse_graze_worldy": true, "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2, "jumpslide_prob": 0.12, "jumpslide_after_distance": 300, "jumpslide_gap_cells": 1, "jumpnarrow_prob": 0.12, "jumpnarrow_after_distance": 300, "jumpnarrow_gap_cells": 1, "jumpnarrow_strip_cells": 2, "jumpgap_prob": 0.15, "jumpgap_after_distance": 300, "jumpgap_chain_max": 3, "turnobstacle_prob": 0.08, "slidegap_prob": 0.12, "slidegap_after_distance": 300, "max_back_to_back_obstacles": 6, "episode_profiles": [{"weight": 45, "overrides": {}}, {"weight": 15, "overrides": {"obstacle_prob": 0.9, "min_space_between_obstacles": 2}}, {"weight": 12, "overrides": {"zone_change_every_cells": 9, "obstacle_prob": 0.55}}, {"weight": 12, "overrides": {"wide_gap_prob": 0.9, "obstacle_prob": 0.7, "min_space_between_obstacles": 3}}, {"weight": 10, "overrides": {"compound_prob": 0.4, "obstacle_prob": 0.6, "min_space_between_obstacles": 3}}, {"weight": 6, "overrides": {"narrows_after_distance": 0.0, "obstacle_prob": 0.7, "min_space_between_obstacles": 3}}], "min_clear_after_turn": 3, "tight_pair_prob": 0.1, "tight_pair_min_cells": 2}' \
  --ckpt-every 1000000 \
  --out runs/ppo_v84_both
echo "V84 DONE"
