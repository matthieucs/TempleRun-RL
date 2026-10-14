#!/usr/bin/env bash
# v88 = THE CORRECTED GENERATOR. First arm trained on the GAME'S OWN difficulty ramp.
#
# The ramp was READ OFF THE DEVICE (hook 17j: PathElement statics via klass+0xB8 from a live
# instance, sampled every ~2s over a 10,768 m run) -- an EXACT distance-keyed schedule, not a fit:
#     dist(m)     minSpaceObstacles  minSpaceTurns  doubleObstaclePct  narrows  flame/slide
#     0..250            6                12             0.00             no        no
#     250..500          5                10             0.00             no        no
#     500..1100         4                 8             0.05             no       YES
#     1100..1500        4                 6             0.05             no       YES
#     1500..2000        3                 5             0.10            YES       YES
#     2000..3250        3                 4             0.15            YES       YES
#     3250..4500        3                 3             0.17            YES       YES
#     4500+             2                 2             0.20            YES       YES
# The sim had held min_space at a CONSTANT 6 for the whole run, so past ~4.5 km the real game packs
# obstacles THREE TIMES tighter than anything any policy had ever trained on. sDoubleObstaclePercent
# (0 -> 0.20) was not modelled at all. sMaxBackToBackObstacles is 2 on device; recipes used 6.
#
# VALIDATION BEFORE TRAINING (all three parts of the sim-to-real gap, scripts/validate/sim_real_audit.py):
#   COVERAGE   17 device pair-types with ZERO sim coverage -> 0. Every "-> Crevasse" compound (the
#              user's reported death: jump an obstacle, land in the crevasse) now exists.
#   DENSITY    spacing histogram moved onto the device shape (arc-share 8.3% -> 14.3% vs device 10.8%).
#   DIFFICULTY v78's sim median 6786 -> 5200 against its MEASURED device median of 4425: the
#              sim-to-real gap closed 1.53x -> 1.18x with NO fitting. That is the honest test.
#
# Also carries the device-verified PERCEPTION fix (turn_aware_lookahead, hook 17h+: junction
# blindness 72.1% -> 27.4%) and the wave-14 airborne trap. Bundled deliberately -- the user relaxed
# the one-change rule for coordinated fixes, and a policy that can see past corners needs the world
# it is seeing to be right.
#
# ⚠️ DEPLOY on hook >= 17h with TURN_AWARE_LOOKAHEAD = true. Compare to v78's stored device median 4425.
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
  --sim-config '{"post_land_gap_rejump_prob": 0.46, "turn_send_lat_frames": 0.7, "turn_offjunction_drop": true, "turn_drop_debounce_s": 0.3, "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": true, "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1, "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "stepup_clip_band": 0.25, "wide_gap_landing_model": true, "wide_gap_land_grace": 0.35, "crevasse_graze_worldy": true, "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2, "jumpslide_prob": 0.12, "jumpslide_after_distance": 300, "jumpslide_gap_cells": 1, "jumpnarrow_prob": 0.12, "jumpnarrow_after_distance": 300, "jumpnarrow_gap_cells": 1, "jumpnarrow_strip_cells": 2, "jumpgap_prob": 0.15, "jumpgap_after_distance": 300, "jumpgap_chain_max": 3, "turnobstacle_prob": 0.08, "slidegap_prob": 0.12, "slidegap_after_distance": 300, "max_back_to_back_obstacles": 2, "episode_profiles": [{"weight": 50, "overrides": {}}, {"weight": 14, "overrides": {"obstacle_prob": 0.9}}, {"weight": 12, "overrides": {"zone_change_every_cells": 9, "obstacle_prob": 0.55}}, {"weight": 12, "overrides": {"wide_gap_prob": 0.9, "obstacle_prob": 0.7}}, {"weight": 6, "overrides": {"compound_prob": 0.4, "obstacle_prob": 0.6}}, {"weight": 6, "overrides": {"narrows_after_distance": 0.0, "obstacle_prob": 0.7}}], "device_difficulty_ramp": true, "turn_aware_lookahead": true, "min_clear_after_turn": 3, "hole_landing_model": true, "hole_trap_apex_frac": 0.5, "hole_trap_kill_prob": 0.35}' \
  --ckpt-every 1000000 \
  --out runs/ppo_v88_ramp
echo "V88 DONE"
