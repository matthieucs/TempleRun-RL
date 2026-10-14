#!/usr/bin/env bash
# v82 = THE CLEAN ARM (USER-CHALLENGED 2026-07-25). v78's EXACT device-proven recipe + ONE knob.
#
# WHY: v79 stacked TWO wave-14 knobs (hole trap + 67.5% one-cell crevasses) and DEVICE-REGRESSED
# (median 2646 vs v78 4425, three sub-1000 dumps). The v80a/v80b arms were both built on that same
# suspect spine, so neither could isolate anything -- killed. This arm changes ONE thing against a
# spine that is device-proven, which is the only way to attribute a result.
#
# THE KNOB: tight_pair_prob 0.55 / min_cells 2. USER (live): 'the policy jumps randomly... a random
# jump is often the reason of death'. MEASURED (scratchpad/spurious_jump.py): the extra jump is NOT
# the sin -- the HUMAN takes 85 spurious jumps per 41km and only 1.2% are lethal, while v79 takes 12
# per 31km and 1-in-6 KILLS. The policy's jumps are BLIND, not merely extra. Root cause in the sim:
# min_space_between_obstacles(6) GUARANTEES a clear landing after any takeoff, so a blind jump is
# structurally FREE in-sim. Device puts 6.2% of consecutive must-jump pairs INSIDE the ~2.5-3.3-cell
# jump arc (n=2684; gaps 1:23 2:49 3:95 4:74 5:272) -- there a blind takeoff lands on the obstacle.
# 0.55 reproduces that 6.2% on THIS spine (0.65 was calibrated on v79's one-cell-crevasse spine).
#
# GATE: trap-stress + crevasse configs vs v78, then a device batch vs v78's 4425 baseline.
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
  --sim-config '{"post_land_gap_rejump_prob": 0.46, "turn_send_lat_frames": 0.7, "turn_offjunction_drop": true, "turn_drop_debounce_s": 0.3, "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": true, "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1, "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "stepup_clip_band": 0.25, "wide_gap_landing_model": true, "wide_gap_land_grace": 0.35, "crevasse_graze_worldy": true, "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2, "jumpslide_prob": 0.12, "jumpslide_after_distance": 300, "jumpslide_gap_cells": 1, "jumpnarrow_prob": 0.12, "jumpnarrow_after_distance": 300, "jumpnarrow_gap_cells": 1, "jumpnarrow_strip_cells": 2, "jumpgap_prob": 0.15, "jumpgap_after_distance": 300, "jumpgap_chain_max": 3, "turnobstacle_prob": 0.08, "slidegap_prob": 0.12, "slidegap_after_distance": 300, "max_back_to_back_obstacles": 6, "episode_profiles": [{"weight": 45, "overrides": {}}, {"weight": 15, "overrides": {"obstacle_prob": 0.9, "min_space_between_obstacles": 2}}, {"weight": 12, "overrides": {"zone_change_every_cells": 9, "obstacle_prob": 0.55}}, {"weight": 12, "overrides": {"wide_gap_prob": 0.9, "obstacle_prob": 0.7, "min_space_between_obstacles": 3}}, {"weight": 10, "overrides": {"compound_prob": 0.4, "obstacle_prob": 0.6, "min_space_between_obstacles": 3}}, {"weight": 6, "overrides": {"narrows_after_distance": 0.0, "obstacle_prob": 0.7, "min_space_between_obstacles": 3}}], "tight_pair_prob": 0.55, "tight_pair_min_cells": 2}' \
  --ckpt-every 1000000 \
  --out runs/ppo_v82_tightpair
echo "V82 DONE"
