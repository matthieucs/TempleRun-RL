#!/usr/bin/env bash
# v83 = THE POST-TURN JUMP FIX (USER-directed 2026-07-25). v78's EXACT recipe + ONE config value.
#
# USER (live, repeatedly): 'the policy tends to jump randomly after a turn. No reason involved.'
# MEASURED (scratchpad/wasted_jump.py, after fixing a broken turn detector): 61-70% of EVERY
# policy's WASTED jumps (arc lands short of the next must-jump = cleared nothing) happen within 3
# cells of a junction. The human hedges post-turn too -- MORE, 2.6/km vs 0.7/km -- but 0/65 of the
# human's are lethal while the policies' run ~7% lethal. So the hedge is not the sin; where it LANDS
# is.
#
# ROOT CAUSE (scratchpad/postturn_obstacles.py) -- P(must-jump obstacle) by cell offset after a
# junction, device vs sim:
#     DEVICE   +1: 4.3%  +2:  0.7%  +3: 21.7%  +4: 15.0%
#     SIM(=2)  +1: 4.7%  +2: 14.4%  +3: 17.4%  +4:  9.2%     <-- +2 is 20x TOO DENSE
#     SIM(=3)  +1: 5.2%  +2:  0.0%  +3: 19.8%  +4: 22.4%     <-- matches device
# The sim has been TRAINING a jump at post-turn cell +2. On device that cell is empty, so the jump
# is wasted -- and a 2.5-3.3-cell arc launched there lands at +3/+4, which is exactly where the
# device concentrates its post-turn obstacles. That is the user's 'random jump after a turn', and
# the sim was teaching it.
#
# min_clear_after_turn is an EXISTING knob present in both sims (no code change, no equivalence
# risk). ONE value, v78's proven spine, fully attributable.
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
  --sim-config '{"post_land_gap_rejump_prob": 0.46, "turn_send_lat_frames": 0.7, "turn_offjunction_drop": true, "turn_drop_debounce_s": 0.3, "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": true, "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1, "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "stepup_clip_band": 0.25, "wide_gap_landing_model": true, "wide_gap_land_grace": 0.35, "crevasse_graze_worldy": true, "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2, "jumpslide_prob": 0.12, "jumpslide_after_distance": 300, "jumpslide_gap_cells": 1, "jumpnarrow_prob": 0.12, "jumpnarrow_after_distance": 300, "jumpnarrow_gap_cells": 1, "jumpnarrow_strip_cells": 2, "jumpgap_prob": 0.15, "jumpgap_after_distance": 300, "jumpgap_chain_max": 3, "turnobstacle_prob": 0.08, "slidegap_prob": 0.12, "slidegap_after_distance": 300, "max_back_to_back_obstacles": 6, "episode_profiles": [{"weight": 45, "overrides": {}}, {"weight": 15, "overrides": {"obstacle_prob": 0.9, "min_space_between_obstacles": 2}}, {"weight": 12, "overrides": {"zone_change_every_cells": 9, "obstacle_prob": 0.55}}, {"weight": 12, "overrides": {"wide_gap_prob": 0.9, "obstacle_prob": 0.7, "min_space_between_obstacles": 3}}, {"weight": 10, "overrides": {"compound_prob": 0.4, "obstacle_prob": 0.6, "min_space_between_obstacles": 3}}, {"weight": 6, "overrides": {"narrows_after_distance": 0.0, "obstacle_prob": 0.7, "min_space_between_obstacles": 3}}], "min_clear_after_turn": 3}' \
  --ckpt-every 1000000 \
  --out runs/ppo_v83_postturn
echo "V83 DONE"
