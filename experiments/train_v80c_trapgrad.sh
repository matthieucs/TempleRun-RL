#!/usr/bin/env bash
# v80c_trapgrad = WAVE 14 parallel arm: hole_trap_kill_prob 0.35 -> 0.60 (gradient, not exposure).
# Spine = v79 (see train_v79_holetrap.sh).
# v79 = WAVE 14: THE AIRBORNE TRAP. v78's EXACT 20M consolidation recipe + ONE mechanism (two knobs,
# both device-measured 2026-07-25), nothing else changed.
#
# THE GAP (scripts/analysis/hole_landing_forensic.py, 1065 device hole crossings / 4 policies):
#   * 12 of v78/v74/v70's 15 device hole deaths happen with the runner ALREADY AIRBORNE, arriving at
#     the hole DESCENDING from an earlier jump (d = cells from takeoff > 1). Taking off on the
#     approach cell (d<=1) is FREE on device: 0 deaths in 858 crossings. Past the arc apex it is not:
#     every death entered with frac = d/span >= 0.51, every free crossing at frac <= 0.41.
#   * In the SIM this was structurally impossible to experience, for TWO reasons, both now fixed:
#     (a) PHYSICS: a 1-cell hole is not `wide_gap`, so the wide-gap arc-landing model skipped it and
#         the ONLY gate was binary `_jump_t > 0` -> an almost-expired arc crossed a hole for free.
#         -> `hole_landing_model` + `hole_trap_apex_frac` 0.5 + `hole_trap_kill_prob` 0.35.
#     (b) GEOMETRY: the sim made EVERY crevasse 2 cells; on device 67.5% are ONE cell (n=409). That
#         extra cell pushes the next obstacle beyond the ~2.5-3.3-cell jump arc, so the trap could
#         never form. -> `crevasse_single_cell_prob` 0.675.
#   Measured after the fix: d>1 exposure 1.6% of sim hole crossings vs 2.3% on device; v78 scored in
#   the corrected sim loses 9% median (5535->5045) with Fall deaths 10->14 = the trap now bites, and
#   it does NOT touch the d<=1 regime (frac<=0.41 < 0.5 can never trigger) so v78's hole skill is safe.
#
# EVERYTHING ELSE IS v78 VERBATIM — including the guided-exploration turn fix (turn_buffer_bonus +
# turn_explore_prob/anneal), which must be re-run, not inherited: this is a CONSOLIDATION from the
# v66-obs64 base, not a finetune (the v52/v63/v71c/v73 erosion lesson).
#
# GATE: scripts/validate/gate_battery.py vs v78 (all 4 configs) -> cooled device A/B. Watch that crevasse does
# NOT erode (the 1-cell form changes crevasse geometry) and that the turn skill survives
# (experiments/preswipe_fraction.py: PRE% must stay ~100%, P(turn|approach) ~1.0).
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
  --sim-config '{"crevasse_single_cell_prob": 0.675, "hole_landing_model": true, "hole_trap_apex_frac": 0.5, "hole_trap_kill_prob": 0.6, "post_land_gap_rejump_prob": 0.46, "turn_send_lat_frames": 0.7, "turn_offjunction_drop": true, "turn_drop_debounce_s": 0.3, "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": true, "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1, "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "stepup_clip_band": 0.25, "wide_gap_landing_model": true, "wide_gap_land_grace": 0.35, "crevasse_graze_worldy": true, "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2, "jumpslide_prob": 0.12, "jumpslide_after_distance": 300, "jumpslide_gap_cells": 1, "jumpnarrow_prob": 0.12, "jumpnarrow_after_distance": 300, "jumpnarrow_gap_cells": 1, "jumpnarrow_strip_cells": 2, "jumpgap_prob": 0.15, "jumpgap_after_distance": 300, "jumpgap_chain_max": 3, "turnobstacle_prob": 0.08, "slidegap_prob": 0.12, "slidegap_after_distance": 300, "max_back_to_back_obstacles": 6, "episode_profiles": [{"weight": 45, "overrides": {}}, {"weight": 15, "overrides": {"obstacle_prob": 0.9, "min_space_between_obstacles": 2}}, {"weight": 12, "overrides": {"zone_change_every_cells": 9, "obstacle_prob": 0.55}}, {"weight": 12, "overrides": {"wide_gap_prob": 0.9, "obstacle_prob": 0.7, "min_space_between_obstacles": 3}}, {"weight": 10, "overrides": {"compound_prob": 0.4, "obstacle_prob": 0.6, "min_space_between_obstacles": 3}}, {"weight": 6, "overrides": {"narrows_after_distance": 0.0, "obstacle_prob": 0.7, "min_space_between_obstacles": 3}}]}' \
  --ckpt-every 1000000 \
  --out runs/ppo_v80c_trapgrad
echo "V80C_TRAPGRAD DONE"
