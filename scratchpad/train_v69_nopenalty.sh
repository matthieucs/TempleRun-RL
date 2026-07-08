#!/usr/bin/env bash
# v65 = v64's anchored-mix recipe + turnobstacle_prob 0.08 (POST-TURN OBSTACLE, device-measured: the game
# places obstacles on the first post-junction cell in ~9% of turns — 72 onsets across 6 captures; the sim's
# min_clear_after_turn(2) never did -> post-turn deaths end the 4-7k runs, 7/15 of pt(-1/1/2) deaths follow
# a turn by <12 frames). The policy must learn the turn-exit INSTANT reaction (obs null through junctions ->
# the obstacle appears the frame the corner is crossed). ONE added knob on the v64 recipe.
# --- v64 rationale (recipe otherwise identical) ---
# v64 = the ANTI-DRIFT consolidation: v59@1.5M + the WAVE-3-VALIDATED anchored curriculum, with ALL wave-4/5
# scenario knobs in the base cfg. WHY (USER-observed + measured 2026-07-08): sequential finetunes ERODED the
# crevasse skill — lvl-3 graze v59 19% -> v62 26-29% -> v63 32% (back to v51 baseline), arc clearance +3.09
# -> +0.00. Each 2M finetune optimizes its new scenario while crevasse exposure in standard generation is too
# sparse to defend the timing (the USER's "mid-to-high hits the edge" = this regression). Verdict: the fix is
# TRAINING, not sim/reward — stop stacking drift-prone finetunes; ONE run holding all skills under pressure:
#  - base: v59's device-true cfg + jumpgap 0.15 + chain_max 3 (rhythm) + b2b 6 (wave-3 gauntlet fix)
#  - episode_profiles (v56-validated shape, 45% standard ANCHOR): dense-gauntlet / crevasse-field (frequent
#    mid<->high transitions = the graze gradient) / long-hole-rhythm (jumpgap chains fire often) /
#    compound-dense / narrow-heavy
#  - 4M steps, cosine 8e-5 -> 1e-5 (v57's schedule, warm-start-safe top)
# Base = v59@1.5M (BEST crevasse arc, +3.09), NOT v62/v63 (already eroded). Gate: offline 3-cfg eval vs
# v59/v62 + sim graze rate, then SAME-SESSION device A/B + analyze_crevasse_device (graze must stay ~19% or
# better AND the rhythm/jumpgap skill must hold).
set -euo pipefail
python scripts/train_fast.py \
  --init-from runs/ppo_v66_obs62_next3.zip \
  --timesteps 4000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 \
  --learning-rate 6e-5 --lr-schedule cosine --lr-final 1e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_graze_worldy":true,"crevasse_cap_arc_offset":0.4,"crevasse_takeoff_std":0.2,"jumpgap_prob":0.15,"jumpgap_after_distance":300,"jumpgap_chain_max":3,"turnobstacle_prob":0.08,"max_back_to_back_obstacles":6,"episode_profiles":[{"weight":45,"overrides":{}},{"weight":15,"overrides":{"obstacle_prob":0.9,"min_space_between_obstacles":2}},{"weight":12,"overrides":{"zone_change_every_cells":9,"obstacle_prob":0.55}},{"weight":12,"overrides":{"wide_gap_prob":0.9,"obstacle_prob":0.7,"min_space_between_obstacles":3}},{"weight":10,"overrides":{"compound_prob":0.4,"obstacle_prob":0.6,"min_space_between_obstacles":3}},{"weight":6,"overrides":{"narrows_after_distance":0.0,"obstacle_prob":0.7,"min_space_between_obstacles":3}}]}' \
  --ckpt-every 250000 \
  --out runs/ppo_v69_nopenalty
echo "V69 DONE"
