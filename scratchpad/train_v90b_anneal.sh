#!/usr/bin/env bash
# v90 = THE LANDING-ARC OBSERVATION (obs 64 -> 65). Aimed at the largest MEASURED death class.
#
# BEHAVIOURAL DEATH BUDGET (scripts/death_behaviour.py, 44 device deaths across 6 capture sets) --
# classified by WHAT THE POLICY DID, not by which hazard the run ended on:
#     COMPOUND TRAP     55%  airborne into the fatal hazard from a jump that WAS justified
#     LATE / NO ACTION  25%  grounded at a must-jump/slide hazard, action never came
#     LATERAL           16%  fell off the side
#     RANDOM-JUMP TRAP   5%  airborne from a jump nothing required
# => 60% of all deaths are "MY ARC LANDED ME SOMEWHERE FATAL". Confirmed by geometry: the
# takeoff->fatal distance for those deaths has median 2.6 cells (18/24 between 2.0 and 3.5), which IS
# the jump span (2.5-3.3 cells). And 84% of deaths had another hazard within 5 cells BEHIND the fatal
# one, 33/37 of them at gap 1-2 -- the user's long-reported "combinations" point, now quantified.
#
# THE GAP: the policy already receives next1/next2/next3_must_jump, but relating those to its own arc
# is velocity-dependent arithmetic it demonstrably does not perform (it dies exactly at the arc
# endpoint). `predicted_landing` is about TURNS; `predicted_jump` is crevasse-specific. NOTHING told
# it where a jump would put it down.
#
# THE FIX: landing_hazard -- one velocity-invariant scalar. span = 0.0204*v - 0.94 (device jump-arc
# fit); landing offset = cell_progress + span; emit 1.0 if a must-jump hazard sits at the landing
# cell, 0.0 if clear, 0.5 if beyond the 3-cell lookahead. Verified speed-adaptive: at v150 a hazard at
# +2 reads 1.0, at v200 the arc flies over +2 (0.0) and +3 becomes the danger. Host-derived from
# already-streamed fields => NO hook change and NO device work; obs parity CPU<->torch 1248/1248.
# Same trick that broke the turn wall (predicted_landing) and crevasse takeoff (predicted_jump).
#
# Trained in the CORRECTED world (device-read difficulty ramp + density-tuned compounds), from a
# surgery-widened base (runs/ppo_v66_obs65_land.zip, 0/400 action mismatches = behaviour preserved).
#
# JUDGE IT ON PER-ENCOUNTER DEATH RATE, NOT THE MEDIAN. 10k median = 0.645%/encounter, from 1.56%
# today (a 2.4x reduction); an 8-episode median has only 11% power to see even a 20% change.
# ⚠️ DEPLOY needs hook >= 17h (TURN_AWARE_LOOKAHEAD true). obs65 -- older hooks/policies are incompatible.
set -euo pipefail
python scripts/train_fast.py \
  --init-from runs/ppo_v90_landing_11001856_steps.zip \
  --timesteps 2000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 \
  --learning-rate 3.662e-05 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --turn-buffer-bonus 0.5 --turn-explore-prob 0.5 --turn-explore-anneal-frac 0.5 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"post_land_gap_rejump_prob": 0.46, "turn_send_lat_frames": 0.7, "turn_offjunction_drop": true, "turn_drop_debounce_s": 0.3, "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1, "hook_turn_buffer": true, "jump_drop_prob": 0.1, "compound_prob": 0.12, "strafe_exec_lat_frames": 1, "narrow_lane_center": 5.0, "narrow_lane_tol": 2.0, "stepup_clip_band": 0.25, "wide_gap_landing_model": true, "wide_gap_land_grace": 0.35, "crevasse_graze_worldy": true, "crevasse_cap_arc_offset": 0.4, "crevasse_takeoff_std": 0.2, "jumpslide_prob": 0.03, "jumpslide_after_distance": 300, "jumpslide_gap_cells": 1, "jumpnarrow_prob": 0.03, "jumpnarrow_after_distance": 300, "jumpnarrow_gap_cells": 1, "jumpnarrow_strip_cells": 2, "jumpgap_prob": 0.03, "jumpgap_after_distance": 300, "jumpgap_chain_max": 3, "turnobstacle_prob": 0.08, "slidegap_prob": 0.03, "slidegap_after_distance": 300, "max_back_to_back_obstacles": 2, "episode_profiles": [{"weight": 50, "overrides": {}}, {"weight": 14, "overrides": {"obstacle_prob": 0.9}}, {"weight": 12, "overrides": {"zone_change_every_cells": 9, "obstacle_prob": 0.55}}, {"weight": 12, "overrides": {"wide_gap_prob": 0.9, "obstacle_prob": 0.7}}, {"weight": 6, "overrides": {"compound_prob": 0.4, "obstacle_prob": 0.6}}, {"weight": 6, "overrides": {"narrows_after_distance": 0.0, "obstacle_prob": 0.7}}], "device_difficulty_ramp": true, "turn_aware_lookahead": true, "min_clear_after_turn": 3, "hole_landing_model": true, "hole_trap_apex_frac": 0.5, "hole_trap_kill_prob": 0.35}' \
  --ckpt-every 1000000 \
  --out runs/ppo_v90b_final
echo "V90 DONE"
