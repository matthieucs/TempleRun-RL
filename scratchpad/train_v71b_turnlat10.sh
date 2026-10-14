#!/usr/bin/env bash
# v71b = v70's EXACT recipe + turn_send_lat_frames 0.4 (DEVICE-TRUE turn-latency pressure) -- the turn
# DEAD-LETTER fix, arm B of the 2026-07-09 twin-arm design (docs/superpowers/specs/
# 2026-07-09-turn-deadletter-fix-design.md). Decoded root cause (turn_death_forensic.py on
# ab_v70_r2b ep0 @4733): the policy only decides a turn ON the junction; at v209 one control frame
# = ~0.47cp so a phase-unlucky first sighting puts the write past the game's consume window (the
# capture shows giR=ON, never consumed). The sim gates the DECISION cp (<=0.56) with NO latency ->
# late junction swipes are free in-sim -> no gradient toward the latency-immune APPROACH-CELL
# pre-swipe (turn_approach_buffer models it; the game's HandleDelayedTurning holds it natively).
# turn_send_lat gates proc_cp = decision_cp + v*lat*dt/cell -> junction-cell late swipes now MISS.
# v52 lesson: 0.4 as an ISOLATED finetune on sharp-optimum v51 regressed by half -- this arm rides
# the CONSOLIDATION regime (v70 held 5 skills / 20M without drift) at the MEASURED dose (the v52 value, now in the regime built to absorb it).
# Gate: offline vs v70 (standard device-true = parity+, lat-0.4 stress = clear win) -> device A/B.
set -euo pipefail
python scripts/train_fast.py \
  --init-from runs/ppo_v66_obs63_n2nl.zip \
  --timesteps 20000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 \
  --learning-rate 8e-5 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"turn_send_lat_frames":1.0,"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_graze_worldy":true,"crevasse_cap_arc_offset":0.4,"crevasse_takeoff_std":0.2,"jumpslide_prob":0.12,"jumpslide_after_distance":300,"jumpslide_gap_cells":1,"jumpnarrow_prob":0.12,"jumpnarrow_after_distance":300,"jumpnarrow_gap_cells":1,"jumpnarrow_strip_cells":2,"jumpgap_prob":0.15,"jumpgap_after_distance":300,"jumpgap_chain_max":3,"turnobstacle_prob":0.08,"slidegap_prob":0.12,"slidegap_after_distance":300,"max_back_to_back_obstacles":6,"episode_profiles":[{"weight":45,"overrides":{}},{"weight":15,"overrides":{"obstacle_prob":0.9,"min_space_between_obstacles":2}},{"weight":12,"overrides":{"zone_change_every_cells":9,"obstacle_prob":0.55}},{"weight":12,"overrides":{"wide_gap_prob":0.9,"obstacle_prob":0.7,"min_space_between_obstacles":3}},{"weight":10,"overrides":{"compound_prob":0.4,"obstacle_prob":0.6,"min_space_between_obstacles":3}},{"weight":6,"overrides":{"narrows_after_distance":0.0,"obstacle_prob":0.7,"min_space_between_obstacles":3}}]}' \
  --ckpt-every 1000000 \
  --out runs/ppo_v71b_turnlat10
echo "V71B DONE"

