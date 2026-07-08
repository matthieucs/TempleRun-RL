#!/usr/bin/env bash
# v63 = the HIGH-GROUND RHYTHM CHAIN fix, accumulated on v62@1.5M. Device forensic (ab_v62_r1 ep0 @794,
# ab_v60_r2 ep4 @1481): high-ground zones chain 2-cell long-holes every ~4 cells (hole x2 -> clear x2 ->
# hole x2 ...) — the documented multi-hole RHYTHM wall. v62's single jumpgap tail trains ONE repetition;
# the device demands a land->run->jump METRONOME over 3+ (the seamless held re-jump arcs into hole #2).
# FIX = jumpgap_chain_max 3 (both sims): the tail becomes [clear x2, hole x2] x n, n uniform 1..3 — same
# prob, deeper rhythm exposure. Warm-start v62@1.5M (crevasse + single-jumpgap skills retained), the ONLY
# change = chain_max 1->3. Equivalence GREEN required (jumpgap_chain track). Gate: eval no-regression
# (jumpgap cfg + crevasse cfg) then SAME-SESSION device A/B vs v62.
set -euo pipefail
python scripts/train_fast.py \
  --init-from runs/ppo_v62_jumpgap_1503232_steps.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_graze_worldy":true,"crevasse_cap_arc_offset":0.4,"crevasse_takeoff_std":0.2,"jumpgap_prob":0.15,"jumpgap_after_distance":300,"jumpgap_chain_max":3}' \
  --ckpt-every 250000 \
  --out runs/ppo_v63_rhythm
echo "V63 DONE"
