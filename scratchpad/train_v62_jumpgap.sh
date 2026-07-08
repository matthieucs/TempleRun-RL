#!/usr/bin/env bash
# v62 = the GAP->GAP death-model fix for the high-ground DOUBLE-JUMP, isolated on v59@1.5M. The proven
# method (chained-obs/jumpslide/jumpnarrow lineage): reward-shaping FAILED twice (v60 gamed by 1-frame
# gaps + over-corrected; v61's must_jump gate let the killer through) -> model the DEATH instead.
# Forensic (rejump_lethality.py 2026-07-08): device = ~17 landing-frame re-jumps/ep, ~2% lethal each ->
# 35% of episodes END within 15 frames of a re-jump (holes packed 2-3 cells apart on high ground). Sim
# (v59 cfg) = 0.8/ep, 0/47 lethal — min_space 6 guarantees a clear landing -> the blind re-jump is FREE
# in-sim -> no learning signal. FIX = `jumpgap_prob` compound (both sims, _pending_post_gap tail like
# jumpslide): after a wide-gap, 2 clear cells then a 2-cell HoleLong. A blind landing-frame re-jump lands
# INSIDE gap-2 (wide-gap landing model -> dt4 Ledge, the exact device signature) or arrives grounded past
# the lip (gap_jump_lip_cp refusal -> Fall); land -> run -> deliberate jump at the lip CLEARS. The policy
# learns don't-re-jump-blind AND the 2-gap rhythm from the DEATH signal, not a gameable penalty.
# Warm-start v59@1.5M (crevasse fix ON), the ONLY new change = jumpgap 0.15 (same exposure as jumpslide).
# NO jump_hold_penalty (shaping retired). Gate: rejump_lethality (sim lethality now >0? re-jumps drop?)
# + eval_ckpt_torch crevasse-cfg + standard (no regression), then device A/B vs v59. v59 = sharp optimum
# -> gate HARD on device; keep the BEST checkpoint (drift past ~1.5M).
set -euo pipefail
python scripts/train_fast.py \
  --init-from runs/ppo_v59_crevasse_1503232_steps.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_graze_worldy":true,"crevasse_cap_arc_offset":0.4,"crevasse_takeoff_std":0.2,"jumpgap_prob":0.15,"jumpgap_after_distance":300}' \
  --ckpt-every 250000 \
  --out runs/ppo_v62_jumpgap
echo "V62 DONE"
