#!/usr/bin/env bash
# v60 = the HIGH-GROUND DOUBLE-JUMP fix, isolated on v59. USER-observed (2026-07-08, watching v59 live) +
# device-decoded (captures/ab_v59_r1.jsonl ep3/ep4 dt=4 Ledge): on HIGH-GROUND the policy HOLDS jump, so
# the instant jump-1 lands a seamless jump-2 fires (device is_jumping never drops between them) whose arc
# descends into the next step-down crevasse / long-hole packed in the high-ground zone -> Ledge/Fall. Only
# high-ground (on flat low-ground the stray jump lands on solid ground). The airborne penalty (v58) MISSED
# it: in-sim jump-1's landing leaves a 1-frame is_jumping=False gap, so the landing-frame re-jump wasn't
# docked -> the policy learned it's free (it is NOT on device where the jumps overlap). FIX = jump_hold_
# penalty: dock a JUMP that follows a JUMP (a SLIDE after a SLIDE), keyed on the policy's OWN action stream
# (last_action) which is identical sim<->device -> the policy learns press-once-and-release, killing the
# seamless re-jump regardless of the airborne-state timing. A single deliberate jump is NEVER docked (one
# press fires the full 0.8s jump). Warm-start v59@1.5M (256-net obs60, crevasse fix ON), the ONLY new change
# = jump_hold_penalty; everything else = v59's cfg (isolated one-change, per the v48/v52 lesson). 0.12 =
# moderate (accumulates over a hold; a lone jump is free). Gate: eval_ckpt_torch (does jump-hold/double-jump
# drop w/o hurting distance?), then device A/B vs v59. MIND: v59 is now the base champion -> gate HARD on device.
set -euo pipefail
python scripts/train/train_fast.py \
  --init-from runs/ppo_v59_crevasse_1503232_steps.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --jump-hold-penalty 0.12 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_graze_worldy":true,"crevasse_cap_arc_offset":0.4,"crevasse_takeoff_std":0.2}' \
  --ckpt-every 250000 \
  --out runs/ppo_v60_jumphold
echo "V60 DONE"
