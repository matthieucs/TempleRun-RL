#!/usr/bin/env bash
# v61 = the WINDOWED + GATED high-ground double-jump fix, from v59@1.5M (v60 regressed -> discard it as a
# base). v60's "consecutive JUMP" penalty was GAMED: the policy inserted 1-frame NoOp gaps (JMP,-,JMP) so
# no two presses were adjacent yet it still re-jumped on landing (ab_v60_r2 ep4 dt=4) AND it over-corrected
# into hole deaths (median 4220->2285). FIX (two parts): (1) WINDOW -> dock a re-jump within jump_hold_window
# (=6) frames of a prior jump; a jump lasts 8 frames so gaps can't dodge it. (2) GATE on must_jump -> NEVER
# dock a jump AT an obstacle (the legit jump + the closed-loop retry for dropped jumps) -> no jump-shyness.
# Only the purposeless post-obstacle re-jump (the lethal high-ground double-jump) is docked, and it can't be
# evaded. Warm-start v59@1.5M (crevasse fix ON), the ONLY new change = jump_hold_penalty (windowed+gated).
# 0.12. Gate: measure_jumphold (held/re-jump down?) + eval (distance NOT hurt, unlike v60), then device A/B.
set -euo pipefail
python scripts/train_fast.py \
  --init-from runs/ppo_v59_crevasse_1503232_steps.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --jump-hold-penalty 0.12 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_graze_worldy":true,"crevasse_cap_arc_offset":0.4,"crevasse_takeoff_std":0.2,"jump_hold_window":6}' \
  --ckpt-every 250000 \
  --out runs/ppo_v61_jumphold
echo "V61 DONE"
