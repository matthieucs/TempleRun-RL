#!/usr/bin/env bash
# v52 = the TURN DEAD-LETTER fix, isolated on v51. v51's dominant device killer is the on-junction turn
# that dead-letters (byte arrives ~1 frame late -> processes past the 0.56 CanTurn deadline at speed ->
# runs off-track, pt=-1, 4/8 v51 device deaths at 3400-8763m). Warm-start v51 (obs60, NO surgery -
# turn_send_lat is a sim knob, not an obs change) + the ONLY change = turn_send_lat_frames=0.4 (models the
# late byte -> pressures the policy to PRE-SWIPE on the approach cell via turn_approach_buffer, which fires
# centered and is latency-IMMUNE). 0.4 = the tested value (mild, trainable; 1.0 crushes a cold base - but
# v51 is a strong warm-start). Everything else = v51's exact cfg (isolated, no dense profiles, per the
# v48/v49/v50 one-change lesson). Parity GREEN at slat=0.4 on junctions. Gate: eval_ckpt_torch, then
# device A/B vs v51 (does closing the turn gap push the 6668-8763m runs toward a 10k median?).
set -euo pipefail
python scripts/train/train_fast.py \
  --init-from runs/ppo_v51_next2_best.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"turn_send_lat_frames":0.4}' \
  --out runs/ppo_v52_turnlat
echo "V52 DONE"
