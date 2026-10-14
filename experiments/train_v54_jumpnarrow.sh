#!/usr/bin/env bash
# v54 = the JUMP->STRAFE death fix, ACCUMULATED on v53. USER-observed + device forensic (v51/v53 narrow
# deaths @5148/3281/2793): a side-narrow right after a jump-obstacle forces a strafe-into-lane during/after
# the jump; the policy commits too LATE (cp~0.9) -> the analog tilt can't reach the lane -> falls off. The
# sim never generated jump->narrow (min_space) so the policy never learned to strafe EARLY after a jump.
# NEW config-gated jumpnarrow_prob places a side-narrow strip after a wide-gap (both sims; v51 median drops
# 2514->1494 with it on = mechanic reproduced). Warm-start v53 (obs60, has the jump->slide fix) so v54
# ACCUMULATES both gap-closing scenario fixes (v53 showed scenario-adds DON'T regress the standard dist,
# unlike the v52 pressure retrain). ONLY new change vs v53 = jumpnarrow. Everything else = v53's cfg
# (jumpslide kept so the jump->slide skill isn't lost). Gate: eval under standard + jumpnarrow cfgs (does
# narrow-after-jump survival improve w/o hurting the rest?), then device edge-case forensic vs v51/v53.
set -euo pipefail
python scripts/train/train_fast.py \
  --init-from runs/ppo_v53_jumpslide_best.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"jumpslide_prob":0.15,"jumpslide_after_distance":300,"jumpslide_gap_cells":1,"jumpnarrow_prob":0.15,"jumpnarrow_after_distance":300,"jumpnarrow_gap_cells":1,"jumpnarrow_strip_cells":2}' \
  --out runs/ppo_v54_jumpnarrow
echo "V54 DONE"
