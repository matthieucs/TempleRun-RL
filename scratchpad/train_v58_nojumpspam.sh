#!/usr/bin/env bash
# v58 = the JUMP-SPAM fix, finetuned on v57. USER-observed (v57 6813m death) + forensic: the policy
# presses JUMP continuously -> on device each press QUEUES a re-jump on landing -> perpetual airborne ->
# the takeoff at the next hole is MISTIMED -> clip (dt=4 Ledge). ROOT: the sim IGNORES jump-while-airborne
# (line _apply_action `_jump_t<=0` guard) so spam is FREE in-sim; and the idle penalty checks the OBSTACLE
# context (must_jump/next_must_jump) not the AIRBORNE STATE, so it can't catch it. (USER: the penalty
# clearly works for STRAFE - because random strafe is self-punishing via lane/cliff deaths - but jump-spam
# has no in-sim consequence.) FIX = airborne_action_penalty: dock JUMP-while-jumping / SLIDE-while-sliding.
# 0.15 (strong; a single ground jump NEVER triggers it, only the reflexive re-press, so it's safe to be
# firm - and it accumulates over a spam sequence). Warm-start v57 (512-net) + keep the challenge curriculum
# + short cosine 8e-5->1e-5 over 2M (focused finetune to break the habit). Gate: eval (spam down? median
# not hurt?) then device (does the 6813-style mistimed-hole death go away?).
set -euo pipefail
python scripts/train_fast.py \
  --init-from runs/ppo_v57_512sched_best.zip \
  --timesteps 2000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 \
  --learning-rate 8e-5 --lr-schedule cosine --lr-final 1e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --airborne-action-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"jumpslide_prob":0.12,"jumpslide_after_distance":300,"jumpslide_gap_cells":1,"jumpnarrow_prob":0.12,"jumpnarrow_after_distance":300,"jumpnarrow_gap_cells":1,"jumpnarrow_strip_cells":2,"max_back_to_back_obstacles":6,"episode_profiles":[{"weight":45,"overrides":{}},{"weight":15,"overrides":{"obstacle_prob":0.9,"min_space_between_obstacles":2}},{"weight":12,"overrides":{"zone_change_every_cells":9,"obstacle_prob":0.55}},{"weight":12,"overrides":{"wide_gap_prob":0.9,"obstacle_prob":0.7,"min_space_between_obstacles":3}},{"weight":10,"overrides":{"compound_prob":0.4,"obstacle_prob":0.6,"min_space_between_obstacles":3}},{"weight":6,"overrides":{"narrows_after_distance":0.0,"obstacle_prob":0.7,"min_space_between_obstacles":3}}]}' \
  --ckpt-every 250000 \
  --out runs/ppo_v58_nojumpspam
echo "V58 DONE"
