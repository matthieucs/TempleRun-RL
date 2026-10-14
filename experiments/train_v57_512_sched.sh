#!/usr/bin/env bash
# v57 = the SCALED train (all 3 USER levers): 512-net + challenge curriculum + LR SCHEDULE + longer.
#  - NET: 512x512 (surgery-widened from v56, function-preserving 0/500 mism). The task is now rich
#    (turns+crevasse+holes+narrows+jump->slide+jump->strafe+gauntlets); 256 showed capacity interference
#    (v54 lost jump->slide when it learned jump->narrow). Sim is faithful now, so capacity -> real skill.
#  - CURRICULUM: the episode_profiles that lifted v56's standard to 2930 (best of session; +16% vs v54).
#  - LR SCHEDULE: cosine 1.2e-4 -> 1e-5. Constant-high collapses a warm-start (v15-v22); constant-low (5e-5)
#    undertrains. Decay does both: adapt the new 512 capacity + curriculum EARLY, then lock in / avoid the
#    late warm-start drift. (tensor-LR in the CUDA graph; smoke-tested.)
#  - LONGER: 4M steps + keep-BEST checkpoint (warm-starts drift late -> eval ckpts, don't trust _final).
# Warm-start v56_512. Gate HARD: eval ckpts @1-4M (standard NOT collapsed + challenge improved), pick best,
# then SAME-SESSION device vs v51 champion. This is the "trained correctly, at capacity" attempt at 10k.
set -euo pipefail
python scripts/train/train_fast.py \
  --init-from runs/ppo_v56_512.zip \
  --timesteps 4000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 \
  --learning-rate 1.2e-4 --lr-schedule cosine --lr-final 1e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"jumpslide_prob":0.12,"jumpslide_after_distance":300,"jumpslide_gap_cells":1,"jumpnarrow_prob":0.12,"jumpnarrow_after_distance":300,"jumpnarrow_gap_cells":1,"jumpnarrow_strip_cells":2,"max_back_to_back_obstacles":6,"episode_profiles":[{"weight":45,"overrides":{}},{"weight":15,"overrides":{"obstacle_prob":0.9,"min_space_between_obstacles":2}},{"weight":12,"overrides":{"zone_change_every_cells":9,"obstacle_prob":0.55}},{"weight":12,"overrides":{"wide_gap_prob":0.9,"obstacle_prob":0.7,"min_space_between_obstacles":3}},{"weight":10,"overrides":{"compound_prob":0.4,"obstacle_prob":0.6,"min_space_between_obstacles":3}},{"weight":6,"overrides":{"narrows_after_distance":0.0,"obstacle_prob":0.7,"min_space_between_obstacles":3}}]}' \
  --ckpt-every 250000 \
  --out runs/ppo_v57_512sched
echo "V57 DONE"
