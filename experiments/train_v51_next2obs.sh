#!/usr/bin/env bash
# v51 = the CHAINED-OBSTACLE fork. v47 surgery-widened to obs60 (+ next2_must_jump/next2_must_slide,
# zero-init = action-identical to v47) + 2M finetune @5e-5. ISOLATED per the v48/v49/v50 lesson
# (change ONE thing per retrain): the ONLY difference from v47 is the new 2-ahead obstacle obs — SAME
# device-true sim config v47 already handles (clip band + wide-gap landing model), NO crevasse/turn
# fixes, NO dense episode_profiles. The sim already MODELS the death (airborne into a slide-obstacle =
# _kill Burnt, sim_env.py:1562) so the new obs has real learning signal. v47 (obs58) STAYS the
# deployable champion on the old hook; this is a new obs60 lineage (needs hook 16z on device).
# Gate: eval_ckpt_torch @1M/2M (obs60 device-true cfg), then SAME-SESSION phone eval vs v47.
set -euo pipefail
python scripts/train/train_fast.py \
  --init-from runs/ppo_v47_obs60_next2.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35}' \
  --out runs/ppo_v51_next2
echo "V51 DONE"
