#!/usr/bin/env bash
# v53 = the JUMP->SLIDE death fix, isolated on v51. Device forensic (v51 dt=3 @4991/3916): a slide
# obstacle immediately after a crevasse/long-hole traps the runner airborne (crevasse jump ~0.8s airtime)
# -> can't slide -> dt=3/5. The sim NEVER generated jump->slide (min_space) so v51's offline death mix
# had 0 dt=3/5 -> never learned it. NEW config-gated jumpslide_prob places a slide 1-2 clear cells after
# a wide-gap (bypasses min_space); both sims reproduce the death (v51 now dies dt=3/5 in-sim). Warm-start
# v51 (obs60, NO surgery - obs unchanged) + the ONLY change = jumpslide_prob (+ its spacing). The policy
# must learn to jump the gap EARLY (land before the slide) + NOT re-jump (the user-observed redundant jump
# extends the airborne window). Everything else = v51's exact cfg (isolated, no dense profiles, per the
# v48/v49/v50/v52 one-change lesson). jumpslide_prob 0.15 = moderate exposure (rare-mechanic learning w/o
# over-specialising the standard distribution). Gate: eval_ckpt_torch (does dt3/5 survival improve w/o
# hurting the rest?), then device A/B vs v51. MIND the v52 lesson: v51 is a sharp optimum, gate HARD on device.
set -euo pipefail
python scripts/train_fast.py \
  --init-from runs/ppo_v51_next2_best.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"jumpslide_prob":0.15,"jumpslide_after_distance":300,"jumpslide_gap_cells":1}' \
  --out runs/ppo_v53_jumpslide
echo "V53 DONE"
