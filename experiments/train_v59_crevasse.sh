#!/usr/bin/env bash
# v59 = the CREVASSE HIGH-GROUND GRAZE fix, isolated on v51. The #1 median killer (crevasse deaths ~610m).
# Device forensic (captures/v51_crevasse_16z.jsonl, 2026-07-08): v51 grazes level-3 (high-ground, cap
# ground_height 15) crevasse caps 30% (n=10, arc_clear median +2.29), level-2 caps 0% -- CONFIRMS the
# documented device ~32%. The sim was OPTIMISTIC: with crevasse_graze_worldy alone the cap sits near v51's
# jump APEX (arc_y ~19.2 >> cap 15) -> clears 100%. v55 added a FIXED cap_arc_offset but that gave a SHARP
# all-or-nothing threshold (no clean partial rate) -> device-disproven. FIX (this session): offset moves the
# cap onto the DESCENDING arc (clearance ~+2) AND crevasse_takeoff_std adds the device's per-crossing
# takeoff-timing VARIANCE (measured pooled takeoff_cp std 0.277) so a FRACTION (~30%) grazes -- a clean
# partial rate, matching device. Both knobs calibrated in-sim via sweep_crevasse_variance.py to ~30% lvl-3
# graze / 0% lvl-2 / low kill. Warm-start v51 (256-net obs60), the ONLY change = the two crevasse knobs;
# everything else = v51's EXACT cfg (isolated, no dense profiles, per the v48/v49/v50/v52 one-change lesson).
# Gate: eval_ckpt_torch (does crevasse survival improve w/o hurting the rest?), then device A/B vs v51.
# MIND the v52/v55 lesson: v51 is a SHARP optimum, crevasse retrains have regressed before -> gate HARD on
# device; a modest 5e-5 finetune, keep the BEST checkpoint.
# CALIBRATION (filled from sweep_crevasse_variance.py): OFFSET=0.4  STD=0.2
set -euo pipefail
python scripts/train/train_fast.py \
  --init-from runs/ppo_v51_next2_best.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_graze_worldy":true,"crevasse_cap_arc_offset":0.4,"crevasse_takeoff_std":0.2}' \
  --ckpt-every 250000 \
  --out runs/ppo_v59_crevasse
echo "V59 DONE"
