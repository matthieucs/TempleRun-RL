#!/usr/bin/env bash
# v55 = the CREVASSE geometry fix, ISOLATED on v51. #1 median-killer (device forensic: crevasse deaths
# strike EARLIEST, median 610m). Decoded: step-UP crevasse (lvl2->3) at LOW speed (~v166, the FIRST
# crevasse ~587m), policy takes off cp0.7-0.9, apexes on the lvl-2 gap cell then DESCENDS onto the lvl-3
# raised cap (vel_y -33..-62) -> clips -> dt=4 Ledge. The sim was too LENIENT (evaluated the cap at the
# jump APEX so v51 CLEARED); the device cap sits ~0.8 cells LATER in the arc. NEW crevasse_cap_arc_offset
# shifts d so the cap falls in the descending arc; paired with crevasse_graze_worldy (arc_y vs cap-top 15,
# graze->stumble / deep->kill) so high-speed crevasses graze/clear and low-speed clip (device-faithful).
# Calibrated offset 0.8 -> v51 in-sim median 2293->1796 (moderate crevasse pressure, not all-or-nothing).
# Warm-start v51 (obs60, NO surgery) + ONLY the crevasse fix (isolated; NOT stacked on v54 to isolate the
# effect of this DYNAMICS change). CAUTION: dynamics changes are regression-prone (v52 turn_send_lat) -
# gate HARD: offline (does crevasse survival improve w/o collapsing standard?) then device. If it regresses,
# keep v51/v54 and the crevasse fix needs takeoff-VARIANCE modeling (offset is a sharp threshold in-sim).
set -euo pipefail
python scripts/train/train_fast.py \
  --init-from runs/ppo_v51_next2_best.zip \
  --timesteps 2000000 \
  --net-arch 256,256 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 --learning-rate 5e-5 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.15 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config '{"jump_exec_lat_frames":1,"jump_exec_lat_jitter":1,"hook_turn_buffer":true,"jump_drop_prob":0.1,"compound_prob":0.12,"strafe_exec_lat_frames":1,"narrow_lane_center":5.0,"narrow_lane_tol":2.0,"stepup_clip_band":0.25,"wide_gap_landing_model":true,"wide_gap_land_grace":0.35,"crevasse_graze_worldy":true,"crevasse_graze_band":10.0,"crevasse_cap_arc_offset":0.8}' \
  --out runs/ppo_v55_crevasse
echo "V55 DONE"
