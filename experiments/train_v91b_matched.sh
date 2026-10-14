#!/usr/bin/env bash
# v91b = EVENT-MATCHED CONTROL (0.065/0.18). v90's EXACT recipe + the HELD-JUMP -> LANDING RE-JUMP model at the DEVICE-MEASURED rate.
#
# THE USER'S #1 COMPLAINT, reported ~50 times: "on the high-ground it jumps for no reason just after
# having jumped over a crevasse". Root-caused 2026-07-26 with scripts/analysis/detect_patterns.py:
#   HELD_JUMP_SPAM   policy 10.67 /km on device   |   HUMAN 0 in 41 km (they TAP)
# In the sim a JUMP pressed while airborne is DISCARDED by the airborne guard, so holding the button
# is FREE and nothing ever teaches release. On DEVICE the still-held press fires a NEW jump the
# instant the runner lands (10% solid / 28-46% gap context) -- on high ground that second jump drops
# into the next crevasse or hole. A SIM-FIDELITY gap, not reward shaping: v58/v60/v61 all failed
# because they taxed a behaviour the world never punished, and were gamed by 1-frame gaps.
#
# DOSE: the raw DEVICE rate, 0.10 solid / 0.28 gap. NOT tuned toward an outcome, and here is why the
# 2026-07-26 note to "calibrate it down first" was withdrawn:
#   * that note rested on an n=64 sweep (6224 -> 2153 at the raw rate). At n=128 in the same world the
#     sweep is smooth, monotone and MILD: 0.00 -> 2908 | 0.02/0.06 -> 2843 | 0.04/0.11 -> 2782 |
#     0.07/0.20 -> 2634. The n=64 collapse was noise -- exactly the failure the project's own
#     eval-power correction warns about.
#   * the knob is BEHAVIOUR-CONDITIONAL, unlike a generation knob. Verified selective: a HOLDING
#     policy gets unwanted re-jumps, a TAPPING policy is completely unaffected. So "the sim median
#     drops" is not evidence of an over-harsh world -- it is the cost of a habit we want removed.
#   * it is a MEASURED property of the game, like the difficulty ramp, so it is not a free parameter.
# v91b runs the event-matched dose (0.065/0.18) as the control: the policy holds 16.5/km in-sim vs
# 10.7 on device (1.54x), so 0.065/0.18 reproduces the device EVENT rate rather than the device
# MECHANISM rate. The pair separates those two readings.
#
# SCHEDULE: 7M, cosine compressed into that budget. v90's curve was 1741@1M, 2459@3M, 2520@6M,
# 2682@10M -- essentially all learning by 3M and the rest LR annealing, so 20M buys nothing 7M does
# not. Consolidation from v90's OWN obs65 base, not a finetune off v90b (the v52/v63/v71c/v73
# erosion lesson), and the habit being unlearned here is deep enough that a low-LR finetune is the
# wrong tool.
#
# JUDGE IT ON: (1) HELD_JUMP_SPAM per km in a SIM rollout -- must fall from 16.5 toward the human's
# 0; (2) per-encounter death rate; NOT the 8-episode device median (11% power to see a 20% gain).
set -euo pipefail
python scripts/train/train_fast.py \
  --init-from runs/ppo_v66_obs65_land.zip \
  --timesteps 7000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 \
  --learning-rate 8e-5 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --turn-buffer-bonus 0.5 --turn-explore-prob 0.5 --turn-explore-anneal-frac 0.5 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer \
  --sim-config "$(cat experiments/_v91b_cfg.json)" \
  --ckpt-every 1000000 \
  --out runs/ppo_v91b_heldjump_matched
echo "V91B DONE"
