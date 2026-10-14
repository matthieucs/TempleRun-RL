#!/usr/bin/env bash
# v117 = v111's EXACT recipe and EXACT world, with ONE difference: the obs carries
# `slide_landing_hazard` (obs 65 -> 66). Nothing else changes -- same _v108_cfg world, same LR/ent
# schedule, same penalties, same 10M budget -- so the result is attributable to the column alone.
#
# THE CUE, measured on 26,854 real device takeoffs with the SHIPPED encoder (not a hand analysis):
#     slide_landing_hazard   takeoffs   slide-deaths     rate
#         0.0                  26803          80        0.30%
#         1.0                     51           8       15.69%     <- 52.6x lift
# and takeoff cp at cue=1 is 0.50 vs 0.28 at cue=0, i.e. when the arc is about to come down on a
# must-slide hazard the policy took off LATE. That is the whole failure: you cannot slide in mid-air,
# so the counterfactual lives entirely at takeoff, and 89% of the victims DID press slide and died
# anyway. The jumps are 100% forced by a must_jump, so this is a timing error, not a decision error.
#
# WHY NO DRILL, against the first instinct (and it is measured, scratchpad/slh_exposure.py): the sim
# ALREADY fires this cue on 1.12% of takeoffs vs the device's 0.19% = it over-serves the situation
# 6x. A jumpslide drill profile (scratchpad/_v117_cfg.json, jumpslide 0.55 @ 9.2% of episodes) moved
# the rate 1.12% -> 1.06% = INERT on its own precondition, so it was dropped before spending steps.
# EXPOSURE IS NOT THE GAP -- the policy meets this situation constantly and still cannot see it.
# This is also why v116's jumpslide drill did nothing (-2.1%): it added practice for a situation
# already over-served, while carrying the airborne_slide_kill 0.24 softening that cost v115 -9.7%.
# v117 keeps v111's harsh (unsoftened) world deliberately.
#
# HONEST CEILING, corrected from the design note: over 1077 pooled device deaths the cue flags 3.0%
# one jump in advance -- NOT the ~33% the spec claimed (that came from a SIM rollout whose flame
# death share, 33%, is nothing like the device's 7%). On v111's own captures it is 3-of-8 slide-type
# deaths and 1-of-8 flagged, but n=8 has ~11% power and is a hint, not a number. So the expected
# effect is SMALL. eval_paired at 8 seeds resolves ~2.5%, which is the right instrument for it.
#
# KILL CRITERION, fixed in advance (this is what the horizon obs lacked): at 2M steps, flip the
# column on real rollout states and measure dP(JUMP). Reference scale -- landing_hazard in mature
# v111 = -0.01163; an eroded cue = ~-0.0005; the horizon obs never left -0.00005 against a +0.0030
# bar. If |dP| < 0.0005 at 2M, STOP; do not run it to 10M and do not spend phone time on it.
set -euo pipefail
python scripts/train_fast.py \
  --backend cpp \
  --init-from runs/ppo_v111_obs66_slh.zip \
  --timesteps 10000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 \
  --ent-coef 0.03 --ent-coef-final 0.005 --ent-schedule cosine \
  --learning-rate 1.2e-4 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --airborne-action-penalty 0.08 --strafe-penalty 0.05 --turn-buffer-bonus 0.5 \
  --turn-explore-prob 0.5 --turn-explore-anneal-frac 0.5 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer --stepdown-bonus-gate --landing-hazard-penalty 0.30 \
  --sim-config "$(cat scratchpad/_v121_cfg.json)" \
  --ckpt-every 2000000 \
  --out runs/ppo_v121_pjfix
echo "V121 DONE"
