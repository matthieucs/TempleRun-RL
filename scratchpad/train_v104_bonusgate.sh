#!/usr/bin/env bash
# v104 = v103 + the reward AMPLIFIER (--stepdown-bonus-gate). v103 proved the world fix alone
# moved P(jump|step-down to low) 98.6%% -> 98.5%% = NOTHING, because reaction_bonus still paid 0.1
# for the jump while the sim now made it cost 0 deaths. Removing the death removed the REQUIREMENT;
# this removes the INCENTIVE. Same "weak signal needs an amplifier" shape as v91 -> v92.
# ORIGINAL v103 HEADER FOLLOWS.
# v103 = v102's recipe + TWO changes, both from the user's live device reports on 2026-07-27, and the
# FIRST run trained on the C++ backend (--backend cpp, 26,192 steps/s vs the torch sim's 3,705 at the
# same 256x16 geometry = 7.07x; 10M in ~6.4 min instead of ~45).
#
# USER (watching v102): "It does not strafe randomly anymore. It does not jump dumbly onto the high
# ground. ... it struggles to handle the edge case where you have 3 obstacles one after the other."
# and: "The problem is also the mid to low transition when you have an obstacle right ahead. The
# policy learnt to jump when it goes from the mid to the low ground. It does not need to. When there
# is an obstacle, either you jump early or you don't jump, but the policy can't do that right now."
# and: "v103 must integrate in the training recipe the presence of 3 obstacles one after the other.
# We should have dedicated episodes for it. You need to solve those two bottlenecks at once."
#
# ---- CHANGE 1: stepdown_grounded_ok (a SIM-FIDELITY fix, device-measured, no fitting) -------------
# scratchpad/stepdown_forensic.py over the human's 41 km and the v92b/v102 device sets, split by the
# exact level transition (P(jump) is a FREE CHOICE for the human):
#     transition     HUMAN P(jump)   grounded crossings / deaths     v92b + v102
#     1->2  UP          100.0%             0  /  -                      100%
#     2->3  UP          100.0%             0  /  -                      100%
#     3->2  down         92.9%             1  /  0                      100%
#     3->1  down         44.4%             5  /  0                      100%
#     2->1  down         29.4%            12  /  0                      100%
# The human walks 17 of 26 step-downs to LOW ground and never dies; every policy jumps 100% of all 55.
# The sim demanded airborne in BOTH directions, so "always jump" was the only correct sim policy. The
# device streams must_jump=True on 100% of crevasse structures INCLUDING mid->low (verified), so the
# OBSERVATION is faithful and this is purely a DEATH-RULE gap: the flag is HARD in-sim, SOFT on the
# phone. Cost of the superstition, with a hazard within 4 cells after the cap: the forced jump kills
# 50% (v102, n=4) and 16.7% (v92b, n=6) vs the human's 0% either way.
# Verified end-to-end: a never-jumping policy used to die at the FIRST crevasse (median 1 crevasse
# cell, 44 m); with the knob on it walks the step-downs and dies at the first step-UP (91 m), which
# still kills correctly. Equivalence: knob OFF is bit-identical (no rng draw), and a new
# `stepdown_soft` variant exercises it ON.
#
# ---- CHANGE 2: a dedicated 3-obstacle-chain episode profile (the user's ask) ----------------------
# Measured first (scratchpad/chain3_forensic.py, counting STRUCTURES -- a 2-cell crevasse or HoleLong
# is ONE obstacle, not two; the cell-level count inflates the tight bucket badly):
#     v102 device   25 chains / 18.6 km = 1.34/km, 12.0% die inside one
#     v92b device   25 chains / 35.1 km = 0.71/km,  0.0%
#     HUMAN         45 chains / 41.2 km = 1.09/km,  0.0%
# So the sim is NOT missing the situation (baseline generation emits 0.83-0.88 chains/km vs the
# device's 1.09-1.34) -- the policy is simply BAD at it. That makes this a training-EXPOSURE problem,
# which is what episode_profiles are for, not a coverage/fidelity one.
# The profile was calibrated by a generator audit rather than guessed, held in the ramp's tightest
# tier (>4.5 km):
#     anchor                                       0.83 chains/km   0.10 tight/km
#     gauntlet (the existing profile)              1.04             0.18
#     op.9 + compound.7 + widegap.6 + zone7        2.72             0.30    <-- chosen
# NOTE obstacle_prob SATURATES under the ramp (0.90 and 0.98 both give ~1.0/km) because the ramp owns
# min_space_between_obstacles -- so the existing gauntlet profile was already at that ceiling and the
# lever had to be crevasse frequency + compounds instead. Only per-episode keys that BOTH the torch
# and C++ backends accept are used (max_back_to_back_obstacles / double_obstacle_percent /
# device_difficulty_ramp are global-only in both, and the C++ front-end rejects them loudly).
# Anchor share held at 45.5% (the anti-drift floor).
#
# ---- JUDGING (the two changes hit DIFFERENT indicators, so they stay attributable) ---------------
#  1. scratchpad/stepdown_sim.py  -> P(jump | step-DOWN to low) must fall from 100% toward the
#     human's 29.4%. If it stays pinned at 100%, the world change alone was too weak a signal and the
#     next lever is reaction_bonus, which pays 0.1 for a JUMP on ANY must_jump cell incl. this soft
#     one (the v91 -> v92 "weak signal needs an amplifier" lesson).
#  2. scratchpad/chain3_forensic.py on a sim rollout -> deaths per 3-chain encounter must fall from
#     12% toward the human's 0%.
#  3. GUARD the fixed complaints: held-jump <= ~1/km, unjustified ground strafes must NOT regress,
#     narrow deaths ~0%.
set -euo pipefail
python scripts/train_fast.py \
  --backend cpp \
  --init-from runs/ppo_v66_obs65_land.zip \
  --timesteps 10000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 --ent-coef 0.01 \
  --learning-rate 8e-5 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --airborne-action-penalty 0.08 --strafe-penalty 0.05 --turn-buffer-bonus 0.5 \
  --turn-explore-prob 0.5 --turn-explore-anneal-frac 0.5 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer --stepdown-bonus-gate \
  --sim-config "$(cat scratchpad/_v103_cfg.json)" \
  --ckpt-every 2000000 \
  --out runs/ppo_v104_bonusgate
echo "V103 DONE"
