#!/usr/bin/env bash
# v107 = the outcome-calibrated world + the USER's two training suggestions.
#
# WHY v106 FAILED (and it is a measurement lesson, not just a dose one): its curve sat FLAT at
# ~1050 from 2M to 10M. Scoring the known-good v104 in that world gave median 1279 vs its DEVICE
# median 3249 = 2.5x too hard. Isolating the four changes showed ONE culprit: tight_pair_prob 0.25
# crushes v104 from 4728 -> 672 on its own. The project's own notes had already measured this on
# 2026-07-25 ("device-matching dose is ~0.08-0.10") and I set 0.25 anyway.
# WORSE: for hours the sim medians were CENSORED -- scripts/analysis/hole_landing_sim.py passed
# max_episode_steps through gym.make(), which the registered env ignores, so every distance capped
# at ~2607 m and every world looked identical. Fixed + asserted; all difficulty numbers above are
# post-fix.
#
# WORLD (each dose calibrated on the OUTCOME = v104's sim median vs its device median 3249):
#   obstacle_type_weights + zone_change_every_cells -> the DEVICE structure mix (crevasse 34.2 ->
#     ~20% vs device 16.1, flame 8.5 -> ~17% vs 18.7). Alone this makes the world EASIER (7148).
#   pair_same_type_prob 0.6 (SLIDES only) -> the device's repeated-slide rhythm, its two most common
#     orderings (TreeSlide->TreeSlide 9.8%, Flame->Flame 8.1%); sim 2.9% -> ~14%.
#   tight_pair_prob 0.05 -> difficulty back to 1.07-1.16x device (0.12 cliffs to 0.42x).
#   jump_dur_stepdown_scale 1.17 -> device-measured 3-cell step-down arc.
#   + landing_hazard now uses that height-scaled span (obs law, CPU + C++).
#
# TRAINING (USER 2026-07-27: "you may start from v104 instead of training from scratch. Or you
# increase the entropy and you put a schedule on its coefficient"): BOTH, because they solve the two
# halves of v106's flat curve -- v104 already has the skills we do not want to relearn (held-jump
# 0.00/km, narrow deaths 0%), and a warm-started policy in a CHANGED world needs exploration
# re-opened or it never samples the alternatives. ent 0.03 -> 0.005 cosine, LR 5e-5 (the documented
# warm-start rate; 3e-4 always collapses a warm start).
set -euo pipefail
python scripts/train/train_fast.py \
  --backend cpp \
  --init-from runs/ppo_v107_calibrated_final.zip \
  --timesteps 10000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 \
  --ent-coef 0.03 --ent-coef-final 0.005 --ent-schedule cosine \
  --learning-rate 5e-5 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --airborne-action-penalty 0.08 --strafe-penalty 0.05 --turn-buffer-bonus 0.5 \
  --turn-explore-prob 0.5 --turn-explore-anneal-frac 0.5 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer --stepdown-bonus-gate \
  --sim-config "$(cat experiments/_v108_cfg.json)" \
  --ckpt-every 2000000 \
  --out runs/ppo_v108_turnfix
echo "V107 DONE"
