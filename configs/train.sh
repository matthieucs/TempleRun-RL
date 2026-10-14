#!/usr/bin/env bash
# Fine-tune the shipped policy in the simulator (10M steps, ~6 min on a GPU box).
#
#   bash configs/train.sh            -> runs/templerun_runner_next_*.zip
#
# configs/sim_world.json is the simulated world the shipped policy was trained in (obstacle mix,
# device-measured latencies, death rules). The reward terms below are explained in README.md.
# The C++ backend (--backend cpp) needs cpp_sim/build.sh first; it was developed on Windows.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train/train_fast.py \
  --backend cpp \
  --init-from models/templerun_runner_v1.zip \
  --timesteps 10000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 \
  --ent-coef 0.03 --ent-coef-final 0.005 --ent-schedule cosine \
  --learning-rate 1.2e-4 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --airborne-action-penalty 0.08 --strafe-penalty 0.05 --turn-buffer-bonus 0.5 \
  --turn-explore-prob 0.5 --turn-explore-anneal-frac 0.5 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer --stepdown-bonus-gate \
  --landing-hazard-penalty 0.30 \
  --stepdown-jump-penalty 8.0 --stepdown-explore-prob 0.5 --stepdown-explore-anneal-frac 0.5 \
  --sim-config "$(cat configs/sim_world.json)" \
  --ckpt-every 2000000 \
  --out runs/templerun_runner_next
