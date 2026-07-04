#!/usr/bin/env bash
# Net-architecture sweep: train each arch (DR+zones) then score on the HARD held-out config.
# Sequential so each run gets all cores. Results appended to logs/sweep_arch.txt.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
STEPS=${STEPS:-1500000}
ENVS=${ENVS:-10}
ARCHS=("64,64" "128,128" "256,256" "128,128,128")
OUT=logs/sweep_arch.txt
: > "$OUT"
for A in "${ARCHS[@]}"; do
  TAG=$(echo "$A" | tr ',' '-')
  echo "=== TRAIN net=$A ($STEPS steps) ===" | tee -a "$OUT"
  $PY scripts/train_sim.py --timesteps "$STEPS" --num-envs "$ENVS" \
      --max-episode-steps 3000 --ent-coef 0.012 --domain-rand \
      --net-arch "$A" --out "runs/ppo_arch_${TAG}" > "logs/train_arch_${TAG}.log" 2>&1
  echo "--- SCORE net=$A on hard config ---" | tee -a "$OUT"
  $PY scripts/score_hard.py "runs/ppo_arch_${TAG}_final.zip" --episodes 80 2>&1 | tee -a "$OUT"
  echo "" | tee -a "$OUT"
done
echo "=== SWEEP DONE ===" | tee -a "$OUT"
