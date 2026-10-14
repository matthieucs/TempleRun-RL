#!/usr/bin/env bash
# Faster net-arch sweep: 3 larger nets at 1M steps (past the ~800k convergence), score on hard config.
# 64,64 already trained+promoted to ppo_arch_64-64_final.zip; scored separately.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
STEPS=${STEPS:-1000000}
ENVS=${ENVS:-10}
ARCHS=("128,128" "256,256" "128,128,128")
OUT=logs/sweep_arch.txt
for A in "${ARCHS[@]}"; do
  TAG=$(echo "$A" | tr ',' '-')
  echo "=== TRAIN net=$A ($STEPS steps) ===" | tee -a "$OUT"
  $PY scripts/train/train_sim.py --timesteps "$STEPS" --num-envs "$ENVS" \
      --max-episode-steps 3000 --ent-coef 0.012 --domain-rand \
      --net-arch "$A" --out "runs/ppo_arch_${TAG}" > "logs/train_arch_${TAG}.log" 2>&1
  echo "--- SCORE net=$A on hard config ---" | tee -a "$OUT"
  $PY scripts/validate/score_hard.py "runs/ppo_arch_${TAG}_final.zip" --episodes 80 2>&1 | tee -a "$OUT"
  echo "" | tee -a "$OUT"
done
echo "--- SCORE net=64,64 (pretrained) on hard config ---" | tee -a "$OUT"
$PY scripts/validate/score_hard.py "runs/ppo_arch_64-64_final.zip" --episodes 80 2>&1 | tee -a "$OUT"
echo "=== SWEEP2 DONE ===" | tee -a "$OUT"
