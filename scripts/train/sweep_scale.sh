#!/usr/bin/env bash
# Scaling sweep toward the 10,000m goal: vary frame_stack (FS), action_history (K), net arch.
# Fixed working recipe: DR+zones, ent_coef 0.04 (escapes NOOP-collapse), action_cost 0.06 (no strafe
# spam), redundant 0.12. Each run -> trains, writes a sidecar, scores per-TURN reliability + hard.
# Results appended to logs/sweep_scale.txt. Models kept (with .json sidecars) for later phone tests.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
STEPS=${STEPS:-800000}
ENVS=${ENVS:-10}
OUT=logs/sweep_scale.txt
: > "$OUT"

# config: FS  K  ARCH  TAG
CONFIGS=(
  "1 4 128,128       fs1k4-128"
  "2 4 128,128       fs2k4-128"
  "4 4 128,128       fs4k4-128"
  "2 4 256,256       fs2k4-256"
  "4 4 256,256       fs4k4-256"
  "4 0 256,256       fs4k0-256"
  "4 8 256,256       fs4k8-256"
  "4 4 256,256,128   fs4k4-256x3"
)

for cfg in "${CONFIGS[@]}"; do
  read -r FS K ARCH TAG <<< "$cfg"
  echo "=== TRAIN $TAG  (frame_stack=$FS action_history=$K arch=$ARCH, $STEPS steps) ===" | tee -a "$OUT"
  $PY scripts/train/train_sim.py --timesteps "$STEPS" --num-envs "$ENVS" \
      --max-episode-steps 3000 --ent-coef 0.04 --domain-rand \
      --net-arch "$ARCH" --redundant-action-penalty 0.12 --action-cost 0.06 \
      --action-history-len "$K" --frame-stack "$FS" \
      --out "runs/ppo_${TAG}" > "logs/train_${TAG}.log" 2>&1
  echo "--- per-TURN reliability ($TAG) ---" | tee -a "$OUT"
  $PY scripts/validate/score_turns.py "runs/ppo_${TAG}_final.zip" --episodes 120 --cap 6000 2>&1 | tee -a "$OUT"
  echo "--- hard-config ($TAG) ---" | tee -a "$OUT"
  $PY scripts/validate/score_hard.py "runs/ppo_${TAG}_final.zip" --episodes 60 2>&1 | grep median_steps | tee -a "$OUT"
  echo "" | tee -a "$OUT"
done
echo "=== SWEEP_SCALE DONE ===" | tee -a "$OUT"
