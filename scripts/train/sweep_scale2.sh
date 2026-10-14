#!/usr/bin/env bash
# Lean+fast scaling sweep (fs=1 control already done = ppo_fs1k4-128). Key configs only, 700k steps,
# FAST per-turn scoring (50 eps, cap 2500). Answers: does frame-stack lift turn reliability, does a
# bigger net help with rich obs, does frame-stack alone match frame-stack+action-history.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
STEPS=${STEPS:-700000}
OUT=logs/sweep_scale.txt   # append to the same results file (keeps config 1)

# FS K ARCH TAG
CONFIGS=(
  "2 4 128,128  fs2k4-128"
  "4 4 128,128  fs4k4-128"
  "4 4 256,256  fs4k4-256"
  "4 0 256,256  fs4k0-256"
)
for cfg in "${CONFIGS[@]}"; do
  read -r FS K ARCH TAG <<< "$cfg"
  echo "=== TRAIN $TAG (fs=$FS K=$K arch=$ARCH, $STEPS steps) ===" | tee -a "$OUT"
  $PY scripts/train/train_sim.py --timesteps "$STEPS" --num-envs 10 \
      --max-episode-steps 3000 --ent-coef 0.04 --domain-rand \
      --net-arch "$ARCH" --redundant-action-penalty 0.12 --action-cost 0.06 \
      --action-history-len "$K" --frame-stack "$FS" \
      --out "runs/ppo_${TAG}" > "logs/train_${TAG}.log" 2>&1
  echo "--- per-TURN reliability ($TAG) ---" | tee -a "$OUT"
  $PY scripts/validate/score_turns.py "runs/ppo_${TAG}_final.zip" --episodes 50 --cap 2500 2>&1 | tee -a "$OUT"
  echo "" | tee -a "$OUT"
done
echo "=== SWEEP_SCALE2 DONE ===" | tee -a "$OUT"
