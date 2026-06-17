#!/usr/bin/env bash
# Self-healing training loop.
#
# Wraps scripts/train.py so a transient phone hiccup (dropped adb forward, interstitial
# ad, low-battery pause) — which collapses the whole SubprocVecEnv — becomes a ~15 s
# auto-restart instead of stopping training for good. Each relaunch --resume's from the
# model train.py saves on exit (runs/ppo_templerun_final.zip), so progress carries over
# (at most the steps since the last checkpoint are lost).
#
# Usage:  scripts/train_loop.sh --num-envs 3 --action-cost 0.03 --reaction-bonus 0.1 --no-progress-bar
# Stop:   TaskStop the background task (or Ctrl-C). The watchdog + forward-keeper still
#         heal the phones between restarts; a phone that is physically down (screen off,
#         dead battery, unplugged) still needs a manual fix — the loop just keeps retrying.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RESUME=runs/ppo_templerun_final.zip
BACKOFF=15
n=0
while true; do
  n=$((n + 1))
  ts=$(date '+%H:%M:%S')
  if [ -f "$RESUME" ]; then resume_args="--resume $RESUME"; else resume_args=""; fi
  echo "[train-loop] === run #$n @ $ts ===  args: $* $resume_args"
  $PY scripts/train.py "$@" $resume_args
  code=$?
  ts=$(date '+%H:%M:%S')
  echo "[train-loop] run #$n exited code=$code @ $ts -> restarting in ${BACKOFF}s (watchdog/keeper heal phones meanwhile)"
  sleep "$BACKOFF"
done
