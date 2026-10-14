#!/usr/bin/env python3
"""Evaluate a trained PPO model (or a random baseline) on TempleRun-v0.

    python scripts/device/evaluate.py --model runs/ppo_templerun_final.zip --episodes 10
    python scripts/device/evaluate.py --random --episodes 5

Reports mean/median episode reward, episode length, and final in-game score.
Requires the device to be up (scripts/device/setup_device.py) with the game in a run.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import statistics

import gymnasium as gym

import templerun_rl  # noqa: F401  (registers TempleRun-v0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="path to a trained PPO .zip")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--random", action="store_true", help="ignore --model; use a uniform-random policy")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--step-dt", type=float, default=0.1)
    p.add_argument("--max-episode-steps", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make(
        "TempleRun-v0",
        host=args.host,
        step_dt=args.step_dt,
        max_episode_steps=args.max_episode_steps,
    )

    policy = None
    if not args.random:
        if not args.model:
            raise SystemExit("Pass --model PATH or --random.")
        from stable_baselines3 import PPO

        policy = PPO.load(args.model)
        print(f"Loaded model: {args.model}")
    else:
        print("Random-policy baseline.")

    rewards, lengths, scores = [], [], []
    print(f"\n{'ep':>3}  {'reward':>10}  {'length':>7}  {'score':>8}")
    for ep in range(args.episodes):
        obs, info = env.reset()
        total, steps, done = 0.0, 0, False
        last_score = info.get("score", 0)
        while not done:
            if policy is None:
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs, deterministic=args.deterministic)
                action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            last_score = info.get("score", last_score)
            done = terminated or truncated
        rewards.append(total)
        lengths.append(steps)
        scores.append(last_score)
        print(f"{ep:>3}  {total:>10.3f}  {steps:>7d}  {last_score:>8d}")

    env.close()

    def summary(name, xs):
        print(f"  {name:<7} mean={statistics.mean(xs):.3f}  median={statistics.median(xs):.3f}  "
              f"min={min(xs):.3f}  max={max(xs):.3f}")

    print("\n=== summary ===")
    summary("reward", rewards)
    summary("length", lengths)
    summary("score", scores)


if __name__ == "__main__":
    main()
