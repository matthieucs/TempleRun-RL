#!/usr/bin/env python3
"""Train a PPO agent on TempleRun-v0 with Stable-Baselines3.

Single-env, real-time training against ONE physical phone (no SubprocVecEnv — there is
exactly one game instance). Bring the device up first:

    python scripts/setup_device.py        # forwards + injects the gadget (leave running)
    # then, in another terminal, with the game in a run/game-over screen:
    python scripts/train.py --timesteps 100000

Checkpoints and the final model are written under --save-dir; TensorBoard logs under
--tensorboard. Ctrl-C saves a final model before exiting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import templerun_rl  # noqa: F401  (registers TempleRun-v0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--save-dir", default="runs")
    p.add_argument("--tensorboard", default="runs/tb")
    p.add_argument("--resume", default=None, help="path to a .zip model to continue training")
    p.add_argument("--checkpoint-freq", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    # env knobs
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--step-dt", type=float, default=0.1)
    p.add_argument("--max-episode-steps", type=int, default=1000)
    p.add_argument("--reward-scale", type=float, default=0.01)
    p.add_argument("--death-penalty", type=float, default=1.0)
    return p.parse_args()


def make_env(args: argparse.Namespace):
    def _factory():
        env = gym.make(
            "TempleRun-v0",
            host=args.host,
            step_dt=args.step_dt,
            max_episode_steps=args.max_episode_steps,
            reward_scale=args.reward_scale,
            death_penalty=args.death_penalty,
        )
        return Monitor(env)

    return DummyVecEnv([_factory])


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, tensorboard_log=args.tensorboard)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,            # encourage exploration across the 7 actions
            learning_rate=3e-4,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[64, 64]),
            tensorboard_log=args.tensorboard,
            seed=args.seed,
            verbose=1,
        )

    checkpoint = CheckpointCallback(
        save_freq=args.checkpoint_freq, save_path=str(save_dir), name_prefix="ppo_templerun"
    )

    final_path = save_dir / "ppo_templerun_final.zip"
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint,
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving final model.")
    finally:
        model.save(str(final_path))
        env.close()
        print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
