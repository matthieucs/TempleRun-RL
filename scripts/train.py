#!/usr/bin/env python3
"""Train a PPO agent on TempleRun-v0 with Stable-Baselines3.

Real-time training against physical phone(s). One env per phone. Bring each device up
first (own terminal, distinct --device-index), then train:

    # one phone:
    python scripts/setup_device.py            # forwards + injects the gadget (leave running)
    python scripts/train.py --timesteps 100000

    # two phones (DummyVecEnv -> SubprocVecEnv so the ~10 Hz waits overlap):
    python scripts/setup_device.py --serial PHONE_A --device-index 0   # terminal 1
    python scripts/setup_device.py --serial PHONE_B --device-index 1   # terminal 2
    python scripts/train.py --timesteps 100000 --num-envs 2            # terminal 3

With --num-envs N, env i connects to device_ports(i) and PPO collects n_steps*N
transitions per rollout. Checkpoints / final model under --save-dir; TensorBoard under
--tensorboard. Ctrl-C saves a final model before exiting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import templerun_rl  # noqa: F401  (registers TempleRun-v0)
from templerun_rl import constants as C


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--save-dir", default="runs")
    p.add_argument("--tensorboard", default="runs/tb")
    p.add_argument("--resume", default=None, help="path to a .zip model to continue training")
    p.add_argument("--checkpoint-freq", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-progress-bar", action="store_true",
                   help="disable the tqdm/rich progress bar (use for background/headless runs)")
    # parallelism: one env per physical phone. Bring each device up first with
    # `setup_device.py --device-index i` (host ports offset by i*100); env i then
    # connects to device_ports(i). >1 env uses SubprocVecEnv so the ~10 Hz waits overlap.
    p.add_argument("--num-envs", type=int, default=1,
                   help="number of phones/envs to train across in parallel (default 1)")
    # env knobs
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--step-dt", type=float, default=0.1)
    p.add_argument("--max-episode-steps", type=int, default=1000,
                   help="episode-length ceiling in steps (~0.1 s each); the run usually ends earlier by death")
    p.add_argument("--grow-episode-steps", action="store_true",
                   help="auto-raise the cap each time the agent survives a full episode (ratchet, never shrinks)")
    p.add_argument("--max-episode-steps-growth", type=float, default=1.5,
                   help="multiply the cap by this on each pure-truncation (with --grow-episode-steps)")
    p.add_argument("--max-episode-steps-ceiling", type=int, default=6000,
                   help="hard upper bound the growing cap can never exceed (~10 min at 0.1 s/step)")
    p.add_argument("--reward-scale", type=float, default=0.01)
    p.add_argument("--death-penalty", type=float, default=1.0)
    # reward shaping (opt-in) — speeds learning / curbs early thrashing; see env._shaping_reward
    p.add_argument("--action-cost", type=float, default=0.0,
                   help="penalty per non-NoOp action (curb thrashing); try ~0.02-0.05")
    p.add_argument("--reaction-bonus", type=float, default=0.0,
                   help="bonus for the correct jump/slide/turn reaction; try ~0.1 (> action-cost)")
    return p.parse_args()


def _make_one(args: argparse.Namespace, index: int):
    """Build a Monitored env bound to device `index`'s host ports.

    Defined at module scope (not a nested closure) so it pickles cleanly for
    SubprocVecEnv on Windows (spawn). cloudpickle captures `args`/`index`.
    """
    def _factory():
        import templerun_rl  # noqa: F401  ensure TempleRun-v0 is registered in this process
        from templerun_rl import constants as _C

        ports = _C.device_ports(index)
        env = gym.make(
            "TempleRun-v0",
            host=args.host,
            state_port=ports["state_port"],
            action_port=ports["action_port"],
            step_dt=args.step_dt,
            max_episode_steps=args.max_episode_steps,
            grow_episode_steps=args.grow_episode_steps,
            max_episode_steps_growth=args.max_episode_steps_growth,
            max_episode_steps_ceiling=args.max_episode_steps_ceiling,
            reward_scale=args.reward_scale,
            death_penalty=args.death_penalty,
            action_cost=args.action_cost,
            reaction_bonus=args.reaction_bonus,
        )
        return Monitor(env)

    return _factory


def make_env(args: argparse.Namespace):
    factories = [_make_one(args, i) for i in range(args.num_envs)]
    if args.num_envs == 1:
        # One phone: sequential is correct and avoids subprocess overhead.
        return DummyVecEnv(factories)
    # N phones: separate processes so each env's real-time wait runs concurrently.
    return SubprocVecEnv(factories)


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
            progress_bar=not args.no_progress_bar,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving final model.")
    finally:
        model.save(str(final_path))
        env.close()
        print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
