#!/usr/bin/env python3
"""Train a PPO policy in the headless sim (``TempleRunSim-v0``) — fast, parallel, free.

The sim mirrors the real env's obs/action spaces (Box(21)/Discrete(7)) and reward (distance-delta),
so a policy trained here is drop-in on the phone (``scripts/evaluate.py`` / deploy). The sim runs at
~10k+ steps/s, so we vectorize across cores and train millions of steps in minutes.

    python scripts/train_sim.py --timesteps 3000000 --num-envs 8
    python scripts/train_sim.py --timesteps 1000000 --reaction-bonus 0.1 --ent-coef 0.01

Saves runs/ppo_sim_final.zip + checkpoints; TensorBoard under runs/tb_sim.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import templerun_rl  # noqa: F401,E402  registers TempleRunSim-v0
import templerun_rl.constants as C  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv  # noqa: E402


def _make_env(args, rank: int):
    def _factory():
        import templerun_rl  # noqa: F401
        env = gym.make(
            "TempleRunSim-v0",
            max_episode_steps=args.max_episode_steps,
            reward_mode="distance",
            reward_scale=args.reward_scale,
            death_penalty=args.death_penalty,
            action_cost=args.action_cost,
            reaction_bonus=args.reaction_bonus,
            turn_spam_penalty=args.turn_spam_penalty,
            idle_action_penalty=args.idle_action_penalty,
            seed=args.seed + rank,
        )
        return env
    return _factory


def _oracle(st: dict) -> int:
    if st["must_jump"]:
        return int(C.Action.JUMP)
    if st["must_slide"]:
        return int(C.Action.SLIDE)
    if st["is_turn"]:
        if st["can_turn_left"]:
            return int(C.Action.TURN_LEFT)
        if st["can_turn_right"]:
            return int(C.Action.TURN_RIGHT)
        return int(C.Action.TURN_LEFT)
    return int(C.Action.NOOP)


def _rollout(policy, n_eps=20, seed0=10000):
    """Run a policy (callable obs->action OR 'oracle'/'random') in a single sim; return survival."""
    env = gym.make("TempleRunSim-v0", max_episode_steps=100000).unwrapped
    steps, dists, deaths = [], [], 0
    rng = np.random.default_rng(seed0)
    for ep in range(n_eps):
        o, i = env.reset(seed=seed0 + ep)
        st = i["raw_state"]
        s = 0
        while True:
            if policy == "oracle":
                a = _oracle(st)
            elif policy == "random":
                a = int(rng.integers(0, C.N_ACTIONS))
            else:
                a = int(policy(o))
            o, r, term, trunc, info = env.step(a)
            st = info["raw_state"]
            s += 1
            if term:
                deaths += 1
                break
            if trunc:
                break
        steps.append(s)
        dists.append(info["distance"])
    return float(np.median(steps)), float(np.median(dists)), deaths


class EvalCallback(BaseCallback):
    """Every ``every`` steps, roll out the greedy policy and log survival vs oracle/random."""
    def __init__(self, every: int, verbose=1):
        super().__init__(verbose)
        self.every = every
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.every:
            return True
        self._last = self.num_timesteps
        model = self.model

        def greedy(obs):
            a, _ = model.predict(obs, deterministic=True)
            return int(a)
        ms, md, deaths = _rollout(greedy, n_eps=15)
        self.logger.record("eval/median_steps", ms)
        self.logger.record("eval/median_distance", md)
        self.logger.record("eval/deaths_per15", deaths)
        if self.verbose:
            print(f"[eval @ {self.num_timesteps}] median steps={ms:.0f} distance={md:.0f} "
                  f"deaths={deaths}/15", flush=True)
        return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--reward-scale", type=float, default=C.DEFAULT_REWARD_SCALE)
    p.add_argument("--death-penalty", type=float, default=1.0)
    p.add_argument("--action-cost", type=float, default=0.02)
    p.add_argument("--reaction-bonus", type=float, default=0.1)
    p.add_argument("--turn-spam-penalty", type=float, default=0.1)
    p.add_argument("--idle-action-penalty", type=float, default=0.05)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ppo_sim")
    args = p.parse_args()

    print(f"baselines: oracle {_rollout('oracle', 10)}, random {_rollout('random', 10)}", flush=True)

    factories = [_make_env(args, i) for i in range(args.num_envs)]
    venv = SubprocVecEnv(factories) if args.num_envs > 1 else DummyVecEnv(factories)

    model = PPO(
        "MlpPolicy", venv,
        n_steps=512, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        ent_coef=args.ent_coef, learning_rate=3e-4, clip_range=0.2,
        policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log="runs/tb_sim", verbose=0,
    )
    ckpt = CheckpointCallback(save_freq=max(1, 200_000 // args.num_envs),
                              save_path="runs", name_prefix="ppo_sim")
    evalcb = EvalCallback(every=200_000)
    model.learn(total_timesteps=args.timesteps, callback=[ckpt, evalcb], progress_bar=False)
    model.save(f"{args.out}_final")
    print(f"saved {args.out}_final.zip", flush=True)

    ms, md, deaths = _rollout(lambda o: int(model.predict(o, deterministic=True)[0]), n_eps=30)
    print(f"FINAL policy: median steps={ms:.0f} distance={md:.0f} deaths={deaths}/30", flush=True)


if __name__ == "__main__":
    main()
