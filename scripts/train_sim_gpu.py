#!/usr/bin/env python3
"""Train PPO on the BATCHED torch sim (``sim_env_torch``) — the ~65x-faster path.

Same recipe surface as ``train_sim.py`` (same knobs, same sim-config JSON, same sidecar
format, checkpoints in runs/), but the env is one ``TorchTempleRunVecEnv`` with hundreds+
of batched envs instead of a SubprocVecEnv of python sims. ``train_sim.py`` is untouched.

PPO buffer geometry: the reference recipe collects 8 envs x 512 steps = 4096 transitions
per update (minibatch 64, 10 epochs). With B envs we keep the SAME 4096-sample updates by
default via ``--n-steps = 4096 // B`` (>= 4); at the default B=256 that's n_steps=16 —
shorter GAE horizon per rollout, but with gae_lambda 0.95 the effective advantage horizon
(~17 steps) is preserved. Override --n-steps/--batch-size to experiment.

Gate before shipping anything trained here: scripts/sim_equivalence.py GREEN +
scripts/scorecard_torch.py --strict GREEN + eval via scratchpad/eval_swipe_deaths.py
(the CPU sim stays the judge).

    python scripts/train_sim_gpu.py --timesteps 6000000 --num-envs 256 \
        --net-arch 256,256 --action-history-len 4 ... (same flags as train_sim.py)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402

import templerun_rl.constants as C  # noqa: E402
from templerun_rl.sim_env_torch import TorchTempleRunVecEnv  # noqa: E402
from train_sim import EvalCallback, _rollout  # noqa: E402  (CPU-sim eval — coarse progress signal)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=6_000_000)
    p.add_argument("--num-envs", type=int, default=256, help="batched envs in ONE tensor sim")
    p.add_argument("--device", default="cuda", help="sim device: cuda | cpu (net always follows SB3 auto)")
    p.add_argument("--n-steps", type=int, default=0,
                   help="PPO rollout length per env; 0 = auto (4096 // num_envs, min 4)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--reward-scale", type=float, default=C.DEFAULT_REWARD_SCALE)
    p.add_argument("--death-penalty", type=float, default=1.0)
    p.add_argument("--action-cost", type=float, default=0.02)
    p.add_argument("--reaction-bonus", type=float, default=0.1)
    p.add_argument("--turn-spam-penalty", type=float, default=0.1)
    p.add_argument("--idle-action-penalty", type=float, default=0.05)
    p.add_argument("--redundant-action-penalty", type=float, default=0.0)
    p.add_argument("--strafe-penalty", type=float, default=0.0)
    p.add_argument("--narrow-hold-bonus", type=float, default=0.0)
    p.add_argument("--aim-center-weight", type=float, default=0.0)
    p.add_argument("--velocity-drop-penalty", type=float, default=0.0)
    p.add_argument("--off-center-penalty", type=float, default=0.0)
    p.add_argument("--slide-jump-penalty", type=float, default=0.0)
    p.add_argument("--turn-approach-buffer", dest="turn_approach_buffer", action="store_true",
                   default=False)
    p.add_argument("--curriculum-max-start-dist", type=float, default=0.0)
    p.add_argument("--init-from", default=None)
    p.add_argument("--action-history-len", type=int, default=0)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--turn-linger", type=int, default=0)
    p.add_argument("--sim-config", default="")
    p.add_argument("--step-dt", type=float, default=C.DEFAULT_STEP_DT)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--net-arch", default="256,256")
    p.add_argument("--domain-rand", dest="domain_rand", action="store_true", default=True)
    p.add_argument("--no-domain-rand", dest="domain_rand", action="store_false")
    p.add_argument("--out", default="runs/ppo_gpu")
    p.add_argument("--eval-every", type=int, default=200_000,
                   help="CPU-sim greedy eval cadence (0 = off)")
    args = p.parse_args()

    cfg = {
        "domain_rand": bool(args.domain_rand),
        "turn_approach_buffer": bool(args.turn_approach_buffer),
        "turn_linger_frames": int(args.turn_linger),
        "curriculum_max_start_dist": float(args.curriculum_max_start_dist),
    }
    if args.sim_config:
        cfg.update(json.loads(args.sim_config))

    shaping = dict(
        action_cost=args.action_cost, reaction_bonus=args.reaction_bonus,
        turn_spam_penalty=args.turn_spam_penalty, idle_action_penalty=args.idle_action_penalty,
        redundant_action_penalty=args.redundant_action_penalty, strafe_penalty=args.strafe_penalty,
        narrow_hold_bonus=args.narrow_hold_bonus, aim_center_weight=args.aim_center_weight,
        velocity_drop_penalty=args.velocity_drop_penalty, off_center_penalty=args.off_center_penalty,
        slide_jump_penalty=args.slide_jump_penalty,
    )

    K, FS = args.action_history_len, args.frame_stack
    print(f"baselines (CPU-sim): oracle {_rollout('oracle', 10, action_history_len=K, frame_stack=FS)}, "
          f"random {_rollout('random', 10, action_history_len=K, frame_stack=FS)}", flush=True)

    venv = TorchTempleRunVecEnv(
        args.num_envs, config=cfg, device=args.device, seed=args.seed,
        step_dt=args.step_dt, max_episode_steps=args.max_episode_steps,
        reward_scale=args.reward_scale, reward_mode="distance",
        death_penalty=args.death_penalty, action_history_len=K, frame_stack=FS, **shaping)
    venv = VecMonitor(venv)

    n_steps = args.n_steps if args.n_steps > 0 else max(4, 4096 // args.num_envs)
    buf = n_steps * args.num_envs
    net_arch = [int(x) for x in str(args.net_arch).split(",") if x.strip()]
    print(f"policy net_arch={net_arch}  B={args.num_envs} n_steps={n_steps} "
          f"(rollout buffer {buf})  sim device={args.device}", flush=True)
    model = PPO(
        "MlpPolicy", venv,
        n_steps=n_steps, batch_size=args.batch_size, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        ent_coef=args.ent_coef, learning_rate=args.learning_rate, clip_range=0.2,
        policy_kwargs=dict(net_arch=net_arch), seed=args.seed,
        tensorboard_log="runs/tb_sim_gpu", verbose=0,
    )
    if args.init_from:
        model.set_parameters(args.init_from)
        print(f"warm-started weights from {args.init_from}", flush=True)

    callbacks = [CheckpointCallback(save_freq=max(1, 100_000 // args.num_envs),
                                    save_path="runs", name_prefix=Path(args.out).name)]
    if args.eval_every > 0:
        callbacks.append(EvalCallback(every=args.eval_every, action_history_len=K, frame_stack=FS))
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(f"{args.out}_final")
    with open(f"{args.out}_final.json", "w") as f:
        json.dump({"frame_stack": FS, "action_history_len": K, "net_arch": net_arch}, f)
    print(f"saved {args.out}_final.zip (+ .json: fs={FS} K={K} arch={net_arch})", flush=True)

    ms, md, deaths = _rollout(lambda o: int(model.predict(o, deterministic=True)[0]), n_eps=30,
                              action_history_len=K, frame_stack=FS)
    print(f"FINAL policy (CPU-sim eval): median steps={ms:.0f} distance={md:.0f} deaths={deaths}/30",
          flush=True)


if __name__ == "__main__":
    main()
