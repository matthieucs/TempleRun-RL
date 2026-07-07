#!/usr/bin/env python3
"""FAST PPO — minimal pure-torch PPO over the batched sim, fully GPU-resident.

Vertical integration: the rollout (obs/actions/rewards/GAE) and the update loop all stay
torch tensors on one device — no numpy round-trips, no SB3 rollout plumbing. The POLICY is
still a stock SB3 MlpPolicy inside a stock PPO object, and saving goes through model.save()
— so checkpoints remain normal SB3 zips, loadable by every eval/deploy tool.

Loss/GAE formulas mirror SB3 PPO defaults exactly (clipped surrogate, plain value MSE,
entropy bonus, grad-clip 0.5, timeout bootstrapping). Validate any recipe change by curve
comparison against scripts/train_sim_gpu.py at equal steps (scripts/eval_ckpt_torch.py).

    python scripts/train_fast.py --timesteps 2000000 ... (same recipe flags)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

import templerun_rl.constants as C  # noqa: E402
from templerun_rl.sim_env_torch import TorchTempleRunVecEnv  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--net-arch", default="256,256")
    p.add_argument("--action-history-len", type=int, default=4)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--turn-linger", type=int, default=3)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--lr-schedule", choices=["constant", "linear", "cosine"], default="constant",
                   help="LR decay over training: constant | linear | cosine (to --lr-final)")
    p.add_argument("--lr-final", type=float, default=None,
                   help="end LR for the schedule (default = --learning-rate = constant)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-from", default=None)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--reward-scale", type=float, default=C.DEFAULT_REWARD_SCALE)
    p.add_argument("--death-penalty", type=float, default=1.0)
    p.add_argument("--action-cost", type=float, default=0.02)
    p.add_argument("--reaction-bonus", type=float, default=0.1)
    p.add_argument("--turn-spam-penalty", type=float, default=0.1)
    p.add_argument("--idle-action-penalty", type=float, default=0.05)
    p.add_argument("--strafe-penalty", type=float, default=0.0)
    p.add_argument("--narrow-hold-bonus", type=float, default=0.0)
    p.add_argument("--aim-center-weight", type=float, default=0.0)
    p.add_argument("--slide-jump-penalty", type=float, default=0.0)
    p.add_argument("--airborne-action-penalty", type=float, default=0.0,
                   help="dock JUMP-while-jumping / SLIDE-while-sliding (jump-spam curb)")
    p.add_argument("--turn-approach-buffer", action="store_true", default=False)
    p.add_argument("--curriculum-max-start-dist", type=float, default=0.0)
    p.add_argument("--domain-rand", dest="domain_rand", action="store_true", default=True)
    p.add_argument("--no-domain-rand", dest="domain_rand", action="store_false")
    p.add_argument("--sim-config", default="")
    p.add_argument("--ckpt-every", type=int, default=100_000)
    p.add_argument("--cuda-graph", action="store_true", default=False,
                   help="capture the PPO minibatch step as a CUDA graph (one launch per bite "
                        "instead of ~30; bit-identical math, pure dispatch optimization). "
                        "Requires --device cuda and total buffer divisible by --batch-size.")
    p.add_argument("--out", default="runs/ppo_fast")
    args = p.parse_args()

    dev = torch.device(args.device)
    cfg = {"domain_rand": args.domain_rand, "turn_approach_buffer": args.turn_approach_buffer,
           "turn_linger_frames": args.turn_linger,
           "curriculum_max_start_dist": args.curriculum_max_start_dist}
    if args.sim_config:
        cfg.update(json.loads(args.sim_config))
    K, FS = args.action_history_len, args.frame_stack
    venv = TorchTempleRunVecEnv(
        args.num_envs, config=cfg, device=args.device, seed=args.seed,
        max_episode_steps=args.max_episode_steps, reward_scale=args.reward_scale,
        reward_mode="distance", death_penalty=args.death_penalty,
        action_cost=args.action_cost, reaction_bonus=args.reaction_bonus,
        turn_spam_penalty=args.turn_spam_penalty, idle_action_penalty=args.idle_action_penalty,
        strafe_penalty=args.strafe_penalty, narrow_hold_bonus=args.narrow_hold_bonus,
        aim_center_weight=args.aim_center_weight, slide_jump_penalty=args.slide_jump_penalty,
        airborne_action_penalty=args.airborne_action_penalty,
        action_history_len=K, frame_stack=FS)

    net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
    # stock SB3 model = policy container + save format; we bypass only its learn() loop.
    model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=args.batch_size,
                policy_kwargs=dict(net_arch=net_arch), learning_rate=args.learning_rate,
                seed=args.seed, device=args.device, verbose=0)
    if args.init_from:
        model.set_parameters(args.init_from)
        print(f"warm-started from {args.init_from}", flush=True)
    policy = model.policy
    opt = policy.optimizer

    # ---- optional CUDA-graph capture of the minibatch step ----
    graph_step = None
    lr_tensor = None   # device tensor holding the LR when graph-captured (set below)
    if args.cuda_graph:
        assert dev.type == "cuda", "--cuda-graph needs --device cuda"
        # Categorical's __init__ host-syncs on a validity check (`valid.all()`), which is
        # illegal during graph capture. It's a sanity assert, not math — disable it.
        torch.distributions.Distribution.set_default_validate_args(False)
        assert (args.n_steps * args.num_envs) % args.batch_size == 0, \
            "buffer must divide evenly by batch-size for graph capture"
        # Adam must be 'capturable' (its step counter lives on-device inside a graph). The LR is a
        # DEVICE TENSOR so an in-place update (lr_tensor.fill_) changes the effective LR the captured
        # graph reads at replay -> supports --lr-schedule without re-capturing the graph.
        lr_tensor = torch.tensor(float(args.learning_rate), device=dev)
        opt = torch.optim.Adam(policy.parameters(), lr=lr_tensor,
                               capturable=True, **{k: v for k, v in
                               [("eps", 1e-5)]})
        policy.optimizer = opt
        bs = args.batch_size
        obs_dim_ = venv.observation_space.shape[0]
        st_obs = torch.zeros(bs, obs_dim_, device=dev)
        st_act = torch.zeros(bs, dtype=torch.int64, device=dev)
        st_logp = torch.zeros(bs, device=dev)
        st_adv = torch.zeros(bs, device=dev)
        st_ret = torch.zeros(bs, device=dev)

        def _bite():
            values, logp, entropy = policy.evaluate_actions(st_obs, st_act)
            a_mb = (st_adv - st_adv.mean()) / (st_adv.std() + 1e-8)
            ratio = torch.exp(logp - st_logp)
            pl = -torch.min(a_mb * ratio,
                            a_mb * ratio.clamp(1 - args.clip_range, 1 + args.clip_range)).mean()
            vl = torch.nn.functional.mse_loss(values.squeeze(-1), st_ret)
            loss = pl + args.vf_coef * vl - args.ent_coef * entropy.mean()
            opt.zero_grad(set_to_none=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            opt.step()

        # warmup on a side stream (required before capture), then record the graph
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _bite()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _bite()

        def graph_step(o, a2, lp, ad, rt):  # noqa: E306
            st_obs.copy_(o)
            st_act.copy_(a2)
            st_logp.copy_(lp)
            st_adv.copy_(ad)
            st_ret.copy_(rt)
            g.replay()
        print("CUDA graph captured for the minibatch step", flush=True)

    B, T = args.num_envs, args.n_steps
    obs_dim = venv.observation_space.shape[0]
    obs_buf = torch.zeros(T, B, obs_dim, device=dev)
    act_buf = torch.zeros(T, B, dtype=torch.int64, device=dev)
    logp_buf = torch.zeros(T, B, device=dev)
    val_buf = torch.zeros(T, B, device=dev)
    rew_buf = torch.zeros(T, B, device=dev)
    done_buf = torch.zeros(T, B, dtype=torch.bool, device=dev)

    # ---- LR schedule: decay learning_rate -> lr_final over training (constant if lr_final unset) ----
    def _sched_lr(g: int) -> float:
        if args.lr_schedule == "constant" or args.lr_final is None:
            return float(args.learning_rate)
        prog = min(1.0, g / max(1, args.timesteps))
        if args.lr_schedule == "linear":
            return float(args.learning_rate + (args.lr_final - args.learning_rate) * prog)
        return float(args.lr_final + 0.5 * (args.learning_rate - args.lr_final)
                     * (1.0 + math.cos(math.pi * prog)))   # cosine

    def _set_lr(v: float) -> None:
        if lr_tensor is not None:
            lr_tensor.fill_(v)          # graph reads this tensor at replay
        else:
            for pg in opt.param_groups:
                pg["lr"] = v

    obs = venv.reset_tensor().to(dev)
    gstep = 0
    next_ckpt = args.ckpt_every
    ep_stats: list[tuple[float, int, float]] = []
    t0 = time.perf_counter()
    Path("runs").mkdir(exist_ok=True)

    while gstep < args.timesteps:
        _set_lr(_sched_lr(gstep))     # schedule the LR for this iteration's minibatch updates
        # ---- rollout (no_grad, all tensors on device) ----
        with torch.no_grad():
            for t in range(T):
                acts, values, logp = policy(obs)
                nobs, rew, done, timeout, term_obs, ep_ret, ep_len, ep_dist = \
                    venv.step_tensor(acts)
                rew = rew.float()
                # SB3 timeout handling: bootstrap the truncated episode's terminal value.
                if bool(timeout.any()):
                    tv = policy.predict_values(term_obs[timeout].to(dev)).squeeze(-1)
                    rew[timeout] += args.gamma * tv
                obs_buf[t] = obs
                act_buf[t] = acts
                logp_buf[t] = logp
                val_buf[t] = values.squeeze(-1)
                rew_buf[t] = rew
                done_buf[t] = done
                if bool(done.any()):
                    for i in torch.nonzero(done).squeeze(1)[:8].tolist():
                        ep_stats.append((float(ep_ret[i]), int(ep_len[i]), float(ep_dist[i])))
                obs = nobs
            last_val = policy.predict_values(obs).squeeze(-1)

            # ---- GAE ----
            adv = torch.zeros_like(rew_buf)
            gae = torch.zeros(B, device=dev)
            nxt_val = last_val
            for t in reversed(range(T)):
                nd = (~done_buf[t]).float()
                delta = rew_buf[t] + args.gamma * nxt_val * nd - val_buf[t]
                gae = delta + args.gamma * args.gae_lambda * nd * gae
                adv[t] = gae
                nxt_val = val_buf[t]
            ret = adv + val_buf

        # ---- update (SB3-equivalent losses) ----
        N = T * B
        b_obs = obs_buf.reshape(N, obs_dim)
        b_act = act_buf.reshape(N)
        b_logp = logp_buf.reshape(N)
        b_adv = adv.reshape(N)
        b_ret = ret.reshape(N)
        for _ in range(args.n_epochs):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, args.batch_size):
                idx = perm[s:s + args.batch_size]
                if graph_step is not None:
                    graph_step(b_obs[idx], b_act[idx], b_logp[idx], b_adv[idx], b_ret[idx])
                    continue
                values, logp, entropy = policy.evaluate_actions(b_obs[idx], b_act[idx])
                a_mb = b_adv[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                ratio = torch.exp(logp - b_logp[idx])
                pl = -torch.min(a_mb * ratio,
                                a_mb * ratio.clamp(1 - args.clip_range, 1 + args.clip_range)).mean()
                vl = torch.nn.functional.mse_loss(values.squeeze(-1), b_ret[idx])
                loss = pl + args.vf_coef * vl - args.ent_coef * entropy.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                opt.step()

        gstep += N
        if gstep >= next_ckpt:
            model.save(f"{args.out}_{gstep}_steps")
            next_ckpt += args.ckpt_every
            recent = ep_stats[-200:]
            if recent:
                md = float(np.median([d for _, _, d in recent]))
                ml = float(np.median([l for _, l, _ in recent]))
                sps = gstep / (time.perf_counter() - t0)
                print(f"[{gstep:>8}] ep_dist~{md:.0f} ep_len~{ml:.0f} | {sps:,.0f} steps/s",
                      flush=True)

    model.save(f"{args.out}_final")
    with open(f"{args.out}_final.json", "w") as f:
        json.dump({"frame_stack": FS, "action_history_len": K, "net_arch": net_arch}, f)
    sps = gstep / (time.perf_counter() - t0)
    print(f"done: {gstep} steps at {sps:,.0f} steps/s -> saved {args.out}_final.zip", flush=True)


if __name__ == "__main__":
    main()
