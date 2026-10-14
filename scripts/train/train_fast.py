#!/usr/bin/env python3
"""FAST PPO — minimal pure-torch PPO over the batched sim, fully GPU-resident.

Vertical integration: the rollout (obs/actions/rewards/GAE) and the update loop all stay
torch tensors on one device — no numpy round-trips, no SB3 rollout plumbing. The POLICY is
still a stock SB3 MlpPolicy inside a stock PPO object, and saving goes through model.save()
— so checkpoints remain normal SB3 zips, loadable by every eval/deploy tool.

Loss/GAE formulas mirror SB3 PPO defaults exactly (clipped surrogate, plain value MSE,
entropy bonus, grad-clip 0.5, timeout bootstrapping). Validate any recipe change by curve
comparison against scripts/train/train_sim_gpu.py at equal steps (scripts/validate/eval_ckpt_torch.py).

    python scripts/train/train_fast.py --timesteps 2000000 ... (same recipe flags)
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

import templerun_rl.constants as C  # noqa: E402
from templerun_rl.sim_env_torch import TorchTempleRunVecEnv  # noqa: E402


def _make_vec_env_cls(backend: str):
    """Pick the rollout backend. Both expose the SAME surface train_fast.py uses:
    reset_tensor / step_tensor (8-tuple, matching shapes AND dtypes) / approach_turn_hint /
    observation_space / action_space / num_envs, so nothing downstream changes.

    'torch' = src/templerun_rl/sim_env_torch.py, the batched-GPU port (the default; every
              recipe up to and including v102 was trained on it).
    'cpp'   = cpp_sim/, a step-exact C++ port of the CPU reference. Gated at 5616/5616
              deterministic + all 8 statistical bands + 60/60 RNG draw-order lockstep.
              The sim runs on the CPU and the tensors are moved to --device.
    """
    if backend == "torch":
        return TorchTempleRunVecEnv
    if backend == "cpp":
        from cpp_sim.cpp_vec_env import CppTempleRunVecEnv
        return CppTempleRunVecEnv
    raise SystemExit(f"unknown --backend {backend!r} (expected 'torch' or 'cpp')")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    # [MEASURED 2026-07-27] 1024x4 replaces 256x16 = 2.9x FASTER at IDENTICAL learning.
    # The trainer is LAUNCH-BOUND, not compute-bound: the sim step is 96% of wall-clock and issues
    # ~2000 tiny kernels, so one CPU core saturates issuing them while the GPU idles at ~13%. Step
    # cost is therefore nearly FLAT in num_envs (70.7 / 68.1 / 70.9 / 77.9 ms at 256 / 512 / 1024 /
    # 2048), so widening the batch is close to free.
    # ⚠️ KEEP num_envs * n_steps == 4096. Widening num_envs ALONE (1024x16) is 3x FASTER but 3x WORSE:
    # median @3M 1036 vs 3262. Total gradient updates are identical either way — what breaks is
    # STALENESS: a 16384-sample rollout applies 128 sequential updates to one policy snapshot, so most
    # get clipped away. The invariant that matters is updates per DATA COLLECTION, not per sample.
    # Holding the 4096-transition rollout fixed keeps the minibatch count, the update count and the
    # staleness identical, and only shortens the GAE horizon (16 -> 4).
    # VALIDATED at matched steps (n=128): @3M 3258 (1024x4) vs 3262 (256x16) = parity, inside the
    # +-10% eval noise floor. @1M it lags (1666 vs 2546) — the shorter horizon learns value more
    # slowly early — then catches up. Re-check parity at 10M on the next long run.
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--n-steps", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--net-arch", default="256,256")
    p.add_argument("--action-history-len", type=int, default=4)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--turn-linger", type=int, default=3)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef-final", type=float, default=None,
                   help="ANNEAL the entropy coefficient --ent-coef -> this value over training "
                        "(USER 2026-07-27: 'you increase the entropy and you put a schedule on its "
                        "coefficient'). High early entropy re-opens exploration when the WORLD has "
                        "changed under a warm-started policy -- v106 sat flat at ~1050 for 8M steps "
                        "because a policy committed to old habits never sampled the alternatives. "
                        "Annealing keeps the back half on-policy and greedy, like the LR cosine.")
    p.add_argument("--ent-schedule", choices=["constant", "linear", "cosine"], default="constant")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--lr-schedule", choices=["constant", "linear", "cosine"], default="constant",
                   help="LR decay over training: constant | linear | cosine (to --lr-final)")
    p.add_argument("--lr-final", type=float, default=None,
                   help="end LR for the schedule (default = --learning-rate = constant)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--gae-lambda-final", type=float, default=None,
                   help="RAMP the credit horizon over training (USER 2026-07-28: 'why should the "
                        "timesteps be the same at the beginning and at the end?'). Correct point, but "
                        "n_steps is the WRONG dial for it: num_envs is fixed at construction and "
                        "batch-size is capped at 512, so ramping n_steps changes num_envs*n_steps = "
                        "updates per DATA COLLECTION -- the one invariant measured to matter (1024x16 "
                        "gave @3M median 1036 vs 3262 at identical update COUNT). The effective credit "
                        "horizon is gamma*lambda, so ramping lambda lengthens credit assignment with "
                        "the rollout geometry untouched. Motivation: episode length triples during a "
                        "run (ep_len ~260 -> 800+), so early on a long horizon mostly bootstraps from "
                        "truncation while late on a short one under-credits.")
    p.add_argument("--gae-schedule", choices=["constant", "linear", "cosine"], default="constant")
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
    p.add_argument("--jump-hold-penalty", type=float, default=0.0,
                   help="dock a HELD jump/slide (JUMP-after-JUMP) -> kills the seamless landing-frame "
                        "re-jump that dies on high-ground (device-decoded 2026-07-08)")
    p.add_argument("--velocity-drop-penalty", type=float, default=0.0,
                   help="USER 2026-07-27: 'the mid->high transition is not fixed, the policy jumps way "
                        "too early. Have you modelled the case when the character takes a hit and is "
                        "slowed down? This should be penalized. It is clearly not.' CORRECT — the "
                        "reward term has existed in reward.py since the crevasse wave but was NEVER "
                        "plumbed into this trainer, so no recipe has ever used it. The sim DOES model "
                        "the wall-clip (crevasse_graze_worldy / stepup_clip_band -> Stumble, x0.85 "
                        "velocity) and already grazes HARDER than the device, so the physics was never "
                        "the gap: a graze costs only a few units of distance, which is far below what "
                        "PPO chases — the same weak-signal failure the held-jump death model hit on "
                        "2026-07-27 until an explicit penalty amplified it. Docks the velocity LOSS "
                        "beyond a 3.0 deadband, so it prices the graze directly.")
    p.add_argument("--turn-buffer-bonus", type=float, default=0.0,
                   help="WAVE 13: reward a valid-direction PRE-SWIPE on the approach cell (the "
                        "latency-robust turn latch). Aligns the dense reward with survival so the "
                        "policy discovers the pre-swipe instead of the dead-lettering junction swipe. "
                        "Pair with hook_turn_buffer + turn_approach_buffer + turn_send_lat_frames.")
    p.add_argument("--turn-approach-buffer", action="store_true", default=False)
    p.add_argument("--landing-hazard-penalty", type=float, default=0.0,
                   help="dock a JUMP whose ARC is predicted to land on a must-jump hazard. AMPLIFIER "
                        "for a cue that is already correct but too quiet: v107 dies 5-of-7 on dt=4 "
                        "Ledge, its fatal takeoff had landing_hazard=1.0 already set, yet flipping "
                        "that cue moves P(JUMP) only -0.0039 on a 0.036 base. Same shape as the "
                        "held-jump fix (death model + airborne_action_penalty -> 10.67 to 0.00/km).")
    p.add_argument("--stepdown-explore-prob", type=float, default=0.0,
                   help="exploring-starts for the foot-crossable step-down: force NoOp with this "
                        "probability (annealed) so the not-jumping counterfactual is SAMPLED. "
                        "Needed because at a 100%% jump rate the penalty has no gradient.")
    p.add_argument("--stepdown-explore-anneal-frac", type=float, default=0.5)
    p.add_argument("--stack-margin-penalty", type=float, default=0.0,
                   help="USER-diagnosed: reaction_bonus pays the SAME for jumping an isolated "
                        "obstacle and a STACKED one, so the policy reuses one trajectory where a "
                        "different one is needed. Docks a takeoff whose arc lands within 1 cell of "
                        "the NEXT hazard, scoped to the stacked case only.")
    p.add_argument("--margin-bonus", type=float, default=0.0,
                   help="grade every hazard-spanning takeoff by the room it leaves. Distance reward "
                        "is identical for landing 1.65 cells clear and 0.35 clear, so the policy only "
                        "learns margin from the 0.03%% of frames where it dies. Dense signal on ~20%%.")
    p.add_argument("--stepdown-jump-penalty", type=float, default=0.0,
                   help="dock a JUMP at a step-DOWN to lower ground when no hard must-jump requires "
                        "it. USER-requested. Foot-crossable (human walks 58%%, never dies) yet every "
                        "policy jumps ~100%%; the jump launches a 2.5-3 cell arc that can end on a "
                        "hazard. Discretionary, so it does not repeat the forced-action trap.")
    p.add_argument("--slide-landing-hazard-penalty", type=float, default=0.0,
                   help="dock a JUMP whose ARC is predicted to land on a must-SLIDE hazard. The obs "
                        "cue alone (v117, obs66) IS learned -- dP(JUMP) -0.00450 @2M -> -0.00631 "
                        "@10M, no erosion of landing_hazard -- but moved NO outcome, which is "
                        "exactly the correct-but-quiet signature the held-jump and landing_hazard "
                        "fixes both had. Device: the cue carries a 52.6x death-rate lift (15.69%% vs "
                        "0.30%% over 26,854 takeoffs) and survivors take off at cp 0.53 vs victims' "
                        "0.62, so the earlier takeoff this prices is available and is the separator.")
    p.add_argument("--turn-takeoff-penalty", type=float, default=0.0,
                   help="dock a JUMP whose ARC is still airborne when an unresolved JUNCTION "
                        "arrives. USER: 'it jumped in a turn, turned and landed in a hole.' "
                        "MEASURED: 5.0%% of takeoffs have a junction within 2 cells and 8.7%% of "
                        "THOSE kill, vs ~1%% for takeoffs generally. STRUCTURAL, not a training "
                        "gap: the lookahead is computed along the CURRENT heading with a junction "
                        "dropout, so a turn firing mid-air lands the runner on a path the obs never "
                        "modelled -- exposure cannot teach a landing whose cue does not exist. "
                        "Pricing the TAKEOFF is the learnable half (is_turn / next_is_turn are both "
                        "in the obs). Third use of the amplifier pattern.")
    p.add_argument("--stepdown-bonus-gate", action="store_true", default=False,
                   help="stop paying reaction_bonus for a JUMP whose only must_jump is a SOFT "
                        "step-down to low ground (needs sim stepdown_grounded_ok). Removing the death "
                        "removed the REQUIREMENT to jump there; this removes the INCENTIVE. v103 "
                        "showed the world fix alone moved P(jump|step-down) 98.6%% -> 98.5%% = nothing.")
    p.add_argument("--turn-explore-prob", type=float, default=0.0,
                   help="WAVE 13 GUIDED EXPLORATION (exploring-starts, still 100%% RL): on an approach "
                        "cell, force the correct-direction pre-swipe with this prob during rollout so "
                        "turn_buffer_bonus fires and its advantage propagates (the approach-turn prob is "
                        "collapsed to ~0 and pure entropy never explores it). Anneals to 0.")
    p.add_argument("--cascade-explore-prob", type=float, default=0.0,
                   help="GUIDED EXPLORATION for OBSTACLE CASCADES (2026-07-28) -- the device that broke "
                        "the turn dead-letter, applied to the last remaining killer. MEASURED: v111 dies "
                        "on 9.62%% of 3-obstacle chains on DEVICE and 8.16%% in SIM = MATCHED, so the sim "
                        "is faithful in both rate and lethality and the policy simply cannot do them "
                        "(HUMAN: 0 of 45). The survivable takeoff exists but carries ~no probability, so "
                        "it is never sampled and no reward can reinforce it -- exactly the shape that "
                        "pinned P(turn|approach) at 0.000000 through five arms. Forces the oracle takeoff "
                        "(the one whose arc lands on clear ground) with prob eps, annealed to 0 so the "
                        "back half is pure on-policy and the policy must SUSTAIN it unforced.")
    p.add_argument("--cascade-explore-anneal-frac", type=float, default=0.5)
    p.add_argument("--turn-explore-anneal-frac", type=float, default=0.5,
                   help="fraction of training over which --turn-explore-prob linearly decays to 0 "
                        "(then the policy must sustain the pre-swipe on its own = pure on-policy RL).")
    p.add_argument("--curriculum-max-start-dist", type=float, default=0.0)
    p.add_argument("--domain-rand", dest="domain_rand", action="store_true", default=True)
    p.add_argument("--no-domain-rand", dest="domain_rand", action="store_false")
    p.add_argument("--sim-config", default="")
    p.add_argument("--backend", default="torch", choices=("torch", "cpp"),
                   help="rollout backend. 'torch' = the batched-GPU sim (default, what every recipe "
                        "up to v102 trained on). 'cpp' = the step-exact C++ port (cpp_sim/), ~170-340x "
                        "the Python reference on the sim itself; the rollout stops being the wall so "
                        "--num-envs / --n-steps cost only PPO time. Build it first: sh cpp_sim/build.sh")
    p.add_argument("--ckpt-every", type=int, default=100_000)
    p.add_argument("--cuda-graph", action="store_true", default=False,
                   help="capture the PPO minibatch step as a CUDA graph (one launch per bite "
                        "instead of ~30; bit-identical math, pure dispatch optimization). "
                        "Requires --device cuda and total buffer divisible by --batch-size.")
    p.add_argument("--out", default="runs/ppo_fast")
    args = p.parse_args()

    dev = torch.device(args.device)
    # entropy coefficient as a DEVICE TENSOR: under --cuda-graph the minibatch step is captured
    # once, so a Python float would be baked in and the schedule would silently never move.
    # Same trick as the LR tensor.
    _ent_now = torch.tensor(0.0, device=dev)
    cfg = {"domain_rand": args.domain_rand, "turn_approach_buffer": args.turn_approach_buffer,
           "turn_linger_frames": args.turn_linger,
           "curriculum_max_start_dist": args.curriculum_max_start_dist}
    if args.sim_config:
        cfg.update(json.loads(args.sim_config))
    K, FS = args.action_history_len, args.frame_stack
    VecEnvCls = _make_vec_env_cls(args.backend)
    venv = VecEnvCls(
        args.num_envs, config=cfg, device=args.device, seed=args.seed,
        max_episode_steps=args.max_episode_steps, reward_scale=args.reward_scale,
        reward_mode="distance", death_penalty=args.death_penalty,
        action_cost=args.action_cost, reaction_bonus=args.reaction_bonus,
        turn_spam_penalty=args.turn_spam_penalty, idle_action_penalty=args.idle_action_penalty,
        strafe_penalty=args.strafe_penalty, narrow_hold_bonus=args.narrow_hold_bonus,
        aim_center_weight=args.aim_center_weight, slide_jump_penalty=args.slide_jump_penalty,
        airborne_action_penalty=args.airborne_action_penalty,
        jump_hold_penalty=args.jump_hold_penalty,
        velocity_drop_penalty=args.velocity_drop_penalty,
        turn_buffer_bonus=args.turn_buffer_bonus,
        stepdown_bonus_gate=args.stepdown_bonus_gate,
        landing_hazard_penalty=args.landing_hazard_penalty,
        slide_landing_hazard_penalty=args.slide_landing_hazard_penalty,
        stepdown_jump_penalty=args.stepdown_jump_penalty,
        margin_bonus=args.margin_bonus,
        stack_margin_penalty=args.stack_margin_penalty,
        turn_takeoff_penalty=args.turn_takeoff_penalty,
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
            loss = pl + args.vf_coef * vl - _ent_now * entropy.mean()
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

    def _sched_ent(g: int) -> float:
        if args.ent_schedule == "constant" or args.ent_coef_final is None:
            return float(args.ent_coef)
        prog = min(1.0, g / max(1, args.timesteps))
        if args.ent_schedule == "linear":
            return float(args.ent_coef + (args.ent_coef_final - args.ent_coef) * prog)
        return float(args.ent_coef_final + 0.5 * (args.ent_coef - args.ent_coef_final)
                     * (1.0 + math.cos(math.pi * prog)))   # cosine

    def _sched_gae(g: int) -> float:
        if args.gae_schedule == "constant" or args.gae_lambda_final is None:
            return float(args.gae_lambda)
        prog = min(1.0, g / max(1, args.timesteps))
        if args.gae_schedule == "linear":
            return float(args.gae_lambda + (args.gae_lambda_final - args.gae_lambda) * prog)
        return float(args.gae_lambda_final + 0.5 * (args.gae_lambda - args.gae_lambda_final)
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
        _ent_now.fill_(_sched_ent(gstep))   # ditto the entropy coefficient (read inside the graph)
        _lam_now = _sched_gae(gstep)        # credit horizon for THIS iteration's GAE pass
        # ---- rollout (no_grad, all tensors on device) ----
        with torch.no_grad():
            for t in range(T):
                acts, values, logp = policy(obs)
                # GUIDED EXPLORATION (WAVE 13): force the approach-cell pre-swipe with prob eps (annealed
                # to 0) so turn_buffer_bonus is actually sampled. Store the EXECUTED action + its policy
                # log-prob so PPO's ratio stays consistent (clipping absorbs the off-policy injection).
                if args.turn_explore_prob > 0.0:
                    frac = args.turn_explore_anneal_frac * args.timesteps
                    eps = args.turn_explore_prob * max(0.0, 1.0 - gstep / max(1.0, frac))
                    if eps > 0.0:
                        hint = venv.approach_turn_hint()
                        inj = (hint >= 0) & (torch.rand(B, device=dev) < eps)
                        if bool(inj.any()):
                            acts = torch.where(inj, hint, acts)
                            _, logp, _ = policy.evaluate_actions(obs, acts)
                # CASCADE guided exploration (2026-07-28): same mechanism, different oracle. Applied
                # AFTER the turn hint so a junction pre-swipe always wins (turns are latency-critical
                # and a missed one dead-letters). Same off-policy contract: store the EXECUTED action
                # with its policy log-prob and let PPO clipping absorb the injection.
                # STEP-DOWN guided exploration: force WALKING a foot-crossable step-down with prob
                # eps so stepdown_jump_penalty is actually sampled (at a 100% jump rate it never is).
                if args.stepdown_explore_prob > 0.0:
                    sfrac = args.stepdown_explore_anneal_frac * args.timesteps
                    seps = args.stepdown_explore_prob * max(0.0, 1.0 - gstep / max(1.0, sfrac))
                    if seps > 0.0:
                        shint = venv.stepdown_walk_hint()
                        sinj = (shint >= 0) & (torch.rand(B, device=dev) < seps)
                        if bool(sinj.any()):
                            acts = torch.where(sinj, shint, acts)
                            _, logp, _ = policy.evaluate_actions(obs, acts)
                if args.cascade_explore_prob > 0.0:
                    cfrac = args.cascade_explore_anneal_frac * args.timesteps
                    ceps = args.cascade_explore_prob * max(0.0, 1.0 - gstep / max(1.0, cfrac))
                    if ceps > 0.0:
                        chint = venv.cascade_hint()
                        cinj = (chint >= 0) & (torch.rand(B, device=dev) < ceps)
                        if bool(cinj.any()):
                            acts = torch.where(cinj, chint, acts)
                            _, logp, _ = policy.evaluate_actions(obs, acts)
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
                gae = delta + args.gamma * _lam_now * nd * gae
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
                loss = pl + args.vf_coef * vl - _ent_now * entropy.mean()
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
