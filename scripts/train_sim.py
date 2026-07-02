#!/usr/bin/env python3
"""Train a PPO policy in the headless sim (``TempleRunSim-v0``) — fast, parallel, free.

The sim mirrors the real env's obs/action spaces (Box(OBS_LEN)/Discrete(7)) and reward (distance-delta),
so a policy trained here is drop-in on the phone (``scripts/evaluate.py`` / deploy). The sim runs at
~10k+ steps/s, so we vectorize across cores and train millions of steps in minutes.

    python scripts/train_sim.py --timesteps 3000000 --num-envs 8
    python scripts/train_sim.py --timesteps 1000000 --reaction-bonus 0.1 --ent-coef 0.01

Saves runs/ppo_sim_final.zip + checkpoints; TensorBoard under runs/tb_sim.
"""
from __future__ import annotations

import argparse
import json
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
            step_dt=args.step_dt,
            reward_mode="distance",
            reward_scale=args.reward_scale,
            action_delay=args.action_delay,
            death_penalty=args.death_penalty,
            action_cost=args.action_cost,
            reaction_bonus=args.reaction_bonus,
            turn_spam_penalty=args.turn_spam_penalty,
            idle_action_penalty=args.idle_action_penalty,
            redundant_action_penalty=args.redundant_action_penalty,
            strafe_penalty=args.strafe_penalty,
            action_history_len=args.action_history_len,
            frame_stack=args.frame_stack,
            config={"domain_rand": bool(args.domain_rand),
                    "turn_linger_frames": int(args.turn_linger),
                    **(json.loads(args.sim_config) if args.sim_config else {})},
            seed=args.seed + rank,
        )
        return env
    return _factory


# Lane geometry (center-0 tilt model, mirrors src/templerun_rl/obs.py): one strafe ~ LANE_STEP, walls
# at ~+-LANE_EDGE. The oracle strafes toward a narrow's safe lane (target = ntl * LANE_STEP) and NEVER
# strafes on a straight — the lane auto-recenters, so gratuitous strafing only risks a wall-stumble.
_LANE_STEP = 3.7
_NARROW_CENTER = 4.5   # [MEASURED 2026-07-02] long-narrow strip center (Region2D bands [2,7]/[3,7])

# RISING-EDGE turn memory (single sequential env). The oracle turns only when is_turn goes
# False->True, so under turn_linger (is_turn HELD True N frames post-turn to model the device
# facing-lag double-turn temptation) it fires ONCE per junction — never re-turns during the linger.
# Reset on episode boundary (distance drops). This is what the trained policy must learn via K=4.
_oracle_mem = {"was_turn": False, "prev_dist": 0.0}


def _oracle(st: dict) -> int:
    # TURN has priority over the lookahead-jump: a junction with an obstacle in the cell AFTER it
    # (is_turn=1 AND next_must_jump=1) must be TURNED, not jumped — jumping misses the turn -> Fall.
    # (The generation keeps the cell right after a junction obstacle-free, so turning is always safe.)
    # TURN PROMPTLY, ONLY on the is_turn flash (device-remodel 2026-07-01, corrected after a device
    # eval): is_turn is a ~1-frame flash at cell start — swipe the instant it appears (early WITHIN the
    # junction cell is correct, no off-center death). Do NOT pre-fire on next_is_turn: there is no safe
    # early buffer on device — a swipe before the junction hits the wall (stumble) and, repeated, turns
    # you off the path. Pre-firing was a sim exploit that regressed the device (226 vs 736). Turn once.
    d = float(st.get("distance", 0.0))
    if d < _oracle_mem["prev_dist"] - 20.0:       # new episode -> reset the edge memory
        _oracle_mem["was_turn"] = False
    _oracle_mem["prev_dist"] = d
    rising = bool(st["is_turn"]) and not _oracle_mem["was_turn"]
    _oracle_mem["was_turn"] = bool(st["is_turn"])
    if rising:                                    # turn ONCE on the junction (edge, not the held linger)
        if st["can_turn_left"]:
            return int(C.Action.TURN_LEFT)
        if st["can_turn_right"]:
            return int(C.Action.TURN_RIGHT)
        return int(C.Action.TURN_LEFT)
    if st["is_turn"]:
        # is_turn still True but NOT a fresh junction = post-turn linger. Do NOT re-turn (that's the
        # fatal double-turn). Fall through to jump/slide/lane so the oracle stays busy + centered.
        pass
    # WIDE gap (2-cell mid<->high crevasse / long hole): jump LATE — at the gap edge (high cell_progress
    # on the pre-gap cell) so the airtime reserves enough to be airborne crossing into the FAR cell.
    # Jumping early (the single-obstacle reflex) runs out of air by the 2nd cell -> Fall. 0.62 is the
    # measured sweep optimum at crevasse speed (lower wastes airtime early; >0.65 skips the window at
    # the ~0.38-cprog/step top-speed stepping). See sim_env wide_gap_extra_cells / sim-calibration memory.
    if st.get("next_wide_gap"):
        return int(C.Action.JUMP) if float(st.get("cell_progress", 1.0)) >= 0.62 else int(C.Action.NOOP)
    # Anticipate: jump/slide on the LOOKAHEAD (obstacle is the NEXT cell) so the action is active
    # when entering the obstacle cell (the sim resolves obstacles on entry).
    if st["must_jump"] or st.get("next_must_jump"):
        return int(C.Action.JUMP)
    if st["must_slide"] or st.get("next_must_slide"):
        return int(C.Action.SLIDE)
    # LANE (ANALOG TILT, 2026-07-02): a strafe byte = HOLD toward +/-edge (first-order lag), NoOp =
    # LEVEL = the game recenters. Center needs NO action; an off-center target (narrow strip center
    # +/-4.5, Region2D-measured) is held by bang-bang DITHER (hold past it / release short of it)
    # since the only stable commanded points are -edge/0/+edge.
    ntl = st.get("narrow_target_lane")
    target = int(ntl) * _NARROW_CENTER if ntl is not None else 0.0
    lane = float(st.get("lane", 0.0))
    if lane < target - 0.4 * _LANE_STEP:
        return int(C.Action.STRAFE_RIGHT)
    if lane > target + 0.4 * _LANE_STEP:
        return int(C.Action.STRAFE_LEFT)
    return int(C.Action.NOOP)


def _rollout(policy, n_eps=20, seed0=10000, action_history_len=0, frame_stack=1):
    """Run a policy (callable obs->action OR 'oracle'/'random') in a single sim; return survival."""
    env = gym.make("TempleRunSim-v0", max_episode_steps=100000,
                   action_history_len=action_history_len, frame_stack=frame_stack).unwrapped
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
    def __init__(self, every: int, action_history_len: int = 0, frame_stack: int = 1, verbose=1):
        super().__init__(verbose)
        self.every = every
        self.action_history_len = int(action_history_len)
        self.frame_stack = int(frame_stack)
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.every:
            return True
        self._last = self.num_timesteps
        model = self.model

        def greedy(obs):
            a, _ = model.predict(obs, deterministic=True)
            return int(a)
        ms, md, deaths = _rollout(greedy, n_eps=15, action_history_len=self.action_history_len,
                                  frame_stack=self.frame_stack)
        self.logger.record("eval/median_steps", ms)
        self.logger.record("eval/median_distance", md)
        self.logger.record("eval/deaths_per15", deaths)
        if self.verbose:
            print(f"[eval @ {self.num_timesteps}] median steps={ms:.0f} distance={md:.0f} "
                  f"deaths={deaths}/15", flush=True)
        return True


class RealPhoneEvalCallback(BaseCallback):
    """Every ``every`` steps, play the current greedy policy on the REAL phone for a few episodes
    and log real survival/distance/score to TensorBoard (eval_real/*). This is the sim-to-real
    check — you watch the real curve climb next to the sim curve. Robust: any phone hiccup is
    caught and the eval is skipped (training never dies for a flaky device)."""
    def __init__(self, every: int, serial: str, device_index: int, episodes: int,
                 max_steps: int, verbose=1):
        super().__init__(verbose)
        self.every = every
        self.serial = serial
        self.device_index = device_index
        self.episodes = episodes
        self.max_steps = max_steps
        self._last = 0
        self._env = None

    def _get_env(self):
        if self._env is None:
            import templerun_rl.constants as _C
            from templerun_rl.env import TempleRunEnv
            ports = _C.device_ports(self.device_index)
            self._env = TempleRunEnv(
                host="127.0.0.1", state_port=ports["state_port"], action_port=ports["action_port"],
                max_episode_steps=self.max_steps, reward_mode="distance", auto_connect=True,
                # fail fast if the phone is asleep/down so a real-eval can't stall training for long
                freeze_timeout=12.0, reset_timeout=6.0, connect_timeout=6.0)
        return self._env

    def _on_step(self) -> bool:
        if self.every <= 0 or self.num_timesteps - self._last < self.every:
            return True
        self._last = self.num_timesteps
        try:
            env = self._get_env()
            steps, dists, scores, deaths, frozen = [], [], [], 0, 0
            for _ in range(self.episodes):
                o, i = env.reset()
                st = i["raw_state"]
                d0 = float(st.get("distance", 0) or 0)
                s = 0
                degraded = False
                while True:
                    a, _ = self.model.predict(o, deterministic=True)
                    o, r, term, trunc, info = env.step(int(a))
                    st = info["raw_state"]
                    if info.get("degraded"):     # phone paused/frozen -> bail fast, don't burn 600 steps
                        degraded = True
                        break
                    s += 1
                    if term:
                        deaths += 1
                        break
                    if trunc:
                        break
                prog = float(st.get("distance", 0) or 0) - d0
                # A frozen/limbo run (degraded, or no distance progress) is NOT real survival —
                # the phone is paused (e.g. notification shade covering the game). Don't count it.
                if degraded or prog < 5.0:
                    frozen += 1
                    continue
                steps.append(s)
                dists.append(prog)
                scores.append(int(st.get("score", 0) or 0))
            if not dists:
                print(f"[REAL eval @ {self.num_timesteps}] FROZEN — topaz not running a real "
                      f"game ({frozen}/{self.episodes} limbo; check the screen/shade)", flush=True)
                self.logger.record("eval_real/frozen", float(frozen))
                self._env = None
                return True
            ms = float(np.median(steps))
            md = float(np.median(dists))
            self.logger.record("eval_real/median_steps", ms)
            self.logger.record("eval_real/median_distance", md)
            self.logger.record("eval_real/median_score", float(np.median(scores)))
            self.logger.record("eval_real/deaths", deaths)
            print(f"[REAL eval @ {self.num_timesteps}] topaz: median steps={ms:.0f} "
                  f"distance={md:.0f} score={int(np.median(scores))} deaths={deaths}/{self.episodes}",
                  flush=True)
        except Exception as e:  # noqa: BLE001  flaky phone must never kill training
            print(f"[REAL eval @ {self.num_timesteps}] SKIPPED ({type(e).__name__}: {e})", flush=True)
            self._env = None  # force reconnect next time
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
    p.add_argument("--redundant-action-penalty", type=float, default=0.0,
                   help="dock the trailing re-jump/re-slide (must_jump set but next_must_jump=0, "
                        "obstacle already entered) — curbs the 'double-jump'. try ~0.12")
    p.add_argument("--strafe-penalty", type=float, default=0.0,
                   help="HARD dock on any non-narrow strafe (narrow_target_lane is None). On device an "
                        "off-track strafe drives the continuous lane OOD (sim lane is 3 discrete values) "
                        "-> the policy walks off the edge. ~0.30 suppresses non-narrow strafing.")
    p.add_argument("--action-history-len", type=int, default=0,
                   help="append the one-hot of the last K actions to the obs (policy memory -> curbs "
                        "multiple-turns-at-a-junction). obs grows by K*7. try 3-6. needs a retrain.")
    p.add_argument("--frame-stack", type=int, default=1,
                   help="stack the last N base-26 obs frames (temporal context -> anticipate turns). "
                        "obs grows to N*OBS_LEN (+K*7). try 2-4. needs a retrain.")
    p.add_argument("--turn-linger", type=int, default=0,
                   help="sim-to-real: keep is_turn True N frames after a turn (a 2nd turn during them "
                        "= over-rotation death) so the policy learns to NOT double-turn. try 2. needs K>0.")
    p.add_argument("--sim-config", default="",
                   help="JSON dict of extra sim config overrides, e.g. '{\"dr_turn_window_frac\":[0.28,0.45]}'")
    p.add_argument("--step-dt", type=float, default=C.DEFAULT_STEP_DT,
                   help="control tick seconds. 0.05 = 20 Hz (halves the on-device turn lag). MUST match "
                        "the hook POLL_MS and eval --step-dt. Pair with turn_linger ~4 at 20 Hz.")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--action-delay", type=int, default=0,
                   help="steps to buffer the action before applying. EXPERIMENTAL/off: delay=1 "
                        "over-models the phone lag and breaks the missed-turn death (a buffered turn "
                        "lands after the junction is exited -> unfair, oracle dies). Keep 0.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--net-arch", default="64,64",
                   help="comma-separated hidden layer sizes for the MLP policy, e.g. '256,256'")
    p.add_argument("--domain-rand", dest="domain_rand", action="store_true", default=True,
                   help="per-episode jitter of turn window / durations / velocity / density for "
                        "sim-to-real robustness (default ON)")
    p.add_argument("--no-domain-rand", dest="domain_rand", action="store_false")
    p.add_argument("--out", default="runs/ppo_sim")
    # periodic REAL-phone eval (sim-to-real): play on the phone every N steps, log to TensorBoard
    p.add_argument("--real-eval-every", type=int, default=0,
                   help="play the policy on the real phone every N steps (0 = off)")
    p.add_argument("--real-eval-serial", default="e4de25a1", help="phone serial for real eval")
    p.add_argument("--real-eval-device-index", type=int, default=0)
    p.add_argument("--real-eval-episodes", type=int, default=2)
    p.add_argument("--real-eval-max-steps", type=int, default=600)
    args = p.parse_args()

    K, FS = args.action_history_len, args.frame_stack
    print(f"baselines: oracle {_rollout('oracle', 10, action_history_len=K, frame_stack=FS)}, "
          f"random {_rollout('random', 10, action_history_len=K, frame_stack=FS)}", flush=True)

    factories = [_make_env(args, i) for i in range(args.num_envs)]
    venv = SubprocVecEnv(factories) if args.num_envs > 1 else DummyVecEnv(factories)

    net_arch = [int(x) for x in str(args.net_arch).split(",") if x.strip()]
    print(f"policy net_arch={net_arch}", flush=True)
    model = PPO(
        "MlpPolicy", venv,
        n_steps=512, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        ent_coef=args.ent_coef, learning_rate=3e-4, clip_range=0.2,
        policy_kwargs=dict(net_arch=net_arch),
        tensorboard_log="runs/tb_sim", verbose=0,
    )
    ckpt = CheckpointCallback(save_freq=max(1, 100_000 // args.num_envs),
                              save_path="runs", name_prefix=Path(args.out).name)
    callbacks = [ckpt, EvalCallback(every=100_000, action_history_len=args.action_history_len,
                                    frame_stack=args.frame_stack)]
    if args.real_eval_every > 0:
        callbacks.append(RealPhoneEvalCallback(
            every=args.real_eval_every, serial=args.real_eval_serial,
            device_index=args.real_eval_device_index, episodes=args.real_eval_episodes,
            max_steps=args.real_eval_max_steps))
        print(f"REAL-phone eval every {args.real_eval_every} steps on {args.real_eval_serial}", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(f"{args.out}_final")
    # sidecar so eval/score know the obs shape (obs dim alone is ambiguous between frame_stack & K).
    import json
    with open(f"{args.out}_final.json", "w") as f:
        json.dump({"frame_stack": args.frame_stack, "action_history_len": args.action_history_len,
                   "net_arch": net_arch}, f)
    print(f"saved {args.out}_final.zip (+ .json: fs={args.frame_stack} K={args.action_history_len} "
          f"arch={net_arch})", flush=True)

    ms, md, deaths = _rollout(lambda o: int(model.predict(o, deterministic=True)[0]), n_eps=30,
                              action_history_len=args.action_history_len, frame_stack=args.frame_stack)
    print(f"FINAL policy: median steps={ms:.0f} distance={md:.0f} deaths={deaths}/30", flush=True)


if __name__ == "__main__":
    main()
