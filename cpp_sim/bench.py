"""Throughput benchmark for the C++ sim, and the end-to-end training projection.

Measures the FULL step -- physics + track generation + snapshot + 37-wide obs
build + frame stack + action history + reward shaping -- i.e. exactly what
``train_fast.py`` pays per env-step, not a subset.

    G:/python312/python.exe cpp_sim/bench.py

A training run may be competing for the CPU, so every cell is the median of 3
short timings and the headline is the median across env counts, not the peak.
"""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent))

from cpp_sim.cpp_vec_env import CppTempleRunVecEnv, _Config, _Base, lib  # noqa: E402

# the device-true training config (what the current recipes run)
CFG = {
    "wide_gap_landing_model": True, "crevasse_apex_all": True,
    "crevasse_graze_worldy": True, "hole_landing_model": True,
    "hole_trap_kill_prob": 0.35, "stepup_clip_band": 0.25,
    "hook_turn_buffer": True, "turn_approach_buffer": True,
    "jump_exec_lat_frames": 1, "jump_exec_lat_jitter": 1,
    "strafe_exec_lat_frames": 1, "domain_rand": True,
    "device_difficulty_ramp": True, "crevasse_single_cell_prob": 0.675,
}
SHAPING = dict(action_cost=0.02, reaction_bonus=0.1, idle_action_penalty=0.05,
               strafe_penalty=0.3, narrow_hold_bonus=0.18, aim_center_weight=1.0,
               slide_jump_penalty=0.15)

UPDATE_MS = 1.27 * 32      # PPO update per iteration: 32 minibatches x 1.27 ms
SAMPLES = 256 * 16         # num_envs x n_steps = samples per iteration


def med3(fn, target=0.7):
    out = []
    for _ in range(3):
        fn()                      # warm
        t0 = time.perf_counter()
        n = 0
        while time.perf_counter() - t0 < target:
            n += fn()
        out.append(n / (time.perf_counter() - t0))
    out.sort()
    return out[1]


def bench_cpp(n_envs, n_steps=64):
    """Full sim_step over n_envs, obs + reward written out, vec auto-reset."""
    conf = _Config(CFG, "procedural", 0.1, 8000, 0.1, "distance", 1.0, 4, 1, SHAPING)
    core = _Base(n_envs, conf, 1234)
    rng = np.random.default_rng(0)
    # a plausible action mix (the actual actions do not change the cost much;
    # what matters is that episodes stay long enough to generate track normally)
    acts = rng.integers(0, 7, size=(n_steps, n_envs)).astype(np.int32)
    acts[:] = np.where(rng.random(acts.shape) < 0.55, 6, acts)   # mostly NoOp

    def run():
        for t in range(n_steps):
            core._step_raw(acts[t], 1)
        return n_steps * n_envs

    return med3(run)


def bench_python(n_steps=4000):
    from templerun_rl.sim_env import TempleRunSimEnv
    env = TempleRunSimEnv(config=CFG, seed=0, max_episode_steps=8000,
                          reward_mode="distance", action_history_len=4,
                          frame_stack=1, **SHAPING)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 7, size=n_steps)
    acts = np.where(rng.random(n_steps) < 0.55, 6, acts)
    t0 = time.perf_counter()
    n = 0
    for a in acts:
        _, _, term, trunc, _ = env.step(int(a))
        n += 1
        if term or trunc:
            env.reset()
    return n / (time.perf_counter() - t0)


def bench_torch(n_envs=256, n_steps=64):
    try:
        import torch
        from templerun_rl.sim_env_torch import TorchTempleRunVecEnv
    except Exception as e:      # noqa: BLE001
        return None, f"unavailable ({e})"
    try:
        venv = TorchTempleRunVecEnv(n_envs, config=CFG, device="cpu", seed=0,
                                    max_episode_steps=8000, reward_scale=0.1,
                                    reward_mode="distance", death_penalty=1.0,
                                    action_history_len=4, frame_stack=1, **SHAPING)
        venv.reset_tensor()
        a = torch.randint(0, 7, (n_envs,))

        def run():
            for _ in range(n_steps):
                venv.step_tensor(a)
            return n_steps * n_envs
        return med3(run, target=0.5), "cpu"
    except Exception as e:      # noqa: BLE001
        return None, f"failed ({e})"


def e2e(sim_sps):
    roll_ms = SAMPLES / sim_sps * 1000.0
    return SAMPLES / (roll_ms + UPDATE_MS) * 1000.0, roll_ms


def main():
    print("=" * 78)
    print("C++ sim throughput -- FULL step (physics + generation + obs + reward)")
    print("  config: device-true death models + ramp + domain_rand, K=4, FS=1")
    print("=" * 78)

    py_sps = bench_python()
    print(f"  Python reference  env.step()      : {py_sps:10.0f} env-steps/s "
          f"({1e6/py_sps:8.2f} us/step)")

    results = {}
    for n in (1, 64, 256, 1024, 4096):
        sps = bench_cpp(n)
        results[n] = sps
        print(f"  C++  {n:5d} envs                    : {sps:10.0f} env-steps/s "
              f"({1e6/sps:8.3f} us/step)   -> {sps/py_sps:6.0f}x Python")

    vals = sorted(results.values())
    head = vals[len(vals) // 2]
    print(f"\n  spread across env counts (CPU contention): "
          f"{min(vals)/1e6:.2f} - {max(vals)/1e6:.2f} M steps/s")
    print(f"  HEADLINE (median across env counts)      : {head:10.0f} env-steps/s")

    t_sps, t_note = bench_torch()
    if t_sps:
        print(f"  torch sim (CPU, 256 envs)                : {t_sps:10.0f} env-steps/s")
    else:
        print(f"  torch sim (CPU)                          : {t_note}")

    print("\n" + "=" * 78)
    print("END-TO-END TRAINING PROJECTION  (256 envs x 16 steps = 4096 samples/iter,")
    print("PPO update = 32 minibatches x 1.27 ms = 40.6 ms/iter)")
    print("=" * 78)
    rows = [("torch GPU sim (measured previously)", 14600.0),
            ("Python CPU sim (measured now)", py_sps),
            ("C++ sim (measured now)", head)]
    if t_sps:
        rows.insert(1, ("torch sim on CPU (measured now)", t_sps))
    base = e2e(14600.0)[0]
    for label, sps in rows:
        tot, roll = e2e(sps)
        print(f"  {label:<36} rollout {roll:8.2f} ms + update {UPDATE_MS:5.1f} ms"
              f"  => {tot:9.0f} steps/s e2e  ({tot/base:5.2f}x)")
    tot, roll = e2e(head)
    print(f"\n  => projected end-to-end speedup vs the torch GPU sim: {tot/base:.1f}x")
    print(f"     the PPO update is now {100.0*UPDATE_MS/(roll+UPDATE_MS):.0f}% of the "
          f"iteration, so the rollout is no longer the wall.")
    print("\n  SENSITIVITY (the projection is update-bound past ~1M steps/s):")
    for sps in (2e5, 5e5, 1e6, 2e6, 5e6):
        tot, roll = e2e(sps)
        print(f"     C++ at {sps/1e6:5.2f} M steps/s -> {tot:8.0f} steps/s e2e "
              f"({tot/base:4.1f}x), update = {100*UPDATE_MS/(roll+UPDATE_MS):3.0f}% of iter")


if __name__ == "__main__":
    main()
