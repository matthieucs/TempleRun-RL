#!/usr/bin/env python3
"""Acceptance gate for the env — validates it is usable to train, WITHOUT training.

Requires the device to be up (scripts/setup_device.py) with the game in a run or on the
game-over screen. Runs gymnasium's env checker, a short random rollout, and a few forced
resets. Exits non-zero on any failure.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys

import numpy as np

import templerun_rl  # noqa: F401
from templerun_rl import TempleRunEnv


def main() -> int:
    print("[1/4] constructing env + connecting to device...")
    try:
        env = TempleRunEnv()
    except RuntimeError as e:
        print("\nFAILED to connect:\n", e)
        print("\nPreflight checklist:")
        print("  - phone connected, USB debugging on")
        print("  - python scripts/setup_device.py is running (3 forwards + 'ready: ...')")
        print("  - the game is in the foreground, in a run or on the game-over screen")
        return 2

    try:
        print("[2/4] gymnasium check_env (space/contract conformance)...")
        from gymnasium.utils.env_checker import check_env

        # check_env runs its own reset/step against the live device.
        check_env(env, skip_render_check=True)
        print("      check_env OK")

        print("[3/4] 50 random steps...")
        obs, info = env.reset()
        assert env.observation_space.contains(obs), "reset obs not in observation_space"
        total = 0.0
        max_score = info.get("score", 0)
        for i in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs), f"step {i}: obs out of space"
            assert np.isfinite(reward), f"step {i}: non-finite reward {reward}"
            total += reward
            max_score = max(max_score, info.get("score", 0))
            if terminated or truncated:
                obs, info = env.reset()
        print(f"      ok — cumulative reward {total:.2f}, max score seen {max_score}")

        print("[4/4] 3 forced resets each yield a fresh run...")
        for r in range(3):
            obs, info = env.reset()
            st = info["raw_state"]
            assert not st["is_game_over"] and not st["is_dead"], f"reset {r}: not a fresh run: {st}"
            print(f"      reset {r}: fresh run (score={st['score']}, velocity={st['velocity']:.0f})")
    except AssertionError as e:
        print("\nFAILED:", e)
        return 1
    finally:
        env.close()

    print("\nSMOKE TEST PASSED — the env is usable to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
