#!/usr/bin/env python3
"""Play a trained policy on the REAL phone and report survival/distance/score.

    python scripts/device/eval_real.py runs/ppo_sim_dr_final.zip --episodes 10

Robust to a flaky phone: a frozen/limbo run (no distance progress) is reported as such and skipped,
not counted as survival. Prints per-run lines + a summary so sim-to-real transfer is visible.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import templerun_rl  # noqa: F401,E402
import templerun_rl.constants as C  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from templerun_rl.env import TempleRunEnv  # noqa: E402

_ANAMES = {0: "SL", 1: "SR", 2: "JMP", 3: "SLD", 4: "TL", 5: "TR", 6: "NOP"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.add_argument("--turn-cooldown", type=int, default=3,
                   help="drop turn actions for N steps after a turn (kills the double-turn over-rotation)")
    p.add_argument("--step-dt", type=float, default=0.1,
                   help="control tick seconds; 0.05 = 20 Hz (match the hook POLL_MS + the model's train rate)")
    args = p.parse_args()

    model = PPO.load(args.model)
    # obs shape from the sidecar (frame_stack & action_history_len); fallback assumes frame_stack=1.
    import json
    import templerun_rl.obs as O
    side = Path(str(args.model).replace(".zip", "") + ".json")
    if side.exists():
        d = json.loads(side.read_text()); fs, K = int(d.get("frame_stack", 1)), int(d.get("action_history_len", 0))
    else:
        fs, K = 1, max(0, (int(model.observation_space.shape[0]) - O.OBS_LEN) // 7)
    ports = C.device_ports(args.device_index)
    env = TempleRunEnv(host="127.0.0.1", state_port=ports["state_port"],
                       action_port=ports["action_port"], max_episode_steps=args.max_steps,
                       reward_mode="distance", auto_connect=True, step_dt=args.step_dt,
                       frame_stack=fs, action_history_len=K, turn_cooldown=args.turn_cooldown,
                       freeze_timeout=20.0, reset_timeout=8.0, connect_timeout=8.0)
    print(f"obs: frame_stack={fs} action_history={K} (dim {model.observation_space.shape[0]}) "
          f"turn_cooldown={args.turn_cooldown}", flush=True)
    print(f"model={args.model}  episodes={args.episodes}  deterministic={args.deterministic}", flush=True)

    dists, scores, steps, deaths, frozen = [], [], [], 0, 0
    for ep in range(args.episodes):
        o, i = env.reset()
        st = i["raw_state"]
        d0 = float(st.get("distance", 0) or 0)
        s, acts = 0, [0] * C.N_ACTIONS
        degraded = False
        while True:
            a, _ = model.predict(o, deterministic=args.deterministic)
            a = int(a)
            acts[a] += 1
            o, r, term, trunc, info = env.step(a)
            st = info["raw_state"]
            if info.get("degraded"):
                degraded = True
                break
            s += 1
            if term or trunc:
                break
        prog = float(st.get("distance", 0) or 0) - d0
        sc = int(st.get("score", 0) or 0)
        mix = " ".join(f"{_ANAMES[k]}:{acts[k]}" for k in range(C.N_ACTIONS) if acts[k])
        if degraded or prog < 5.0:
            frozen += 1
            print(f"  ep{ep:2d} FROZEN/limbo (prog={prog:.0f}) — phone paused?  [{mix}]", flush=True)
            continue
        dists.append(prog)
        scores.append(sc)
        steps.append(s)
        if term:
            deaths += 1
        print(f"  ep{ep:2d} dist={prog:7.1f} score={sc:7d} steps={s:4d} "
              f"{'DIED' if term else 'cap '}  [{mix}]", flush=True)

    if dists:
        print("=" * 64)
        print(f"SUMMARY {args.model}")
        print(f"  runs counted : {len(dists)}/{args.episodes}  (frozen/limbo {frozen})")
        print(f"  distance     : median {np.median(dists):.0f}  best {max(dists):.0f}  "
              f"mean {np.mean(dists):.0f}")
        print(f"  score        : median {int(np.median(scores))}  best {max(scores)}")
        print(f"  steps        : median {int(np.median(steps))}  deaths {deaths}/{len(dists)}")
    else:
        print(f"NO valid runs ({frozen}/{args.episodes} frozen) — check the phone screen.")


if __name__ == "__main__":
    main()
