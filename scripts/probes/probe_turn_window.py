#!/usr/bin/env python3
"""Measure the REAL executable turn window: at each junction, deliberately DELAY the swipe to a
target cell_progress and log the outcome (turned / stumbled / fell). Decides the latched
`turn_window_open` obs semantics (CLAUDE.md §13: is_turn is a ~1-frame flash at cprog ~0.23 — but is
the turn still EXECUTABLE at 0.5? 0.7? 0.9? Nobody ever measured; the human always swipes instantly).

Per junction: ride centered, wait for is_turn (or the turn cell via can_turn_*), swipe TurnLeft/Right
only once cell_progress >= the episode's target delay (cycled over --delays). Between junctions,
survive reactively (jump/slide) and NoOp (analog-tilt recenter keeps us centered; under legacy hook
stay center too since we never strafe).

Outcome per attempt (from the frames after the swipe): TURNED (facing changed), STUMBLE
(did_stumble/stumble_count bump, no facing change), FALL (death_type after swipe / ran off).
MISSED = the junction ended before the swipe fired (delay beyond the cell).

    python scripts/probes/probe_turn_window.py --episodes 12 --delays 0.1,0.3,0.5,0.7,0.9 \
        --out captures/turn_window_probe.jsonl
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import templerun_rl.constants as C  # noqa: E402
from templerun_rl.env import TempleRunEnv  # noqa: E402

SL, SR, JMP, SLD, TL, TR, NOP = 0, 1, 2, 3, 4, 5, 6
LOG = ("path_type", "is_turn", "can_turn_left", "can_turn_right", "cell_progress", "facing",
       "lane", "velocity", "is_stumbling", "did_stumble", "stumble_count", "stumble_kill_timer",
       "is_dead", "death_type", "distance", "next_is_turn", "_hv")


def survive_action(st):
    if st.get("must_jump") or st.get("next_must_jump") or st.get("next_wide_gap"):
        return JMP
    if st.get("must_slide") or st.get("next_must_slide"):
        return SLD
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--delays", default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--out", default="captures/turn_window_probe.jsonl")
    a = p.parse_args()
    delays = [float(x) for x in a.delays.split(",")]
    ports = C.device_ports(a.device_index)
    env = TempleRunEnv(host="127.0.0.1", state_port=ports["state_port"],
                       action_port=ports["action_port"], max_episode_steps=a.max_steps,
                       reward_mode="distance", auto_connect=True,
                       freeze_timeout=20.0, reset_timeout=8.0, connect_timeout=8.0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("w")
    n_turn = 0
    try:
        for ep in range(a.episodes):
            o, info = env.reset()
            st = info["raw_state"]
            waiting = False          # on a junction cell, delaying the swipe
            for step in range(a.max_steps):
                target = delays[n_turn % len(delays)]
                act = NOP
                surv = survive_action(st)
                on_turn_cell = bool(st.get("is_turn") or
                                    ((st.get("can_turn_left") or st.get("can_turn_right"))
                                     and not st.get("next_is_turn")))
                phase = ""
                if on_turn_cell:
                    waiting = True
                if waiting and on_turn_cell:
                    cp = float(st.get("cell_progress") or 0.0)
                    if cp >= target:
                        act = TL if st.get("can_turn_left") else TR
                        phase = f"swipe@{cp:.2f}/tgt{target}"
                        waiting = False
                        n_turn += 1
                    else:
                        phase = f"wait@{cp:.2f}/tgt{target}"
                elif waiting and not on_turn_cell:
                    # junction gone before we swiped (delay beyond the cell / crossed out) -> MISSED
                    phase = f"missed/tgt{target}"
                    waiting = False
                    n_turn += 1
                elif surv is not None:
                    act = surv
                o, r, t, tr, info = env.step(act)
                st = info["raw_state"]
                if info.get("degraded"):
                    print(f"ep {ep}: degraded, abandoning episode", flush=True)
                    break
                if phase or st.get("is_turn") or st.get("did_stumble") or st.get("is_dead"):
                    rec = {"ep": ep, "step": step, "action": act, "phase": phase}
                    for k in LOG:
                        rec[k] = st.get(k)
                    f.write(json.dumps(rec) + "\n")
                if t or tr:
                    break
            f.flush()
            print(f"ep {ep}: dist={st.get('distance')} attempts so far={n_turn}", flush=True)
    finally:
        f.close()
        env.close()
    print(f"done: {n_turn} junction attempts -> {out}")


if __name__ == "__main__":
    main()
