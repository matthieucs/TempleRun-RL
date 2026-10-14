#!/usr/bin/env python3
"""Phase-0 probe: verify the TURN-MISTAKE = stumble-then-recover mechanic (user-described, 2026-07-01).

User: swiping (turning) at the wrong time makes the character hit the side wall -> slow down -> recenter
= a STUMBLE (survivable). A SECOND mistake in a short window -> DEATH. NOT instant-death like the sim
models. This drives deliberate turn-mistakes on a STRAIGHT and logs the stumble/death signals to confirm
+ calibrate the kill window (StumbleKillTimer @0x11C, streamed as stumble_kill_timer).

Two episode types alternated:
  A (single mistake): one turn on a straight, then watch ~15 frames -> expect stumble + SURVIVE.
  B (double mistake): two turns ~2 frames apart on a straight -> expect death on/after the 2nd.

Reactively jumps/slides to survive real obstacles so a death is attributable to the turn-mistakes.

    python scripts/probes/probe_turn_mistake.py --episodes 8
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import templerun_rl.constants as C  # noqa: E402
from templerun_rl.env import TempleRunEnv  # noqa: E402

SL, SR, JMP, SLD, TL, TR, NOP = 0, 1, 2, 3, 4, 5, 6
ANM = {0: "SL", 1: "SR", 2: "JMP", 3: "SLD", 4: "TL", 5: "TR", 6: "NOP"}
F = ("path_type", "is_turn", "is_stumbling", "velocity", "lane", "stumble_kill_timer",
     "did_stumble", "stumble_count", "is_dead", "death_type", "distance")


def survive_action(st):
    """Reactive jump/slide so a real obstacle doesn't cause the death we're attributing to turns."""
    if st.get("must_jump") or st.get("next_must_jump"):
        return JMP
    if st.get("must_slide") or st.get("next_must_slide"):
        return SLD
    return None


def on_clear_straight(st):
    """A frame where a turn would be a genuine MISTAKE: not a junction, no obstacle here/next."""
    return (not st.get("is_turn") and not st.get("next_is_turn")
            and not st.get("must_jump") and not st.get("next_must_jump")
            and not st.get("must_slide") and not st.get("next_must_slide")
            and (st.get("path_type") in (1, 2)))


def log(tag, step, a, st):
    print(f"  {tag:>7} s{step:<2} {ANM[a]:>3} pt={st.get('path_type'):>2} "
          f"turn={str(st.get('is_turn'))[:1]} stmb={str(st.get('is_stumbling'))[:1]} "
          f"vel={0 if st.get('velocity') is None else st.get('velocity'):>5.1f} "
          f"lane={0 if st.get('lane') is None else st.get('lane'):>6.2f} "
          f"skt={0 if st.get('stumble_kill_timer') is None else st.get('stumble_kill_timer'):>5.2f} "
          f"dstmb={str(st.get('did_stumble'))[:1]} scnt={st.get('stumble_count')} "
          f"dead={str(st.get('is_dead'))[:1]} dt={st.get('death_type')}")


def episode(env, kind):
    o, info = env.reset()
    st = info["raw_state"]
    fired = 0
    for step in range(90):
        a = NOP
        surv = survive_action(st)
        if surv is not None:
            a = surv
        elif on_clear_straight(st) and fired == 0:
            a = TL                       # mistake #1
            fired = 1
        elif kind == "B" and fired == 1 and not st.get("is_stumbling"):
            # second mistake shortly after the first (while the kill window may still be open)
            a = TL
            fired = 2
        o, r, t, tr, info = env.step(a)
        st = info["raw_state"]
        if info.get("degraded"):
            print("  device degraded; skip"); return
        if a == TL or st.get("is_stumbling") or st.get("is_dead"):
            log(kind, step, a, st)
        if t or tr:
            print(f"  -> END dead={st.get('is_dead')} death_type={st.get('death_type')} "
                  f"dist={st.get('distance'):.1f} stumbles={st.get('stumble_count')}")
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--episodes", type=int, default=8)
    a = p.parse_args()
    ports = C.device_ports(a.device_index)
    env = TempleRunEnv(host="127.0.0.1", state_port=ports["state_port"], action_port=ports["action_port"],
                       max_episode_steps=95, reward_mode="distance", auto_connect=True,
                       freeze_timeout=20.0, reset_timeout=8.0, connect_timeout=8.0)
    try:
        for ep in range(a.episodes):
            kind = "A" if ep % 2 == 0 else "B"
            print(f"\n--- ep{ep} type {kind} ({'single' if kind=='A' else 'double'} turn-mistake) ---")
            episode(env, kind)
    finally:
        env.close()


if __name__ == "__main__":
    main()
