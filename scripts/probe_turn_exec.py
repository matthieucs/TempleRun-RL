#!/usr/bin/env python3
"""Turn-EXECUTION probe: measure the game's delayed-turn buffer (HandleDelayedTurning + the
GameInput.TimeSinceShouldTurn timeout). Send ONE turn byte at a controlled cell_progress on a
junction cell, then hold NoOp; log whether the turn EXECUTED (facing changed), where the runner
LANDED (post-turn lane), whether it SURVIVED, and the buffer state (should_turn_*, time_since_*).

This maps: swipe-cprog -> {executes / expires-and-runs-past / lands off-center}. The sim executes
turns instantly; the device buffers+times-out, so an early swipe at the is_turn flash can expire
before the corner -> run into the water (the observed death). Requires hook >= lateral-accel-16c.

    python scripts/probe_turn_exec.py --episodes 16 --cprogs 0.1,0.25,0.4,0.55,0.7 \
        --out captures/turn_exec_probe.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import templerun_rl.constants as C  # noqa: E402
from templerun_rl.env import TempleRunEnv  # noqa: E402

SL, SR, JMP, SLD, TL, TR, NOP = 0, 1, 2, 3, 4, 5, 6
LOG = ("path_type", "is_turn", "can_turn_left", "can_turn_right", "next_is_turn", "cell_progress",
       "facing", "lane", "velocity", "is_dead", "death_type", "distance",
       "should_turn_left", "should_turn_right", "just_turned", "time_since_should_turn",
       "time_since_last_turn", "_hv")


def survive(st):
    if st.get("must_jump") or st.get("next_must_jump") or st.get("next_wide_gap"):
        return JMP
    if st.get("must_slide") or st.get("next_must_slide"):
        return SLD
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--cprogs", default="0.1,0.25,0.4,0.55,0.7")
    p.add_argument("--out", default="captures/turn_exec_probe.jsonl")
    a = p.parse_args()
    cprogs = [float(x) for x in a.cprogs.split(",")]
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
            armed = False          # currently on a junction cell, waiting to fire the swipe
            fired = False          # already swiped this junction
            logging = 0            # frames left to log after a swipe (capture the outcome)
            for step in range(a.max_steps):
                target = cprogs[n_turn % len(cprogs)]
                act = NOP
                surv = survive(st)
                on_turn_cell = bool(st.get("is_turn") or st.get("can_turn_left") or
                                    st.get("can_turn_right"))
                phase = ""
                if on_turn_cell and not fired:
                    armed = True
                    cp = float(st.get("cell_progress") or 0.0)
                    if cp >= target:
                        act = TL if st.get("can_turn_left") else TR
                        phase = f"swipe@{cp:.2f}/tgt{target}"
                        fired = True
                        logging = 10
                        n_turn += 1
                    else:
                        phase = f"arm@{cp:.2f}/tgt{target}"
                elif not on_turn_cell and armed and not fired:
                    # ran onto/off the junction before reaching the target cprog -> record a MISS
                    phase = f"passed/tgt{target}"
                    fired = True
                    logging = 10
                    n_turn += 1
                elif not on_turn_cell and fired and logging == 0:
                    armed = False
                    fired = False           # ready for the next junction
                elif surv is not None:
                    act = surv

                o, r, t, tr, info = env.step(act)
                st = info["raw_state"]
                if info.get("degraded"):
                    print(f"ep {ep}: degraded", flush=True)
                    break
                if phase or logging > 0 or st.get("is_dead"):
                    rec = {"ep": ep, "step": step, "action": act, "phase": phase,
                           "target": target}
                    for k in LOG:
                        rec[k] = st.get(k)
                    f.write(json.dumps(rec) + "\n")
                    if logging > 0:
                        logging -= 1
                if t or tr:
                    break
            f.flush()
            print(f"ep {ep}: dist={st.get('distance')} attempts={n_turn}", flush=True)
    finally:
        f.close()
        env.close()
    print(f"done: {n_turn} turn attempts -> {out}")


if __name__ == "__main__":
    main()
