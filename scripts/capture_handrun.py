#!/usr/bin/env python3
"""Record a device run driven by a HAND policy (no NN, no training) for the replay_diff loop.

Survives obstacles reactively (jump/slide on the lookahead) and turns at junctions aiming the cell
CENTER (cprog >= --turn-aim). Logs the FULL field set incl. `distance` (needed by set_state so the
sim's velocity curve indexes correctly) per control step, in the capture_calib schema that
scripts/replay_diff.py consumes. Use it to gather current-hook captures for the policy-free
sim-vs-game diff (turns, obstacles, level-changes, ...).

    .venv/bin/python scripts/capture_handrun.py --episodes 10 --out captures/turn_calib.jsonl
    .venv/bin/python scripts/capture_handrun.py --turn-aim 0.20 --out captures/turn_early.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import templerun_rl.constants as C  # noqa: E402
from templerun_rl.client import StateClient, ActionClient  # noqa: E402

SL, SR, JMP, SLD, TL, TR, NOP = 0, 1, 2, 3, 4, 5, 6
# everything set_state / replay_diff needs — distance is CRITICAL (velocity = curve(distance)).
LOG = ("distance", "path_type", "path_level", "obstacle", "is_turn", "next_is_turn",
       "can_turn_left", "can_turn_right", "must_jump", "must_slide", "next_must_jump",
       "next_must_slide", "next_wide_gap", "narrow_target_lane", "lane", "cell_progress", "facing",
       "velocity", "is_jumping", "is_sliding", "is_strafing", "is_stumbling", "is_on_ground",
       "is_dead", "is_game_over", "ground_height", "ground_change_higher", "stumble_kill_timer",
       "stumble_count", "did_stumble", "death_type", "_hv")


def reset(ac, sc, seq, timeout=8.0):
    for b in (8, 9):
        ac.send(b)
        time.sleep(0.4)
    t = time.monotonic()
    while time.monotonic() - t < timeout:
        try:
            st, seq = sc.wait_fresh(seq, timeout=3.0)
        except TimeoutError:
            continue
        if st and (st.get("is_on_ground") or (st.get("path_type") or -1) >= 0) and not st.get("is_dead"):
            return seq
    return seq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--turn-aim", type=float, default=0.38, help="swipe once cprog >= this at a junction")
    p.add_argument("--out", default="captures/handrun.jsonl")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--duration", type=float, default=300.0)
    a = p.parse_args()
    ports = C.device_ports(a.device_index)
    sc = StateClient(host=a.host, port=ports["state_port"], connect_timeout=8.0).connect(required=True)
    ac = ActionClient(host=a.host, port=ports["action_port"], connect_timeout=8.0).connect(required=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("w")
    seq = 0
    seq = reset(ac, sc, seq)
    n = ep = step = cd = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.duration and ep < a.episodes:
        try:
            st, seq = sc.wait_fresh(seq, timeout=3.0)
        except TimeoutError:
            continue
        if st is None:
            continue
        if st.get("is_dead"):
            rec = {"ep": ep, "step": step, "action": NOP}
            rec.update({k: st.get(k) for k in LOG})
            f.write(json.dumps(rec) + "\n")
            n += 1
            print(f"ep {ep}: dist={st.get('distance'):.0f} dt={st.get('death_type')} pt={st.get('path_type')}",
                  flush=True)
            ep += 1
            step = cd = 0
            f.flush()
            seq = reset(ac, sc, seq)
            continue
        act = NOP
        if cd > 0:
            cd -= 1
        elif st.get("must_jump") or st.get("next_must_jump") or st.get("next_wide_gap"):
            act = JMP
        elif st.get("must_slide") or st.get("next_must_slide"):
            act = SLD
        elif st.get("is_turn") and float(st.get("cell_progress") or 0) >= a.turn_aim:
            act = TL if st.get("can_turn_left") else TR
            cd = 5
        ac.send(act)
        rec = {"ep": ep, "step": step, "action": act}
        rec.update({k: st.get(k) for k in LOG})
        f.write(json.dumps(rec) + "\n")
        n += 1
        step += 1
        if step % 40 == 0:
            f.flush()
    f.close()
    sc.close()
    ac.close()
    print(f"wrote {n} frames, {ep} eps -> {out}")


if __name__ == "__main__":
    main()
