#!/usr/bin/env python3
"""Record a device run driven by a HAND policy (no NN, no training) for the replay_diff loop.

Survives obstacles reactively (jump/slide on the lookahead) and turns at junctions aiming the cell
CENTER (cprog >= --turn-aim). Logs the FULL field set incl. `distance` (needed by set_state so the
sim's velocity curve indexes correctly) per control step, in the capture_calib schema that
scripts/validate/replay_diff.py consumes. Use it to gather current-hook captures for the policy-free
sim-vs-game diff (turns, obstacles, level-changes, ...).

    .venv/bin/python scripts/device/capture_handrun.py --episodes 10 --out captures/turn_calib.jsonl
    .venv/bin/python scripts/device/capture_handrun.py --turn-aim 0.20 --out captures/turn_early.jsonl
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import templerun_rl.constants as C  # noqa: E402
from templerun_rl.client import StateClient, ActionClient  # noqa: E402

SL, SR, JMP, SLD, TL, TR, NOP = 0, 1, 2, 3, 4, 5, 6
# everything set_state / replay_diff needs — distance is CRITICAL (velocity = curve(distance)).
LOG = ("distance", "path_type", "next_path_type", "path_level", "obstacle", "is_turn", "next_is_turn",
       "can_turn_left", "can_turn_right", "must_jump", "must_slide", "next_must_jump",
       "next_must_slide", "next_wide_gap", "narrow_target_lane", "lane", "cell_progress", "facing",
       "velocity", "is_jumping", "is_sliding", "is_strafing", "is_stumbling", "is_on_ground",
       "is_dead", "is_game_over", "ground_height", "ground_change_higher", "stumble_kill_timer",
       "stumble_count", "did_stumble", "death_type",
       "narrow_band_lo", "narrow_band_hi", "on_region", "_hv")


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
    p.add_argument("--narrow-aim", action="store_true",
                   help="also strafe toward the streamed narrow_target_lane (for the narrows track)")
    p.add_argument("--narrow-cycle", action="store_true",
                   help="cycle L/C/R at each narrow instead of aiming the safe lane (measures the "
                        "fall band directly, like capture_calib --narrow-override)")
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
    narrow_count = 0
    NARROW_CENTER = 4.5
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
        pt = int(st.get("path_type") or -1)
        npt = int(st.get("next_path_type") or -1)
        on_narrow = 17 <= pt <= 30
        near_narrow = 17 <= npt <= 30
        if cd > 0:
            cd -= 1
        elif st.get("must_jump") or st.get("next_must_jump") or st.get("next_wide_gap"):
            act = JMP
        elif st.get("must_slide") or st.get("next_must_slide"):
            act = SLD
        elif st.get("is_turn") and float(st.get("cell_progress") or 0) >= a.turn_aim:
            act = TL if st.get("can_turn_left") else TR
            cd = 5
        elif (a.narrow_aim or a.narrow_cycle) and (on_narrow or near_narrow):
            if a.narrow_cycle:
                if near_narrow and not on_narrow:      # count on approach
                    narrow_count += 1
                tgt = {0: 0.0, 1: -NARROW_CENTER, 2: NARROW_CENTER}[narrow_count % 3]  # C / L / R
            else:
                ntl = st.get("narrow_target_lane")
                tgt = (int(ntl) * NARROW_CENTER) if ntl is not None else 0.0
            lane = float(st.get("lane") or 0)
            if lane < tgt - 1.0:
                act = SR
            elif lane > tgt + 1.0:
                act = SL
            else:
                act = NOP
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
