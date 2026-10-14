#!/usr/bin/env python3
"""Passive state-stream logger for HUMAN play — read-only, injects NO actions (won't fight your touch).

You play on the phone (deliberately hitting each crevasse type + each narrow); this records every state
frame to JSONL, segmented into runs (a new "ep" each time distance resets at a fresh run start). Analyze
with scripts/analysis/analyze_calib.py (same schema as capture_calib.py).

Crevasse types to sample deliberately (the sim must distinguish them):
  * same-level gap (HoleLong) — path_level unchanged across the gap
  * altitude transitions — path_level changes: low->mid, mid->high (step UP) and mid->low, high->mid (DOWN)
For each, cross it a few times: clear it (jump well) AND fall in (jump early / don't jump) so the analyzer
sees both outcomes. Same for narrows: hold each lane (L/C/R) so we learn which one survives.

    python scripts/device/log_stream.py --out captures/human.jsonl          # Ctrl-C to stop
    python scripts/device/log_stream.py --device-index 0 --duration 600     # auto-stop after 10 min
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import templerun_rl.constants as C  # noqa: E402
import templerun_rl.obs as O  # noqa: E402
from templerun_rl.client import StateClient  # noqa: E402

# full curated field set (superset of analyze_calib's needs); we dump whatever the hook streams.
LOG_FIELDS = (
    "distance", "score", "velocity", "path_type", "next_path_type", "obstacle", "path_level",
    "lane", "cell_progress", "facing",
    "is_jumping", "is_sliding", "is_strafing", "is_on_ground", "is_stumbling",
    "is_dead", "is_game_over", "is_turn", "next_is_turn",
    "can_turn_left", "can_turn_right", "must_jump", "must_slide",
    "next_must_jump", "next_must_slide", "next_wide_gap", "narrow_target_lane",
    # --- Phase-0 PROBE fields (hook obs-probe-12a): the game's own metric signals ---
    "player_x_offset", "over_turn_area",
    "dist_to_turn", "dist_to_turn_end", "dist_since_turn",
    "dist_to_change_temple", "dist_to_change_other", "dist_since_level_change", "dist_remainder",
    "ground_height", "is_over_ground", "ground_change_higher",
    "stumble_kill_timer", "did_stumble", "stumble_count", "death_type",
    "time_since_strafe", "strafe_duration", "max_velocity", "vel_x", "vel_z",
    # coins (hook obs-probe-13a): a tilt = coins OR narrow -> need coin placement to disambiguate
    "has_coins", "coin_loc_start", "coin_loc_end", "coin_in_air", "coin_count",
    "next_has_coins", "next_coin_loc_start", "next_coin_loc_end",
    # --- Region2D narrow-band probe (hook 15c): game-polygon safe band of the upcoming narrow ---
    "narrow_band_lo", "narrow_band_hi", "on_region", "_hv",
)


def lane_bucket(lane) -> str:
    if lane is None:
        return "?"
    if float(lane) < O.LANE_LEFT_MAX:
        return "L"
    if float(lane) > O.LANE_RIGHT_MIN:
        return "R"
    return "C"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--out", default="captures/human.jsonl")
    p.add_argument("--duration", type=float, default=0.0, help="auto-stop after N seconds (0 = until Ctrl-C)")
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args()

    ports = C.device_ports(a.device_index)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    sc = StateClient(host=a.host, port=ports["state_port"], connect_timeout=8.0).connect(required=True)
    print(f"logging state stream :{ports['state_port']} -> {out}  (Ctrl-C to stop)")

    ep, step, prev_dist, last_seq = 0, 0, None, 0
    n = 0
    t0 = time.monotonic()
    last_print = t0
    f = out.open("w")
    try:
        while True:
            if a.duration and time.monotonic() - t0 > a.duration:
                break
            try:
                st, seq = sc.wait_fresh(last_seq, timeout=5.0)
            except TimeoutError:
                # no frames (game paused / menu) — keep waiting, don't crash.
                continue
            last_seq = seq
            if st is None:
                continue
            dist = float(st.get("distance") or 0.0)
            # new run: distance reset to ~0 after a real run (covers death -> menu -> replay).
            if prev_dist is not None and dist < 8.0 and prev_dist > 30.0:
                ep += 1
                step = 0
            prev_dist = dist

            rec = {"ep": ep, "step": step, "action": None, "narrow_cmd": "", "t": round(time.monotonic() - t0, 3)}
            for k in LOG_FIELDS:
                rec[k] = st.get(k)
            rec["lane_bucket"] = lane_bucket(st.get("lane"))
            f.write(json.dumps(rec) + "\n")
            n += 1
            step += 1
            if n % 40 == 0:
                f.flush()   # periodic flush so a mid-run peek / early stop keeps the data

            now = time.monotonic()
            if now - last_print > 1.0:
                last_print = now
                print(f"\rep={ep} dist={dist:7.1f} pt={st.get('path_type')} lvl={st.get('path_level')} "
                      f"obst={st.get('obstacle')} lane={st.get('lane')} cprog={st.get('cell_progress')} "
                      f"frames={n}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        f.flush()
        f.close()
        sc.close()
    print(f"wrote {n} frames over {ep + 1} run(s) -> {out}")


if __name__ == "__main__":
    main()
