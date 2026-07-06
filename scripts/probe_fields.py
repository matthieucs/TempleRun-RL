#!/usr/bin/env python3
"""Phase-0 field probe: confirm the game's own metric signals (hook obs-probe-12a) on the phone.

The make-or-break test is PlayerXOffset. The old `lane` (px - nearest-cell-center) reads ~0 in EVERY
settled lane, so the policy can't localize (narrows + strafe-off-edge). PlayerXOffset (CharacterPlayer
@0x190) is the game's OWN lateral offset — it SHOULD settle at ~3 distinct persistent values (L/C/R)
and glide during a strafe. This drives a fixed NOOP->Strafe->settle script at run start and reports,
per commanded lane, the settled player_x_offset vs the old lane. If player_x_offset separates the
lanes while lane collapses to ~0, the lane fix = read PlayerXOffset.

    python scripts/probe_fields.py --episodes 8 --out captures/probe.jsonl

Read-only w.r.t. game state; only sends strafe/noop actions in a live run (no hook rules).
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import templerun_rl.constants as C  # noqa: E402
from templerun_rl.env import TempleRunEnv  # noqa: E402

SL, SR, JMP, SLD, TL, TR, NOP = 0, 1, 2, 3, 4, 5, 6

# Fixed script: settle center, go LEFT + settle, back to center, go RIGHT + settle, back to center.
# commanded_lane after each action (start 0 = center, -1 = left, +1 = right).
SETTLE = 6
SCRIPT = ([NOP] * 4
          + [SL] + [NOP] * SETTLE      # -> left
          + [SR] + [NOP] * SETTLE      # -> center
          + [SR] + [NOP] * SETTLE      # -> right
          + [SL] + [NOP] * SETTLE)     # -> center

# fields we care about for the verdict
PROBE = ("player_x_offset", "lane", "is_strafing", "facing", "distance", "is_dead",
         "over_turn_area", "is_turn", "cell_progress", "dist_to_turn",
         "dist_to_change_temple", "dist_to_change_other", "ground_height", "is_over_ground",
         "stumble_kill_timer", "death_type")


def commanded_lane_trace():
    """Yield (action, commanded_lane_after) so we can bucket settled frames by lane."""
    lane = 0
    for a in SCRIPT:
        if a == SL:
            lane = max(-1, lane - 1)
        elif a == SR:
            lane = min(1, lane + 1)
        yield a, lane


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--out", default="captures/probe.jsonl")
    a = p.parse_args()

    ports = C.device_ports(a.device_index)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = TempleRunEnv(
        host="127.0.0.1", state_port=ports["state_port"], action_port=ports["action_port"],
        max_episode_steps=len(SCRIPT) + 5, reward_mode="distance", auto_connect=True,
        freeze_timeout=20.0, reset_timeout=8.0, connect_timeout=8.0,
    )

    script = list(commanded_lane_trace())
    # settled = a NOOP frame at least 3 steps after the last strafe (glide done). bucket by lane.
    by_lane_xoff = defaultdict(list)
    by_lane_lane = defaultdict(list)
    n_written = 0
    hv_seen = None

    f = out.open("w")
    try:
        for ep in range(a.episodes):
            o, info = env.reset()
            st = info["raw_state"]
            hv_seen = st.get("_hv", hv_seen)
            since_strafe = 99
            for step, (act, cmd_lane) in enumerate(script):
                o, r, t, tr, info = env.step(act)
                st = info["raw_state"]
                if info.get("degraded"):
                    print(f"ep {ep}: device degraded, skipping rest", flush=True)
                    break
                since_strafe = 0 if act in (SL, SR) else since_strafe + 1
                rec = {"ep": ep, "step": step, "action": act, "cmd_lane": cmd_lane,
                       "since_strafe": since_strafe}
                for k in PROBE:
                    rec[k] = st.get(k)
                f.write(json.dumps(rec) + "\n")
                n_written += 1
                # settled sample: not mid-strafe, glide finished, alive
                if (not st.get("is_dead") and not st.get("is_strafing") and since_strafe >= 3
                        and st.get("player_x_offset") is not None):
                    by_lane_xoff[cmd_lane].append(float(st.get("player_x_offset")))
                    by_lane_lane[cmd_lane].append(float(st.get("lane") or 0.0))
                if t or tr:
                    break
            print(f"ep {ep}: dist={st.get('distance')} dead={st.get('is_dead')} "
                  f"settled_samples={sum(len(v) for v in by_lane_xoff.values())}", flush=True)
    finally:
        f.close()
        env.close()

    # ---- verdict ----
    lbl = {-1: "LEFT ", 0: "CENTER", 1: "RIGHT"}

    def summarize(d):
        return {k: (round(stats.mean(v), 2),
                    round(stats.pstdev(v), 2) if len(v) > 1 else 0.0, len(v))
                for k, v in sorted(d.items())}

    print(f"\n=== PROBE VERDICT (hook {hv_seen}) — {n_written} frames -> {out} ===")
    xo, ln = summarize(by_lane_xoff), summarize(by_lane_lane)
    print("settled per commanded lane (mean, std, n):")
    print(f"{'lane':<8}{'PlayerXOffset':<26}{'old lane (px-cx)':<26}")
    for k in (-1, 0, 1):
        xs = xo.get(k, ("-", "-", 0))
        ls = ln.get(k, ("-", "-", 0))
        print(f"{lbl[k]:<8}mean={xs[0]:<7} std={xs[1]:<6} n={xs[2]:<5}  "
              f"mean={ls[0]:<7} std={ls[1]:<6} n={ls[2]}")
    # separation test: do the 3 lanes' PlayerXOffset means differ by clearly more than their spread?
    means = [xo[k][0] for k in (-1, 0, 1) if k in xo and isinstance(xo[k][0], float)]
    if len(means) >= 2:
        spread = max(means) - min(means)
        print(f"\nPlayerXOffset lane-separation span = {round(spread, 2)} "
              f"(want >> per-lane std; old lane span was ~0 = the bug)")
        lane_means = [ln[k][0] for k in (-1, 0, 1) if k in ln and isinstance(ln[k][0], float)]
        if lane_means:
            print(f"old-lane span = {round(max(lane_means) - min(lane_means), 2)} (≈0 confirms it's broken)")
        print("\nVERDICT: " + ("PlayerXOffset SEPARATES the lanes -> use it as the real lane obs."
                                if spread > 1.0 else
                                "PlayerXOffset did NOT separate lanes -> fall back to a computed projection."))
    else:
        print("\nNot enough settled samples (runs died early). Re-run with more --episodes.")


if __name__ == "__main__":
    main()
