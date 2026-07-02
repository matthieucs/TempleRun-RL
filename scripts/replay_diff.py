#!/usr/bin/env python3
"""ONE-STEP TRANSITION FIDELITY — validate the sim WITHOUT a policy, without training.

For every consecutive (frame_i, action_i, frame_{i+1}) in a REAL device capture (capture_calib format:
per-frame raw_state + the hook-dispatched action byte), force the sim to frame_i's exact real state,
apply the exact real action, and compare the sim's resulting state to the REAL frame_{i+1}. This
directly answers: "given the real state + the real action, does the sim transition like the game?"
No policy, no accumulation drift — each step is an independent, real-anchored transition test.

The signal that matters most: **is_dead agreement**. A frame where the device DIED but the sim
predicts SURVIVE (or vice versa) is a fidelity bug, pinpointed to the exact (path_type, action). The
report buckets transitions by (path_type family, action) so you see WHICH mechanic diverges.

    .venv/bin/python scripts/replay_diff.py captures/tilt_v4_eval.jsonl
    .venv/bin/python scripts/replay_diff.py captures/eval_stumble.jsonl --skip-lateral

Caveat: the sim generates the NEXT cell procedurally (the capture doesn't carry the recorded track),
so OBSTACLE-CLEAR outcomes that depend on the next cell can diverge for non-fidelity reasons — the
tool flags action-driven transitions (turn, stumble, immediate death) which need no future track.
Lateral (strafe) transitions only compare cleanly for captures from the CURRENT accel-tilt hook
(--skip-lateral for old-hook captures).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import gymnasium as gym  # noqa: E402
import templerun_rl  # noqa: E402
import templerun_rl.constants as C  # noqa: E402

ANM = {0: "SL", 1: "SR", 2: "JMP", 3: "SLD", 4: "TL", 5: "TR", 6: "NOP"}


def pt_family(pt) -> str:
    if not isinstance(pt, int) or pt < 0:
        return "off"
    if pt in (1, 2):
        return "straight"
    if 3 <= pt <= 10:
        return "junction"
    if 11 <= pt <= 16:
        return "hole"
    if 17 <= pt <= 30:
        return "narrow"
    if 31 <= pt <= 37:
        return "levelchange"
    return f"pt{pt}"


def hidden_of(f: dict) -> dict:
    return {k: f.get(k) for k in ("lane", "distance", "velocity", "ground_height",
                                  "stumble_kill_timer", "stumble_count", "death_type")
            if f.get(k) is not None}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("capture")
    p.add_argument("--skip-lateral", action="store_true",
                   help="ignore strafe transitions (old-hook captures use different lateral semantics)")
    p.add_argument("--death-window", type=int, default=3,
                   help="device death counts as agreement if it lands within +/-N frames of the sim's "
                        "(the sim kills instantly on a bad turn; the device rotates off + dies ~5 frames "
                        "later -> same OUTCOME, different timing). 0 = frame-exact.")
    p.add_argument("--show", type=int, default=20, help="max example divergences to print")
    a = p.parse_args()

    frames = [json.loads(l) for l in Path(a.capture).read_text().splitlines() if l.strip()]
    # group by episode so we don't diff across a reset boundary
    eps = defaultdict(list)
    for f in frames:
        eps[f.get("ep", 0)].append(f)

    env = gym.make("TempleRunSim-v0", max_episode_steps=100000,
                   config=dict(domain_rand=False)).unwrapped
    cell_len = float(env.cfg["cell_length"])

    # per (family, action): agreement + errors
    buckets = defaultdict(lambda: {"n": 0, "dead_agree": 0, "lane_err": 0.0, "vel_err": 0.0,
                                   "flag_agree": 0, "landn": 0, "land_err": 0.0})
    death_mismatches = []          # sim/device life-death disagree beyond the window — the fidelity bugs
    n_pairs = 0
    w = int(a.death_window)

    for ep, fr in eps.items():
        for i in range(len(fr) - 1):
            cur, nxt = fr[i], fr[i + 1]
            act = cur.get("action")
            if act is None or int(act) >= 7:
                continue
            if nxt.get("ep") != cur.get("ep"):
                continue
            if a.skip_lateral and int(act) in (0, 1):
                continue
            if cur.get("is_dead"):
                continue                       # already dead — nothing to transition
            fam = pt_family(cur.get("path_type"))
            # force the sim to the real state, then OVERRIDE cell_progress (set_state doesn't restore it)
            env.set_state(cur, hidden_of(cur))
            cp = cur.get("cell_progress")
            if cp is not None:
                env._dist_into_cell = float(cp) * cell_len
            env.step(int(act))
            sim = env.raw_state()
            n_pairs += 1
            b = buckets[(fam, ANM[int(act)])]
            b["n"] += 1
            sim_dead = bool(sim.get("is_dead"))
            # device death within +/-window frames of THIS transition = same outcome (timing tolerated)
            real_dead_soon = any(bool(fr[j].get("is_dead"))
                                 for j in range(i + 1, min(len(fr), i + 1 + w + 1)))
            if sim_dead == real_dead_soon:
                b["dead_agree"] += 1
            elif sim_dead and not real_dead_soon:
                death_mismatches.append((ep, fr[i].get("step"), fam, ANM[int(act)],
                                         "sim DIED / device lived", round(cur.get("lane") or 0, 1),
                                         round(cp or 0, 2), cur.get("path_type"), None))
            else:
                death_mismatches.append((ep, fr[i].get("step"), fam, ANM[int(act)],
                                         "device DIED / sim lived", round(cur.get("lane") or 0, 1),
                                         round(cp or 0, 2), cur.get("path_type"), nxt.get("death_type")))
            # transient-flag agreement (is_jumping/is_sliding/is_stumbling) + lane/velocity error
            fa = sum(1 for k in ("is_jumping", "is_sliding", "is_stumbling")
                     if bool(sim.get(k)) == bool(nxt.get(k)))
            b["flag_agree"] += fa / 3.0
            if nxt.get("lane") is not None:
                b["lane_err"] += abs(float(sim.get("lane", 0)) - float(nxt.get("lane")))
            if nxt.get("velocity"):
                b["vel_err"] += abs(float(sim.get("velocity", 0)) - float(nxt.get("velocity")))
            # TURN LANDING: for a turn that BOTH sides survive, compare where the runner LANDS a couple
            # frames later (the turn takes ~2 frames on device). This is the direct turn-model target.
            if int(act) in (4, 5) and not sim_dead and not real_dead_soon:
                j = min(len(fr) - 1, i + 2)
                dev_land = fr[j].get("lane")
                if dev_land is not None:
                    b["landn"] += 1
                    b["land_err"] += abs(float(sim.get("lane", 0)) - float(dev_land))

    print(f"one-step transition diff: {a.capture}  ({n_pairs} real (state,action) transitions, "
          f"death window +/-{w})\n")
    print(f"{'family':>12} {'act':>4} | {'n':>4} | dead-agree | flag-agree | lane_err | vel_err | turn_land_err")
    tot_n = tot_agree = 0
    for key in sorted(buckets, key=lambda k: (-buckets[k]["n"])):
        b = buckets[key]
        n = b["n"]
        tot_n += n
        tot_agree += b["dead_agree"]
        land = f"{b['land_err']/b['landn']:.1f} (n{b['landn']})" if b["landn"] else "-"
        print(f"{key[0]:>12} {key[1]:>4} | {n:>4} | {100*b['dead_agree']/n:8.0f}%  | "
              f"{100*b['flag_agree']/n:8.0f}%  | {b['lane_err']/n:7.1f}  | {b['vel_err']/n:6.1f} | {land}")
    print(f"\noverall is_dead agreement: {100*tot_agree/max(1,tot_n):.1f}%  "
          f"({tot_n - tot_agree} mismatches of {tot_n})")
    if death_mismatches:
        print(f"\nDEATH MISMATCHES (the fidelity bugs) — showing {min(a.show, len(death_mismatches))}:")
        print(f"  {'ep':>3} {'step':>5} {'family':>11} {'act':>4}  {'verdict':<26} lane cprog pt dtype")
        for m in death_mismatches[:a.show]:
            print(f"  {m[0]:>3} {m[1]:>5} {m[2]:>11} {m[3]:>4}  {m[4]:<26} {m[5]:>4} {m[6]:>5} "
                  f"{m[7]} {m[8]}")
    else:
        print("\nno death mismatches — sim reproduces every real life/death transition.")


if __name__ == "__main__":
    main()
