#!/usr/bin/env python3
"""REAL-vs-SIM EQUIVALENCE — the sim_equivalence idea, with the DEVICE as the reference.

USER (2026-07-26): "Why don't you build an equivalence program like you did to compare sim and
sim_torch? Can't you build an equivalence between the real game and the simulator? ... The main
question is that given the same seed and the same actions are the two equivalent? I'm not sure that
the answer is yes."

Right question. The seed cannot be shared -- device runs are NOT seedable (probe 16u: generation
draws from Unity's GLOBAL RNG, so re-seeding gives only a ~6-cell identical prefix). But the seed is
not what we need: we need the same TRACK, and the track is READABLE. So:

    1. rebuild the exact cell sequence the runner traversed, from the capture
    2. INJECT it into the sim (same mechanism sim_equivalence uses for its fixed scenarios)
    3. REPLAY the recorded action stream, frame for frame
    4. compare -- above all, DOES THE SIM DIE WHERE THE DEVICE DIED?

A device death the sim survives is a pinpointed fidelity gap: the sim lets a play live that the real
game punishes, so the policy never learns to avoid it. The reverse (sim kills, device survives) is a
phantom the policy wastes capacity avoiding.

TWO EARLIER ATTEMPTS AND WHY THEY FELL SHORT: replay_diff.py is one-step only ("the capture doesn't
carry the recorded track"), so it is blind to anything multi-frame (jump arcs, chains).
trajectory_diff.py does rebuild the track but its death locator is documented as buggy. BOTH were
written while the hook's neighbour walk died at every corner, so a continuous track could not be
rebuilt past the first junction -- that is the bug fixed in hook 17h (turn-aware walk).

FLEXIBILITY IS PRESERVED, exactly as in sim_equivalence: this harness uses INJECTED tracks for an
exact comparison, while TRAINING keeps its procedural generator and all its diversity knobs. Fixed
tracks to verify, procedural tracks to train -- the same two-tier split the project already uses.

    python scripts/real_equivalence.py captures/w15_v88_device.jsonl
    python scripts/real_equivalence.py captures/*.jsonl --sim-config '{"device_difficulty_ramp":true}'
"""
from __future__ import annotations
import argparse, glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scratchpad"))
from templerun_rl.sim_env import TempleRunSimEnv, _element   # noqa: E402
from hole_landing_forensic import load_eps, cell_seq          # noqa: E402

HOLE, NARROW, CREV = 1, 2, 7
LONG_HOLE_PT = {12, 13, 15, 16}
CREV_PT = {33, 34, 35, 36}
DT = {0: "-", 1: "Fall", 2: "Tree", 3: "Slide", 4: "Ledge", 5: "Burnt"}


def rebuild_track(ep, cells):
    """The cell sequence the runner ACTUALLY traversed, as sim cells.

    is_turn must be true if ANY frame on the cell says so: the flag is only presented across the
    executable window, so reading it on the last frame of a junction finds ~0.1% junctions instead
    of ~4% (this cost an hour on 2026-07-26 before it was caught).
    """
    per = defaultdict(list)
    for i, fr in enumerate(ep):
        per[cells[i]].append(fr)
    track, meta = [], []
    for k in sorted(per):
        fs = per[k]
        pt = Counter(int(f.get("path_type") or 1) for f in fs).most_common(1)[0][0]
        ob = Counter(int(f.get("obstacle") or 0) for f in fs).most_common(1)[0][0]
        lvl = Counter(int(f.get("path_level") or 2) for f in fs).most_common(1)[0][0]
        is_turn = any(f.get("is_turn") for f in fs)
        can_l = any(f.get("can_turn_left") for f in fs)
        can_r = any(f.get("can_turn_right") for f in fs)
        nlane = None
        for f in fs:
            if f.get("narrow_target_lane") is not None and ob == NARROW:
                nlane = int(f["narrow_target_lane"])
                break
        e = _element(type=pt, level=lvl, obstacles=ob, is_turn=is_turn,
                     can_turn_left=can_l, can_turn_right=can_r)
        # sim-side structural flags the device does not name explicitly
        e["wide_gap"] = (ob == HOLE and pt in LONG_HOLE_PT) or (ob == CREV)
        e["stepup"] = False
        e["narrow_lane"] = nlane
        track.append(e)
        meta.append(k)
    # a crevasse cap that RAISES the level is a step-up (the apex check fires there)
    for i in range(1, len(track)):
        if track[i]["obstacles"] == CREV and track[i]["level"] > track[i - 1]["level"]:
            track[i]["stepup"] = True
    return track, meta


def replay(ep, cells, cfg):
    """Force the sim onto the recorded track + action stream; return its death cell (or None)."""
    track, meta = rebuild_track(ep, cells)
    if len(track) < 6:
        return None, None, None
    env = TempleRunSimEnv(track_source="procedural", config=cfg, seed=0,
                          max_episode_steps=100000, reward_mode="distance",
                          action_history_len=0, frame_stack=1)
    env.reset(seed=0)
    env._track = [dict(c) for c in track] + [_element() for _ in range(64)]
    env._cell_index = 0
    env._dist_into_cell = 0.0
    env._velocity = float(ep[0].get("velocity") or 100.0)
    env._distance_real = float(ep[0].get("distance") or 0.0)
    # RE-ANCHOR EACH FRAME. A blind open-loop replay desyncs within ~10 cells: the device runs at
    # 10Hz control with variable frame timing, the sim steps a fixed dt, so the two drift onto
    # different cells and every later action lands in the wrong place (this is why trajectory_diff's
    # death locator never worked). Instead, force the sim's POSITION/VELOCITY to the device's each
    # frame -- but leave its internal jump/slide timers running, so multi-frame arc mechanics still
    # evolve in the sim and can still be judged. Drift cannot accumulate; the death decision stays
    # the sim's own.
    # NOTE capture alignment: logged action(i) is the policy's response to state(i-1), so the action
    # recorded on frame i must be applied at frame i-1's position.
    sim_death = None
    cell0 = cells[0]
    for i in range(1, len(ep)):
        act = ep[i].get("action")
        if act is None:
            continue
        prev = ep[i - 1]
        idx = cells[i - 1] - cell0
        if not (0 <= idx < len(track)):
            break
        env._cell_index = idx
        env._dist_into_cell = float(prev.get("cell_progress") or 0.0) * float(env.cfg["cell_length"])
        env._velocity = float(prev.get("velocity") or env._velocity)
        # anchor the LANE too: without it every replay ended in a lateral Fall, because strafe
        # dynamics (analog tilt, exec latency) accumulate differently and walk the runner off the
        # side. Lane is an observed device quantity, so anchoring it isolates the LONGITUDINAL
        # mechanics (jump arcs, obstacle resolution) that this harness is actually testing.
        if prev.get("lane") is not None:
            env._lane = float(prev["lane"])
            env._lat_target = float(prev["lane"])
        env._distance_real = float(prev.get("distance") or env._distance_real)
        env.step(int(act))
        if env._is_dead:
            sim_death = (cell0 + env._cell_index, env._death_type)
            break
    dev_death = (cells[-1], int(ep[-1].get("death_type") or 0)) if ep[-1].get("is_dead") else None
    return sim_death, dev_death, len(track)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--sim-config", default="")
    a = ap.parse_args()
    # NO exec latency in a replay: the capture records the action the policy SENT, and the device
    # already applied its own ~1-frame latency before producing the state we compare against. Adding
    # the sim's latency on top double-counts it -- the jump then fires a frame late against a
    # re-anchored position and the runner dies on cells the device cleared.
    cfg = {"jump_exec_lat_frames": 0, "jump_exec_lat_jitter": 0, "hook_turn_buffer": True,
           "strafe_exec_lat_frames": 0, "turn_send_lat_frames": 0.0,
           "turn_approach_buffer": True, "turn_linger_frames": 3, "domain_rand": False,
           "curriculum_max_start_dist": 0.0, "jump_drop_prob": 0.0,
           "wide_gap_landing_model": True, "crevasse_apex_all": True,
           "crevasse_graze_worldy": True, "stepup_clip_band": 0.25}
    if a.sim_config:
        cfg.update(json.loads(a.sim_config))

    agree = both = dev_only = sim_only = 0
    early = []
    for g in a.captures:
        for path in sorted(glob.glob(g)):
            for epk, ep in load_eps(path):
                cells = cell_seq(ep)
                sd, dd, ntrack = replay(ep, cells, cfg)
                if ntrack is None:
                    continue
                if dd and sd:
                    both += 1
                    # "agrees" = the sim dies within 2 cells of where the device did
                    if abs(sd[0] - dd[0]) <= 2:
                        agree += 1
                    else:
                        early.append((path.split("\\")[-1].split("/")[-1], epk,
                                      sd[0], DT.get(sd[1], sd[1]), dd[0], DT.get(dd[1], dd[1]), ntrack))
                elif dd and not sd:
                    dev_only += 1
                elif sd and not dd:
                    sim_only += 1
    tot = both + dev_only + sim_only
    print(f"\nREAL-vs-SIM EQUIVALENCE  ({tot} episodes replayed on their RECORDED track + actions)\n")
    print(f"  both died, sim within 2 cells of the device  : {agree}")
    print(f"  both died, but at DIFFERENT places           : {both - agree}")
    print(f"  ** device DIED, sim SURVIVED                 : {dev_only}   <- sim too LENIENT (the gap"
          f" that lets bad plays live)")
    print(f"  ** sim DIED, device survived                 : {sim_only}   <- sim too HARSH (a phantom"
          f" the policy wastes capacity on)")
    if tot:
        print(f"\n  agreement on WHERE the run ends: {100*agree/tot:.0f}%")
    if early:
        print("\n  mismatched death locations (sim_cell/type vs device_cell/type):")
        for r in early[:12]:
            print(f"    {r[0][:26]:26} ep{r[1]} sim died c{r[2]} {r[3]:6} | device died c{r[4]} {r[5]:6}"
                  f"  (track {r[6]} cells)")


if __name__ == "__main__":
    main()
