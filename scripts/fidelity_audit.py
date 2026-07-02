#!/usr/bin/env python3
"""Sim-vs-real fidelity audit: compare the headless sim to a REAL device capture across many dimensions,
so we know the sim is a faithful replica (not just on the two hand-flagged obstacles).

Real = a log_stream.py / capture_calib.py JSONL of device play. Sim = an oracle rollout collecting the
same per-frame fields. Compares: velocity-vs-distance curve, per-frame EVENT RATES (turn / must_jump /
must_slide / narrow / crevasse / airborne / sliding), path_level time, and turn/obstacle spacing.

    python scripts/fidelity_audit.py captures/human.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import templerun_rl  # noqa: E402  (registers env)
from train_sim import _oracle  # noqa: E402

NARROW = lambda pt: isinstance(pt, int) and 17 <= pt <= 30          # noqa: E731
CREV = lambda fr: int(fr.get("obstacle") or 0) == 7 or (isinstance(fr.get("path_type"), int)
                                                        and 33 <= fr["path_type"] <= 36)  # noqa: E731


def event_rates(frames) -> dict:
    """Per-ALIVE-frame fraction of each event + mean velocity, over a list of state dicts."""
    al = [f for f in frames if not f.get("is_dead") and (f.get("path_type") not in (-1, None))]
    n = max(1, len(al))
    def frac(pred):
        return 100.0 * sum(1 for f in al if pred(f)) / n
    vels = [f.get("velocity") for f in al if f.get("velocity")]
    return {
        "frames": len(al),
        "is_turn%": frac(lambda f: f.get("is_turn")),
        "must_jump%": frac(lambda f: f.get("must_jump")),
        "must_slide%": frac(lambda f: f.get("must_slide")),
        "narrow%": frac(lambda f: NARROW(f.get("path_type"))),
        "crevasse%": frac(lambda f: CREV(f)),
        "airborne%": frac(lambda f: f.get("is_jumping")),
        "sliding%": frac(lambda f: f.get("is_sliding")),
        "vel_med": float(np.median(vels)) if vels else 0.0,
    }


def vel_curve(frames, edges):
    """Mean velocity in each distance bin."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        vs = [f.get("velocity") for f in frames
              if f.get("velocity") and lo <= float(f.get("distance") or 0) < hi and not f.get("is_dead")]
        out.append(float(np.mean(vs)) if vs else float("nan"))
    return out


def level_dist(frames):
    c = Counter(f.get("path_level") for f in frames
                if f.get("path_level") not in (-1, None) and not f.get("is_dead"))
    tot = max(1, sum(c.values()))
    return {k: round(100.0 * v / tot, 1) for k, v in sorted(c.items())}


def sim_frames(n_eps=120, seed0=4000):
    """Oracle rollout collecting raw_state per step. Mix of slow + high-speed starts to cover the range."""
    out = []
    env = gym.make("TempleRunSim-v0", max_episode_steps=8000,
                   config=dict(domain_rand=True, curriculum_max_start_dist=6000.0)).unwrapped
    for ep in range(n_eps):
        o, i = env.reset(seed=seed0 + ep)
        st = i["raw_state"]
        st["ep"] = ep                      # tag episode so turn_window() can segment runs
        out.append(dict(st))
        while True:
            o, r, term, trunc, info = env.step(_oracle(st))
            st = info["raw_state"]
            st["ep"] = ep
            out.append(dict(st))
            if term or trunc:
                break
    return out


def turn_window(frames) -> dict:
    """Measure the TURN mechanic as the game/sim actually presents it (the diff the project never had).

    For each contiguous run (segmented by ``ep``), over ALIVE frames:
      * is_turn cprog distribution + run length  -> how wide/late the turnable window is at 10 Hz
      * facing-change cprog                       -> where the rotation actually lands
      * next_is_turn run length                   -> the anticipation warning on the approach cell
    Device (human) reference (human_calib_v2, 365 turns): is_turn is a 1-FRAME flash at cprog med 0.23
    /p90 0.38; facing flips ~1 step later (med 0.68); next_is_turn ~3 frames. A faithful sim must match
    these -- a wider/later is_turn window teaches the policy to aim late and MISS the real 1-frame flash.
    """
    isturn_cp, face_cp = [], []
    isturn_run, next_run = [], []
    al = [f for f in frames if not f.get("is_dead")]
    # segment into contiguous runs by ep
    runs, cur, ep = [], [], None
    for f in al:
        e = f.get("ep")
        if ep is not None and e != ep:
            runs.append(cur); cur = []
        cur.append(f); ep = e
    if cur:
        runs.append(cur)
    for g in runs:
        i = 0
        while i < len(g):
            if g[i].get("is_turn"):
                j = i
                while j < len(g) and g[j].get("is_turn"):
                    isturn_cp.append(g[j].get("cell_progress") or 0.0); j += 1
                isturn_run.append(j - i); i = j
            else:
                i += 1
        i = 0
        while i < len(g):
            if g[i].get("next_is_turn"):
                j = i
                while j < len(g) and g[j].get("next_is_turn"):
                    j += 1
                next_run.append(j - i); i = j
            else:
                i += 1
        for k in range(1, len(g)):
            a, b = g[k - 1].get("facing"), g[k].get("facing")
            if a and b and a != b:
                face_cp.append(g[k].get("cell_progress") or 0.0)
    def q(a):
        a = np.array(a, float)
        if not len(a):
            return dict(n=0)
        return dict(n=len(a), med=float(np.median(a)),
                    p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)))
    def hist(runlens):
        c = Counter(runlens)
        tot = max(1, sum(c.values()))
        return {k: f"{100.0 * v / tot:.0f}%" for k, v in sorted(c.items())}
    return {"is_turn_cprog": q(isturn_cp), "facing_change_cprog": q(face_cp),
            "is_turn_runlen": hist(isturn_run), "next_is_turn_runlen": hist(next_run)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("real_jsonl")
    p.add_argument("--sim-eps", type=int, default=120)
    a = p.parse_args()

    real = [json.loads(l) for l in Path(a.real_jsonl).read_text().splitlines() if l.strip()]
    sim = sim_frames(a.sim_eps)
    print(f"real frames={len(real)}  sim frames={len(sim)}\n")

    # --- A. velocity vs distance ---
    edges = [0, 200, 500, 1000, 1500, 2000, 3000, 4000, 6000, 10000]
    rc, sc = vel_curve(real, edges), vel_curve(sim, edges)
    print("=== A. velocity vs distance (mean velocity per bin) ===")
    print(f"  {'dist bin':>12} | {'REAL':>7} | {'SIM':>7} | diff")
    for (lo, hi), rv, sv in zip(zip(edges[:-1], edges[1:]), rc, sc):
        d = (sv - rv) if (rv == rv and sv == sv) else float("nan")
        print(f"  {f'{lo}-{hi}':>12} | {rv:7.1f} | {sv:7.1f} | {d:+.1f}")

    # --- B. per-frame event rates ---
    rr, sr = event_rates(real), event_rates(sim)
    print("\n=== B. per-alive-frame event rates (%) ===")
    print(f"  {'metric':>12} | {'REAL':>7} | {'SIM':>7}")
    for k in ("is_turn%", "must_jump%", "must_slide%", "narrow%", "crevasse%", "airborne%", "sliding%", "vel_med"):
        print(f"  {k:>12} | {rr[k]:7.2f} | {sr[k]:7.2f}")

    # --- C. path_level time distribution ---
    print("\n=== C. path_level time distribution (% of alive frames) ===")
    print(f"  REAL: {level_dist(real)}")
    print(f"  SIM : {level_dist(sim)}")

    # --- D. DISTANCE-MATCHED event rates (removes the early/late sampling bias) ---
    print("\n=== D. event rates MATCHED by distance band (real vs sim) ===")
    bands = [(500, 1500), (1500, 3000), (3000, 6000)]
    for lo, hi in bands:
        rb = [f for f in real if lo <= float(f.get("distance") or 0) < hi]
        sb = [f for f in sim if lo <= float(f.get("distance") or 0) < hi]
        if not rb or not sb:
            continue
        rr2, sr2 = event_rates(rb), event_rates(sb)
        print(f"  -- {lo}-{hi} (real {rr2['frames']}f / sim {sr2['frames']}f) --")
        for k in ("is_turn%", "must_jump%", "must_slide%", "narrow%", "crevasse%", "airborne%"):
            print(f"       {k:>12} | real {rr2[k]:6.2f} | sim {sr2[k]:6.2f}")

    # --- E. TURN WINDOW mechanic (the structural gap: when is_turn is presented + when the turn lands) ---
    rt, stt = turn_window(real), turn_window(sim)
    print("\n=== E. TURN WINDOW (is_turn presentation vs where the turn lands) ===")
    print("  a wider/later sim window than device => policy aims late + MISSES the real 1-frame flash")
    def show(tag, d):
        ic, fc = d["is_turn_cprog"], d["facing_change_cprog"]
        print(f"  {tag:>5}: is_turn cprog med={ic.get('med', float('nan')):.2f} "
              f"p10={ic.get('p10', float('nan')):.2f} p90={ic.get('p90', float('nan')):.2f} (n={ic.get('n', 0)})")
        print(f"         facing-change cprog med={fc.get('med', float('nan')):.2f} "
              f"p90={fc.get('p90', float('nan')):.2f}")
        print(f"         is_turn runlen {d['is_turn_runlen']}   next_is_turn runlen {d['next_is_turn_runlen']}")
    show("REAL", rt)
    show("SIM", stt)


if __name__ == "__main__":
    main()
