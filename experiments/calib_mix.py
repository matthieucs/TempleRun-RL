#!/usr/bin/env python3
"""Calibrate the sim's HAZARD-STRUCTURE mix and density onto the device, by iteration.

Measured 2026-07-27 night over human + v92b + v104 captures, counting STRUCTURES (a 2-cell crevasse
or HoleLong is ONE obstacle) -- this is the unit the player actually reacts to:

    structure    device     sim(v105)   ratio
    Hole          32.1%       21.4%     0.67x
    Stumble       10.0%        6.3%     0.62x
    TreeJump      10.8%       16.5%     1.53x
    TreeSlide     12.3%       13.1%     1.07x
    Flame         18.7%        8.5%     0.46x
    Crevasse      16.1%       34.2%     2.12x
    per 1000 cells 121.3       72.5     0.60x

This is the ROOT of the pair-composition complaint (USER: "randomize the obstacles that follow one
another"): crevasses lead 38.9% of sim pairs vs 13.6% on device purely because there are 2.1x too
many of them, and Flame->Flame is 0.03x because flames are half-served. Fixing the mix fixes the
pairs, which is why this is calibrated here rather than by adding another bespoke compound.

WHY ITERATE: obstacle_type_weights is NOT the emitted share -- holes also arrive via the wide-gap and
compound paths, and crevasses come from the zone-change clock, not the weight table at all. So each
round measures the emitted mix and rescales the weights by target/observed (the documented approach),
with crevasse frequency and overall density on their own knobs.

    python experiments/calib_mix.py
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from pair_coverage import structs_from_seq, NAME  # noqa: E402

TARGET = {1: 32.1, 3: 10.0, 4: 10.8, 5: 12.3, 6: 18.7, 7: 16.1}   # % of hazard structures
T_DENSITY = 121.3                                                  # structures per 1000 cells
NON_CREV = [1, 3, 4, 5, 6]


def emit(cfg_over, ncells=30000, start=5000.0):
    from templerun_rl.sim_env import TempleRunSimEnv
    cfg = json.load(open(_ROOT / "experiments" / "_v105_cfg.json"))
    cfg.update({"episode_profiles": [], "domain_rand": False, "curriculum_max_start_dist": 0.0})
    cfg.update(cfg_over)
    env = TempleRunSimEnv(config=cfg)
    env.reset(seed=777)
    env._distance_real = start
    seq = []
    for _ in range(ncells):
        seq.append(int(env._next_cell().get("obstacles", 0)))
        env._distance_real += 8.7
    c = Counter(o for _, _, o in structs_from_seq(seq))
    tot = sum(c.values()) or 1
    share = {o: 100 * c.get(o, 0) / tot for o in TARGET}
    return share, 1000 * tot / len(seq)


def show(share, dens, label):
    err = sum(abs(share[o] - TARGET[o]) for o in TARGET) + abs(dens - T_DENSITY) / 4
    line = "  ".join(f"{NAME.get(o,o)[:4]} {share[o]:4.1f}" for o in sorted(TARGET))
    print(f"  {label:22} {line}  dens {dens:5.1f}   err {err:5.1f}")
    return err


def main():
    base = json.load(open(_ROOT / "experiments" / "_v105_cfg.json"))
    w = dict(base.get("obstacle_type_weights") or
             {1: 0.22, 4: 0.22, 5: 0.22, 6: 0.14, 3: 0.12, 2: 0.08})
    w = {int(k): float(v) for k, v in w.items()}
    zone = float(base.get("zone_change_every_cells", 14))
    op = float(base.get("obstacle_prob", 0.35))
    print("  " + " " * 22 + "  ".join(f"{NAME.get(o,o)[:4]:>9}" for o in sorted(TARGET)))
    print(f"  {'DEVICE TARGET':22} " + "  ".join(f"{TARGET[o]:4.1f}     " for o in sorted(TARGET))
          + f"dens {T_DENSITY:5.1f}")
    share, dens = emit({})
    show(share, dens, "v105 baseline")
    for it in range(5):
        # 1) crevasse frequency: zone clock scales with the ratio of emitted -> target crevasse share
        if share[7] > 0:
            zone = max(4.0, min(60.0, zone * (share[7] / TARGET[7])))
        # 2) non-crevasse weights: rescale by target/observed, renormalise (narrow weight untouched)
        for o in NON_CREV:
            if share[o] > 0.05:
                w[o] = max(1e-3, w[o] * (TARGET[o] / share[o]))
        nsum = sum(w[o] for o in NON_CREV)
        keep = 1.0 - w.get(2, 0.08)
        for o in NON_CREV:
            w[o] = w[o] / nsum * keep
        # 3) density: obstacle_prob toward the structure rate
        if dens > 0:
            op = max(0.05, min(0.98, op * (T_DENSITY / dens) ** 0.5))
        over = {"obstacle_type_weights": {int(k): round(v, 4) for k, v in w.items()},
                "zone_change_every_cells": int(round(zone)), "obstacle_prob": round(op, 3)}
        share, dens = emit(over)
        show(share, dens, f"iter {it+1}")
    print(f"\n  FINAL overrides:\n    obstacle_type_weights = "
          f"{ {int(k): round(v,4) for k,v in w.items()} }")
    print(f"    zone_change_every_cells = {int(round(zone))}    obstacle_prob = {round(op,3)}")
    json.dump({"obstacle_type_weights": {str(int(k)): round(v, 4) for k, v in w.items()},
               "zone_change_every_cells": int(round(zone)), "obstacle_prob": round(op, 3)},
              open(_ROOT / "experiments" / "_mix_calib.json", "w"), indent=1)
    print("    -> written to experiments/_mix_calib.json")


if __name__ == "__main__":
    main()
