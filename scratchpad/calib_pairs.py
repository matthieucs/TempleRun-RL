#!/usr/bin/env python3
"""Calibrate the RANDOMIZED tight pair against the DEVICE pair composition.

USER (2026-07-27): "you really have to randomize the obstacles that follow one another. You shall not
skip the 2 obstacles in a row training!"

Measured gap (scratchpad/pair_coverage.py) -- the sim can only emit GAP-LED pairs, because every
bespoke compound (jumpslide / jumpnarrow / jumpgap / slidegap) starts with a wide gap:
      crevasse-led pairs      device 13.6%   sim 38.9%
      TreeSlide -> TreeSlide  device  9.8%   sim  2.1%   (0.21x)
      Flame     -> Flame      device  8.1%   sim  0.2%   (0.03x)
      pairs / 1000 cells      device 27.1    sim 21.7

`tight_pair_prob` is the RIGHT mechanism (it relaxes the spacing floor for a NORMAL placement, whose
type comes from obstacle_type_weights -- so both ends are drawn from the mix) and the compounds are
what skew the composition. This sweeps both together and scores each candidate on THREE numbers at
once, because matching a single statistic is exactly the error that cost two 10M arms on 2026-07-25
(see the calibrate-on-outcome rule).

    python scratchpad/calib_pairs.py
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scratchpad"))
from pair_coverage import structs_from_seq, pairs_from_structs  # noqa: E402

# device targets, pooled human + v92b + v104 (scratchpad/pair_coverage.py)
T_CREV_LED = 13.6
T_SLIDE_REPEAT = 17.9      # TreeSlide->TreeSlide + Flame->Flame
T_SAME_TYPE = 24.4
T_RATE = 27.1              # pairs per 1000 cells


def score(cfg_over, ncells=30000, start=5000.0):
    from templerun_rl.sim_env import TempleRunSimEnv
    cfg = json.load(open(_ROOT / "scratchpad" / "_v105_cfg.json"))
    cfg.update({"episode_profiles": [], "domain_rand": False, "curriculum_max_start_dist": 0.0})
    cfg.update(cfg_over)
    env = TempleRunSimEnv(config=cfg)
    env.reset(seed=31337)
    env._distance_real = start
    seq = []
    for _ in range(ncells):
        seq.append(int(env._next_cell().get("obstacles", 0)))
        env._distance_real += 8.7
    c = Counter()
    for key, _ in pairs_from_structs(structs_from_seq(seq), 3):
        c[key] += 1
    tot = sum(c.values()) or 1
    crev = 100 * sum(v for k, v in c.items() if k[0] == 7) / tot
    slide = 100 * (c.get((5, 5), 0) + c.get((6, 6), 0)) / tot
    same = 100 * sum(v for k, v in c.items() if k[0] == k[1]) / tot
    rate = 1000 * tot / len(seq)
    err = (abs(crev - T_CREV_LED) / T_CREV_LED + abs(slide - T_SLIDE_REPEAT) / T_SLIDE_REPEAT
           + abs(same - T_SAME_TYPE) / T_SAME_TYPE + abs(rate - T_RATE) / T_RATE)
    return crev, slide, same, rate, err


def main():
    print(f"{'candidate':44} {'crevLed':>8} {'slideRep':>9} {'sameTyp':>8} {'rate':>6} {'err':>6}")
    print(f"{'DEVICE TARGET':44} {T_CREV_LED:7.1f}% {T_SLIDE_REPEAT:8.1f}% {T_SAME_TYPE:7.1f}% "
          f"{T_RATE:5.1f}      -")
    cands = [("v105 baseline", {})]
    for tp in (0.10, 0.25, 0.45):
        cands.append((f"tight {tp}", {"tight_pair_prob": tp, "tight_pair_min_cells": 1}))
    # the compounds are what make it crevasse-led -> halve / quarter them alongside the tight pair
    for tp in (0.25, 0.45):
        for sc in (0.5, 0.25):
            cands.append((f"tight {tp} + compounds x{sc}", {
                "tight_pair_prob": tp, "tight_pair_min_cells": 1,
                "jumpslide_prob": 0.12 * sc, "jumpnarrow_prob": 0.12 * sc,
                "jumpgap_prob": 0.15 * sc, "slidegap_prob": 0.12 * sc}))
    best = None
    for name, over in cands:
        crev, slide, same, rate, err = score(over)
        flag = ""
        if best is None or err < best[0]:
            best, flag = (err, name, over), "  <<< best"
        print(f"{name:44} {crev:7.1f}% {slide:8.1f}% {same:7.1f}% {rate:5.1f} {err:6.2f}{flag}")
    print(f"\nBEST: {best[1]}  err={best[0]:.2f}")
    print(f"  overrides: {json.dumps(best[2])}")


if __name__ == "__main__":
    main()
