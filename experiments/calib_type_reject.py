#!/usr/bin/env python3
"""CALIBRATION PROBE for `type_reject_model` (the draw-then-substitute obstacle sampler).

Measures the emitted JOINT pair statistics against the device targets in
docs/design/2026-07-28-obstacle-type-sampler-rewrite.md §2:

    slide-led    share of all pairs   device 50.7%
    slide-repeat share of all pairs   device 17.9%   (TreeSlide->TreeSlide 9.8 + Flame->Flame 8.1)
    Hole->Hole   share of all pairs   device  6.1%   (independent 10.3%)
    same-type    share of all pairs   device 24.6%

⚠️ THE MEASUREMENT TRAPS (spec §5) -- every one of these is honoured here and asserted where it can be:
  * gym/env construction takes `config=`, NOT `sim_config=`; an unknown kwarg is swallowed silently.
  * every knob is re-read off `env.cfg` after construction and asserted (`_assert_cfg`).
  * the cell dict keys the obstacle as `obstacles` (PLURAL).
  * generation is driven through `env._next_cell()` -- `_next_cell_procedural()` bypasses the
    `_last_obst_type` / `_cells_since_hazard` trackers and makes every pair knob look inert.
  * counts are in STRUCTURE units (adjacent same-code cells collapsed) -- a 2-cell HoleLong would
    otherwise read as a Hole->Hole pair and roughly triple that rate.
  * shares are JOINT (share of ALL consecutive structure pairs), which is how the device column is
    defined. The CONDITIONAL rate is a different quantity (~ the type weight).
  * `_distance_real` is pinned to 3000 (the late ramp), where cascades actually occur.

    python experiments/calib_type_reject.py --report            # baseline vs each candidate
    python experiments/calib_type_reject.py --fit               # iterative coordinate fit
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from templerun_rl.sim_env import TempleRunSimEnv  # noqa: E402

MUST_JUMP = {1, 3, 4, 7}
MUST_SLIDE = {5, 6}
SLIDES = MUST_SLIDE
NAME = {1: "Hole", 2: "Narrow", 3: "Stumble", 4: "TreeJump", 5: "TreeSlide", 6: "Flame",
        7: "Crevasse"}

DEVICE = {"slide_led": 0.507, "slide_repeat": 0.179, "hole_hole": 0.061, "same_type": 0.246}


# ------------------------------------------------------------------ measurement
def structs_from_seq(seq):
    """Collapse adjacent same-code hazard cells into ONE structure (spec §5)."""
    out, i = [], 0
    while i < len(seq):
        o = seq[i]
        if o in MUST_JUMP or o in MUST_SLIDE:
            j = i
            while j + 1 < len(seq) and seq[j + 1] == o:
                j += 1
            out.append(o)
            i = j + 1
        else:
            i += 1
    return out


def _assert_cfg(env, cfg):
    """Spec §5: prove the config actually landed before measuring anything."""
    for k, v in cfg.items():
        got = env.cfg.get(k)
        if isinstance(v, dict):
            assert dict(got or {}) == dict(v), f"config {k} not applied: {got!r} != {v!r}"
        elif isinstance(v, float):
            assert abs(float(got) - v) < 1e-12, f"config {k} not applied: {got!r} != {v!r}"
        else:
            assert got == v, f"config {k} not applied: {got!r} != {v!r}"


def measure(cfg, ncells=120000, seeds=(4242, 91, 7), start=3000.0):
    pairs = Counter()
    total_cells = 0
    for sd in seeds:
        env = TempleRunSimEnv(config=cfg)          # `config=`, NOT `sim_config=`
        env.reset(seed=sd)
        _assert_cfg(env, cfg)
        env._distance_real = start                 # late ramp; held, so one regime
        seq = []
        for _ in range(ncells):
            cell = env._next_cell()                # NOT _next_cell_procedural()
            seq.append(int(cell.get("obstacles", 0)))   # `obstacles`, plural
            env._distance_real = start
        total_cells += len(seq)
        st = structs_from_seq(seq)
        for a, b in zip(st, st[1:]):
            pairs[(a, b)] += 1
    tot = sum(pairs.values())
    # ~0.10 hazard structures per cell in this world; anything near zero means the generator is not
    # actually running (the `_next_cell_procedural` / silently-ignored-config traps).
    assert tot > 0.02 * total_cells, f"only {tot} pairs over {total_cells} cells -- generation stalled"
    s = lambda pred: sum(v for k, v in pairs.items() if pred(k)) / tot          # noqa: E731
    return {
        "pairs": pairs, "n_pairs": tot, "n_cells": total_cells,
        "slide_led": s(lambda k: k[0] in SLIDES),
        "slide_repeat": s(lambda k: k[0] == k[1] and k[0] in SLIDES),
        "hole_hole": s(lambda k: k == (1, 1)),
        "same_type": s(lambda k: k[0] == k[1]),
        "ts_ts": s(lambda k: k == (5, 5)),
        "fl_fl": s(lambda k: k == (6, 6)),
    }


def loss(m):
    """Spec §3 priority order: slide-led > slide-repeat > Hole->Hole > same-type."""
    w = {"slide_led": 8.0, "slide_repeat": 4.0, "hole_hole": 2.0, "same_type": 1.0}
    return sum(wt * (m[k] - DEVICE[k]) ** 2 for k, wt in w.items())


# ------------------------------------------------------------------ reporting
def row(label, m):
    print(f"  {label:<34} {100*m['slide_led']:6.1f} {100*m['slide_repeat']:6.1f} "
          f"{100*m['hole_hole']:6.1f} {100*m['same_type']:6.1f}   "
          f"[{100*m['ts_ts']:.1f}/{100*m['fl_fl']:.1f}]  n={m['n_pairs']}")


def header():
    print(f"  {'':<34} {'sldLed':>6} {'sldRep':>6} {'HoleHo':>6} {'sameTy':>6}   [TS->TS/Fl->Fl]")
    print(f"  {'DEVICE TARGET':<34} {50.7:6.1f} {17.9:6.1f} {6.1:6.1f} {24.6:6.1f}   [  9.8/  8.1]")


def load_base(path):
    cfg = json.load(open(path))
    cfg = {k: v for k, v in cfg.items()}
    cfg["obstacle_type_weights"] = {int(k): float(v)
                                    for k, v in cfg["obstacle_type_weights"].items()}
    cfg["episode_profiles"] = []          # CPU ref has none; keep one regime
    cfg["domain_rand"] = False
    cfg["curriculum_max_start_dist"] = 0.0
    return cfg


# rejectable `cur` types: the GEOMETRY half of the weight table (trees are the unfiltered pass).
CUR = (1, 2, 6)
PREV = (1, 3, 4, 5, 6, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=str(_ROOT / "experiments" / "_v108_cfg.json"),
                    help="the champion world to calibrate inside")
    ap.add_argument("--ncells", type=int, default=120000)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--redraws", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    base = load_base(a.cfg)
    print(f"\nCONFIG {Path(a.cfg).name}  |  ncells={a.ncells} x 3 seeds  |  _distance_real=3000\n")
    header()
    m0 = measure(base, a.ncells)
    row("sim now (pair_same_type 0.6)", m0)

    off = dict(base, pair_same_type_prob=0.0)
    row("sim, pair_same_type 0.0 (i.i.d.)", measure(off, a.ncells))
    sat = dict(base, pair_same_type_prob=1.0)
    row("sim, pair_same_type 1.0 (saturated)", measure(sat, a.ncells))

    if not a.fit:
        return

    # ---------------------------------------------------------- coordinate fit
    # The rejection probability is NOT the emitted share (the calib_mix.py lesson), so fit it
    # iteratively: sweep one (prev, cur) entry at a time on a coarse grid, keep the best.
    print(f"\nFITTING type_reject (redraws={a.redraws}) -- coordinate descent, "
          f"{a.rounds} rounds over {len(PREV)}x{len(CUR)} entries")
    tbl = {}
    best = dict(base, type_reject_model=True, type_reject={}, type_reject_redraws=a.redraws,
                pair_same_type_prob=0.0, pair_anti_repeat_prob=0.0)
    bm = measure(best, a.ncells)
    bl = loss(bm)
    print(f"  start loss {bl:.5f}")
    grid = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.98]
    for r in range(a.rounds):
        for p in PREV:
            for c in CUR:
                cur_best, cur_loss = tbl.get(f"{p},{c}", 0.0), bl
                for g in grid:
                    t2 = dict(tbl)
                    if g:
                        t2[f"{p},{c}"] = g
                    else:
                        t2.pop(f"{p},{c}", None)
                    if t2 == tbl:
                        continue
                    cfg2 = dict(best, type_reject=t2)
                    l2 = loss(measure(cfg2, a.ncells // 4))
                    if l2 < cur_loss:
                        cur_loss, cur_best = l2, g
                if cur_best:
                    tbl[f"{p},{c}"] = cur_best
                else:
                    tbl.pop(f"{p},{c}", None)
                bl = cur_loss
        m = measure(dict(best, type_reject=tbl), a.ncells)
        bl = loss(m)
        print(f"  round {r}: loss {bl:.5f}  table={json.dumps(tbl, sort_keys=True)}")
        row(f"  after round {r}", m)

    final = dict(best, type_reject=tbl)
    print("\nFINAL")
    header()
    row("sim now (pair_same_type 0.6)", m0)
    row(f"type_reject fitted (redraws={a.redraws})", measure(final, a.ncells))
    print("\n  type_reject = " + json.dumps(tbl, sort_keys=True))
    out = _ROOT / "experiments" / f"_type_reject_fit_r{a.redraws}.json"
    json.dump({"type_reject_model": True, "type_reject_redraws": a.redraws,
               "type_reject": tbl}, open(out, "w"), indent=1, sort_keys=True)
    print(f"  written {out}")


if __name__ == "__main__":
    main()
