#!/usr/bin/env python3
"""OBSTACLE-PAIR COMPOSITION: what follows what, device vs sim.

USER (2026-07-27, watching v104 on the phone): "The struggle is really when you have 3 obstacles in a
row. It really handles it poorly. I don't know how you do it but you really have to randomize the
obstacles that follow one another. You shall not skip the 2 obstacles in a row training!"

The v103 chain profile raised chain FREQUENCY (0.83 -> 2.72 chains/km) but never touched COMPOSITION,
and composition is where the sim is structurally poor: every tight pair it can emit comes from a
BESPOKE compound with a FIXED type ordering --
    jumpslide  = wide-gap -> TreeSlide/Flame        jumpnarrow = wide-gap -> narrow strip
    jumpgap    = wide-gap -> HoleLong               slidegap   = slide   -> hole
    compound   = narrow -> jump barrier -> narrow   turnobstacle = junction -> obstacle
so the second element is DETERMINED by the first. The device draws both ends from its full obstacle
mix, which is a far larger set of orderings. This measures that directly, over hazard STRUCTURES
(a 2-cell crevasse / HoleLong is ONE obstacle, not two).

Reports, for pairs within --gap clear cells: the (A -> B) distribution on each side, the device
pair-types the sim NEVER emits, and a diversity number (how many distinct orderings cover 90% of the
mass). That last one is the honest summary of "is it randomized or is it 5 canned patterns".

    python experiments/pair_coverage.py --device "captures/human_20260708_*.jsonl" "captures/w17_v*_device.jsonl" \
                                       --sim-config experiments/_v105_cfg.json
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse, glob, json, sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from hole_landing_forensic import load_eps, cell_seq  # noqa: E402

MUST_JUMP = {1, 3, 4, 7}
MUST_SLIDE = {5, 6}
NAME = {1: "Hole", 3: "Stumble", 4: "TreeJump", 5: "TreeSlide", 6: "Flame", 7: "Crevasse"}


def structs_from_seq(seq):
    """seq = list of obstacle codes per cell (0 = clear). Collapse adjacent same-code runs."""
    out, i = [], 0
    while i < len(seq):
        o = seq[i]
        if o in MUST_JUMP or o in MUST_SLIDE:
            j = i
            while j + 1 < len(seq) and seq[j + 1] == o:
                j += 1
            out.append((i, j, o))
            i = j + 1
        else:
            i += 1
    return out


def pairs_from_structs(st, gap):
    for a, b in zip(st, st[1:]):
        clear = b[0] - a[1] - 1
        if 0 <= clear <= gap:
            yield (a[2], b[2]), clear


def device_pairs(globs, gap):
    c, cells = Counter(), 0
    for g in globs:
        for p in sorted(glob.glob(g)):
            for _, ep in load_eps(p):
                cs = cell_seq(ep)
                per = {}
                for i, fr in enumerate(ep):
                    o = int(fr.get("obstacle") or 0)
                    if fr.get("path_type") in (33, 34, 35, 36):
                        o = 7
                    if cs[i] not in per or (not per[cs[i]] and o):
                        per[cs[i]] = o
                seq = [per[k] for k in sorted(per)]
                cells += len(seq)
                for key, _ in pairs_from_structs(structs_from_seq(seq), gap):
                    c[key] += 1
    return c, cells


def sim_pairs(cfg_path, gap, ncells, start):
    from templerun_rl.sim_env import TempleRunSimEnv
    cfg = json.load(open(cfg_path))
    cfg.update({"episode_profiles": [], "domain_rand": False, "curriculum_max_start_dist": 0.0})
    env = TempleRunSimEnv(config=cfg)
    env.reset(seed=4242)
    env._distance_real = start
    seq = []
    for _ in range(ncells):
        # NOTE: _next_cell(), not _next_cell_procedural() -- the pair_same_type_prob trackers live
        # in _next_cell (the single point every generated cell passes through). Calling the
        # procedural generator directly silently disables that mechanism in the audit.
        cell = env._next_cell()
        seq.append(int(cell.get("obstacles", 0)))
        env._distance_real += 8.7
    c = Counter()
    for key, _ in pairs_from_structs(structs_from_seq(seq), gap):
        c[key] += 1
    return c, len(seq)


def show(c, cells, label):
    tot = sum(c.values())
    print(f"\n=== {label} ===  {tot} obstacle PAIRS over {cells} cells "
          f"({1000*tot/max(1,cells):.1f} per 1000 cells)")
    if not tot:
        return {}
    share = {k: v / tot for k, v in c.items()}
    run, n90 = 0.0, 0
    for k, v in sorted(share.items(), key=lambda kv: -kv[1]):
        run += v
        n90 += 1
        if run >= 0.90:
            break
    print(f"  distinct orderings: {len(c)}   |   {n90} of them cover 90% of the mass")
    for k, v in c.most_common(10):
        print(f"     {NAME.get(k[0],k[0]):9} -> {NAME.get(k[1],k[1]):9} {v:5}  {100*share[k]:5.1f}%")
    return share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", nargs="+", required=True)
    ap.add_argument("--sim-config", required=True)
    ap.add_argument("--gap", type=int, default=3, help="max CLEAR cells between the two obstacles")
    ap.add_argument("--ncells", type=int, default=40000)
    ap.add_argument("--start", type=float, default=5000.0)
    a = ap.parse_args()
    dc, dcells = device_pairs(a.device, a.gap)
    sc, scells = sim_pairs(a.sim_config, a.gap, a.ncells, a.start)
    ds = show(dc, dcells, "DEVICE + HUMAN")
    ss = show(sc, scells, f"SIM ({Path(a.sim_config).name})")
    print("\n--- device orderings the sim NEVER emits (by device share) ---")
    missing = [(k, v) for k, v in sorted(ds.items(), key=lambda kv: -kv[1]) if k not in sc]
    if not missing:
        print("   none")
    for k, v in missing[:15]:
        print(f"     {NAME.get(k[0],k[0]):9} -> {NAME.get(k[1],k[1]):9}  {100*v:5.1f}% of device pairs")
    print(f"   TOTAL uncovered: {100*sum(v for _, v in missing):.1f}% of device pairs")


if __name__ == "__main__":
    main()
