#!/usr/bin/env python3
"""Analyze a capture_calib.py JSONL: extract the two sim-fidelity measurements.

  1. CREVASSE / level transition (obstacle==7, path_type 33-36): real gap SPAN (cells crossed while
     on the gap) and the jump cell_progress that clears vs falls in, split Horz (33/34) vs Vert (35/36).
  2. NARROWS (path_type 17-30): for each subtype x facing, which lane bucket SURVIVED vs fell off, plus
     the hook's narrow_target_lane guess and the strip LENGTH (cells). Rebuilds the safe-lane truth.

    python scripts/analysis/analyze_calib.py captures/calib.jsonl
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# gap path_types we've SEEN referenced (same-level HoleLong 12/13/15/16; altitude StepCap 33-36).
# These are starting guesses — the report PRINTS the actual device path_types so we learn the truth.
GAP_PT = {12, 13, 15, 16, 33, 34, 35, 36}
NARROW_LO, NARROW_HI = 17, 30
FACING = {1: "N", 2: "E", 3: "S", 4: "W", 0: "-", None: "-"}
LEVELS = {1: "low", 2: "mid", 3: "high"}
CP_DROP = 0.30                    # cell_progress fall this big = a new cell boundary


def lvl(v) -> str:
    return LEVELS.get(v, str(v))


def load(path: str):
    """Return {ep: [frame, ...]} preserving order."""
    eps = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        eps[r["ep"]].append(r)
    return eps


def died(frames) -> bool:
    last = frames[-1]
    return bool(last.get("is_dead") or last.get("is_game_over"))


def cells_in(span) -> int:
    """Number of distinct cells covered by a contiguous frame span (count cell_progress resets)."""
    drops = 0
    prev = None
    for fr in span:
        cp = fr.get("cell_progress")
        if cp is None:
            continue
        if prev is not None and cp + CP_DROP < prev:
            drops += 1
        prev = cp
    return drops + 1


def events(frames, pred):
    """Yield maximal contiguous frame-runs (bridging <=1 gap frame) where pred(frame) holds."""
    run, gap = [], 0
    for fr in frames:
        if pred(fr):
            run.append(fr)
            gap = 0
        elif run:
            gap += 1
            if gap > 1:
                yield run
                run, gap = [], 0
    if run:
        yield run


def recent_takeoff_cp(frames, i0):
    """cell_progress at the most recent jump takeoff (is_jumping rising edge) at/just before frame index i0."""
    for j in range(i0, max(-1, i0 - 12), -1):
        if j <= 0:
            break
        if frames[j].get("is_jumping") and not frames[j - 1].get("is_jumping"):
            return frames[j].get("cell_progress")
    return None


def crevasse_stats(eps) -> dict:
    """Machine-readable crevasse measurements (altitude transitions + same-level gaps).
    Used by both crevasse_report() and extract_device_reference.py."""
    trans = Counter()                                  # (from,to) -> n
    trans_pt = defaultdict(Counter)                    # (from,to) -> path_types around it
    trans_obst = defaultdict(Counter)                  # (from,to) -> obstacle values around it
    trans_out = defaultdict(lambda: [0, 0])            # (from,to) -> [surv, died]
    span_by = defaultdict(list)                        # (from,to) -> [cell-span,...]
    cp_by = defaultdict(lambda: {"surv": [], "died": []})

    for ep, frames in eps.items():
        last_change = -10
        for i in range(1, len(frames)):
            a_, b_ = frames[i - 1].get("path_level"), frames[i].get("path_level")
            if a_ is None or b_ is None or a_ == b_ or a_ < 0 or b_ < 0:
                continue
            if i - last_change < 5:                    # debounce flicker
                continue
            last_change = i
            key = (a_, b_)
            trans[key] += 1
            win = frames[max(0, i - 8): i + 8]
            for f in win:
                pt = f.get("path_type")
                if pt is not None and pt >= 0:
                    trans_pt[key][pt] += 1
                trans_obst[key][f.get("obstacle")] += 1
            gap = [f for f in win if int(f.get("obstacle") or 0) == 7
                   or int(f.get("path_type") or -1) in GAP_PT]
            if gap:
                span_by[key].append(cells_in(gap))
            cp = recent_takeoff_cp(frames, i)
            died_here = died(frames) and any(f.get("is_dead") for f in frames[i: i + 8])
            trans_out[key][1 if died_here else 0] += 1
            if cp is not None:
                cp_by[key]["died" if died_here else "surv"].append(round(float(cp), 2))

    altitude = {}
    for key in sorted(trans):
        fr, to = key
        s, d = trans_out[key]
        altitude[f"{fr}->{to}"] = {
            "n": trans[key], "survived": s, "fell": d,
            "path_types": {str(k): v for k, v in trans_pt[key].most_common(6)},
            "obstacles": {str(k): v for k, v in trans_obst[key].items()},
            "gap_cell_span": {str(k): v for k, v in Counter(span_by[key]).items()},
            "takeoff_cprog_surv": sorted(cp_by[key]["surv"]),
            "takeoff_cprog_died": sorted(cp_by[key]["died"]),
        }

    spans, pts = [], Counter()
    out = [0, 0]
    cp_b = {"surv": [], "died": []}
    for ep, frames in eps.items():
        idx = {id(fr): i for i, fr in enumerate(frames)}
        for ev in events(frames, lambda f: int(f.get("obstacle") or 0) == 7
                         or int(f.get("path_type") or -1) in GAP_PT):
            lv = {f.get("path_level") for f in ev}
            if len(lv) > 1:                            # has a level change -> covered in altitude section
                continue
            spans.append(cells_in(ev))
            for f in ev:
                pts[f.get("path_type")] += 1
            i0, iN = idx[id(ev[0])], idx[id(ev[-1])]
            cp = recent_takeoff_cp(frames, i0)
            fatal = died(frames) and any(f.get("is_dead") for f in frames[iN: iN + 5])
            out[1 if fatal else 0] += 1
            if cp is not None:
                cp_b["died" if fatal else "surv"].append(round(float(cp), 2))
    same_level = {
        "events": sum(out), "survived": out[0], "fell": out[1],
        "cell_span": {str(k): v for k, v in Counter(spans).items()},
        "path_types": {str(k): v for k, v in pts.items()},
        "takeoff_cprog_surv": sorted(cp_b["surv"]),
        "takeoff_cprog_died": sorted(cp_b["died"]),
    }
    return {"altitude": altitude, "same_level": same_level}


def crevasse_report(eps) -> None:
    """Discover crevasse types by ALTITUDE TRANSITION (path_level change) + same-level gaps. Prints the
    real device path_types/obstacles so we don't rely on assumed enum values."""
    st = crevasse_stats(eps)
    print("\n=== CREVASSES — A) altitude transitions (path_level change) ===")
    if not st["altitude"]:
        print("  (none — runs never reached an altitude change)")
    for key, d in st["altitude"].items():
        fr, to = (int(x) for x in key.split("->"))
        print(f"  {lvl(fr)}->{lvl(to)} ({key}): n={d['n']}  survived={d['survived']} fell={d['fell']}")
        print(f"      path_types={d['path_types']}  obstacles={d['obstacles']}")
        if d["gap_cell_span"]:
            print(f"      gap cell-span={d['gap_cell_span']}  (span>=2 => need wide_gap_extra_cells>=1)")
        print(f"      jump-takeoff cprog  survived={d['takeoff_cprog_surv']} fell={d['takeoff_cprog_died']}")

    b = st["same_level"]
    print("\n=== CREVASSES — B) same-level gaps (obstacle==7 or HoleLong, NO level change) ===")
    print(f"  events={b['events']} survived={b['survived']} fell={b['fell']}  cell-span={b['cell_span']}")
    print(f"  path_types={b['path_types']}")
    print(f"  jump-takeoff cprog  survived={b['takeoff_cprog_surv']} fell={b['takeoff_cprog_died']}")


def narrow_stats(eps) -> dict:
    """Machine-readable narrow safe-lane table: {"<subtype>,<facing>": {...}} + strip lengths.
    Used by both narrow_report() and extract_device_reference.py."""
    # table[(subtype, facing)][lane_bucket] = [n_survived, n_died]
    table = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    guess = defaultdict(Counter)          # hook narrow_target_lane guess per (subtype, facing)
    lane_vals = defaultdict(list)         # (subtype, facing) -> continuous lane values held on SURVIVED strips
    lengths = []
    for ep, frames in eps.items():
        idx = {id(fr): i for i, fr in enumerate(frames)}
        for ev in events(frames, lambda f: NARROW_LO <= int(f.get("path_type") or -1) <= NARROW_HI):
            lengths.append(cells_in(ev))
            # subtype/facing taken from the first on-narrow frame; lane = the bucket held across it.
            sub = int(ev[0].get("path_type") or -1)
            fac = FACING.get(ev[0].get("facing"), "-")
            buckets = Counter(f.get("lane_bucket", "?") for f in ev)
            held = buckets.most_common(1)[0][0]
            tail = frames[idx[id(ev[-1])]: idx[id(ev[-1])] + 5]
            fatal = died(frames) and any(f.get("is_dead") for f in tail)
            table[(sub, fac)][held][1 if fatal else 0] += 1
            if not fatal:
                lane_vals[(sub, fac)] += [round(float(f["lane"]), 2) for f in ev
                                          if f.get("lane") is not None]
            # hook's guess: narrow_target_lane is set on the APPROACH frames (next_path_type in 17-30),
            # just BEFORE the on-narrow event — scan back from the event start.
            i0 = idx[id(ev[0])]
            for j in range(i0, max(-1, i0 - 6), -1):
                g = frames[j].get("narrow_target_lane")
                if g is not None:
                    guess[(sub, fac)][{-1: "L", 0: "C", 1: "R"}.get(int(g), "?")] += 1
                    break
    out = {}
    for key in sorted(table):
        sub, fac = key
        cells = table[key]
        safe, best = "?", -1
        for b in ("L", "C", "R"):
            sv, dd = cells[b]
            if sv + dd and sv > best and dd == 0:
                best, safe = sv, b
        lv = sorted(lane_vals[key])
        out[f"{sub},{fac}"] = {
            "lanes": {b: cells[b] for b in ("L", "C", "R") if sum(cells[b])},
            "hook_guess": guess[key].most_common(1)[0][0] if guess[key] else "-",
            "safe": safe,
            "surv_lane_band": [lv[0], lv[-1]] if lv else None,
        }
    return {"strip_len": {str(k): v for k, v in Counter(lengths).items()}, "map": out}


def narrow_report(eps) -> None:
    st = narrow_stats(eps)
    print("\n=== NARROWS (path_type 17-30): subtype x facing -> surviving lane ===")
    if st["strip_len"]:
        print(f"strip length (cells) distribution = {st['strip_len']}")
    print(f"{'sub':>4} {'fac':>3} | {'lane: surv/died':<28} | hook_guess | inferred_safe")
    for key, d in st["map"].items():
        sub, fac = key.split(",")
        parts = [f"{b}:{sv}/{dd}" for b, (sv, dd) in d["lanes"].items()]
        # only flag a contradiction when we know BOTH the safe lane and the hook's guess.
        flag = "  <-- HOOK MAP WRONG" if (d["safe"] in ("L", "C", "R") and d["hook_guess"] in ("L", "C", "R")
                                          and d["safe"] != d["hook_guess"]) else ""
        print(f"{sub:>4} {fac:>3} | {' '.join(parts):<28} | {d['hook_guess']:^10} | {d['safe']}{flag}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    a = p.parse_args()
    eps = load(a.jsonl)
    n_frames = sum(len(v) for v in eps.values())
    n_dead = sum(died(v) for v in eps.values())
    print(f"loaded {len(eps)} episodes, {n_frames} frames, {n_dead} ended in death")
    crevasse_report(eps)
    narrow_report(eps)


if __name__ == "__main__":
    main()
