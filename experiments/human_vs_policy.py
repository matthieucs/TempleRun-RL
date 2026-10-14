#!/usr/bin/env python3
"""PAIRED HUMAN-vs-POLICY timing analysis (2026-07-09, USER-directed pivot).

Compares HOW the human survives the contexts that kill the policy — especially long-hole
(pt12/13/15/16) stacks, the shared #1 death mode. Both input kinds are capture JSONLs:
  - human: scripts/device/record_human_state.py output (30 Hz, action=None, has gi_* swipe intent)
  - policy: scripts/device/capture_calib.py output (10 Hz control steps, explicit action)

Jump events = is_jumping rising edges (identical semantic on both). All gaps expressed in
SECONDS (human t field @30Hz; policy step*0.1s) and cp units so the rates compare.

Usage: python experiments/human_vs_policy.py --human captures/human_*.jsonl --policy captures/ab_v66_*.jsonl
"""
from __future__ import annotations
import argparse, glob, json, math, sys
from collections import defaultdict

HOLE_PT = set(range(11, 17))
LONG_HOLE_PT = {12, 13, 15, 16}
CREVASSE_PT = set(range(33, 37))


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def pctl(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def load(path):
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def episodes(frames):
    """Split on the ep field; keep only alive frames per ep (plus the death frame)."""
    eps = defaultdict(list)
    for fr in frames:
        eps[fr.get("ep", 0)].append(fr)
    out = []
    for k in sorted(eps):
        fs = eps[k]
        # trim leading dead/menu frames
        i = 0
        while i < len(fs) and fs[i].get("is_dead"):
            i += 1
        # trim to first death (inclusive)
        j = i
        while j < len(fs) and not fs[j].get("is_dead"):
            j += 1
        seg = fs[i:min(j + 1, len(fs))]
        if len(seg) > 30:
            out.append(seg)
    return out


def tof(fr, dt_fallback, prev_t=None):
    return fr.get("t")


def analyze(paths, kind):
    """kind: 'human' (30Hz, t field) or 'policy' (10Hz, step*0.1)."""
    R = {
        "eps": 0, "dists": [], "deaths": [],          # (dist, pt, dt)
        "jumps": [],                                    # dicts per jump event
        "slides": 0,
        "hole_encounters": 0, "hole_deaths": 0,
        "longhole_encounters": 0, "longhole_deaths": 0,
        "stumbles": 0, "meters": 0.0,
        "rejump_gaps": [],                              # seconds grounded between landing and next takeoff
        "airtimes": [],
        "chain_step_gaps": [],                          # rejump gaps specifically inside hole chains
        "held_frac_n": 0, "held_frac_d": 0,             # policy only: jump pressed while airborne
        "swipe_lat": [],                                # human only: gi_t_should_jump at takeoff
        "coin_lane_match": [0, 0],                      # [matched, total] large excursions w/ coins present
    }
    for path in paths:
        frames = load(path)
        for ep in episodes(frames):
            R["eps"] += 1
            dist_end = float(ep[-1].get("distance") or 0)
            R["dists"].append(dist_end)
            R["meters"] += dist_end
            death = ep[-1] if ep[-1].get("is_dead") else None
            if death is not None:
                R["deaths"].append((dist_end, death.get("path_type"), death.get("death_type")))

            # per-frame time
            def T(i):
                if kind == "human":
                    return ep[i].get("t") or (i / 30.0)
                return ep[i].get("step", i) * 0.1

            prev_jump = False
            prev_slide = False
            last_land_t = None
            last_takeoff_t = None
            last_stumble_cnt = 0
            seen_cells = set()      # (pt bucket, approx cell id) to count encounters once
            last_pt = None
            last_cell_key = None
            in_chain = False        # >=2 hole cells within the last ~2 cells

            recent_holes = []       # takeoff times of hole cells passed, to detect chains

            for i, fr in enumerate(ep):
                pt = fr.get("path_type")
                jumping = bool(fr.get("is_jumping"))
                sliding = bool(fr.get("is_sliding"))
                dist = float(fr.get("distance") or 0)
                vel = float(fr.get("velocity") or 0)
                cp = fr.get("cell_progress")
                t = T(i)

                # --- cell-encounter counting (new cell = pt change or cp wrap) ---
                cell_key = (pt, int(dist // 8))  # ~cell length 8: coarse but stable
                if pt in HOLE_PT and cell_key != last_cell_key:
                    R["hole_encounters"] += 1
                    if pt in LONG_HOLE_PT:
                        R["longhole_encounters"] += 1
                    recent_holes.append(t)
                last_cell_key = cell_key

                # --- stumbles ---
                sc = fr.get("stumble_count") or 0
                if sc > last_stumble_cnt:
                    R["stumbles"] += sc - last_stumble_cnt
                last_stumble_cnt = max(last_stumble_cnt, sc)

                # --- jump edges ---
                if jumping and not prev_jump:
                    # takeoff
                    prev = ep[i - 1] if i > 0 else fr
                    tk_cp = prev.get("cell_progress")
                    tk_pt = prev.get("path_type")
                    nx_pt = prev.get("next_path_type")
                    gap_now = tk_pt in HOLE_PT or tk_pt in CREVASSE_PT
                    gap_next = (nx_pt in HOLE_PT or nx_pt in CREVASSE_PT) if nx_pt is not None else \
                        bool(prev.get("next_must_jump"))
                    rejump_gap = (t - last_land_t) if last_land_t is not None else None
                    is_chain = len([h for h in recent_holes if t - h < 2.5]) >= 2
                    R["jumps"].append({
                        "t": t, "dist": dist, "vel": vel, "cp": tk_cp,
                        "on_gap": gap_now, "before_gap": gap_next, "pt": tk_pt,
                        "level": prev.get("path_level"),
                        "rejump_gap": rejump_gap, "chain": is_chain,
                    })
                    if rejump_gap is not None and rejump_gap < 3.0:
                        R["rejump_gaps"].append(rejump_gap)
                        if is_chain:
                            R["chain_step_gaps"].append(rejump_gap)
                    if kind == "human":
                        gj = prev.get("gi_t_should_jump")
                        if gj is not None and 0 <= gj < 1.0:
                            R["swipe_lat"].append(gj)
                    last_takeoff_t = t
                if (not jumping) and prev_jump:
                    last_land_t = t
                    if last_takeoff_t is not None:
                        at = t - last_takeoff_t
                        if 0.1 < at < 2.0:
                            R["airtimes"].append(at)
                prev_jump = jumping

                if sliding and not prev_slide:
                    R["slides"] += 1
                prev_slide = sliding

                # --- policy: held-jump (action==2 while airborne) ---
                if kind == "policy" and jumping:
                    R["held_frac_d"] += 1
                    if fr.get("action") == 2:
                        R["held_frac_n"] += 1

                # --- human: coin-vs-lane for big excursions ---
                if kind == "human":
                    lane = fr.get("lane")
                    if lane is not None and abs(lane) > 4.5:
                        R["coin_lane_match"][1] += 1
                        if fr.get("has_coins") or fr.get("next_has_coins"):
                            R["coin_lane_match"][0] += 1

            # death attribution to holes
            if death is not None:
                dpt = death.get("path_type")
                if dpt in HOLE_PT:
                    R["hole_deaths"] += 1
                    if dpt in LONG_HOLE_PT:
                        R["longhole_deaths"] += 1
    return R


def vbucket(v):
    if v < 130: return "<130"
    if v < 170: return "130-170"
    if v < 200: return "170-200"
    return "200+"


def report(name, R):
    print(f"\n{'='*74}\n{name}: {R['eps']} eps | median dist {median(R['dists']):.0f} | "
          f"max {max(R['dists']) if R['dists'] else 0:.0f} | total {R['meters']:.0f} m")
    print(f"  deaths: {len(R['deaths'])} | on holes {R['hole_deaths']} "
          f"(long {R['longhole_deaths']}) | stumbles/1000m {1000*R['stumbles']/max(R['meters'],1):.2f}")
    he, hd = R["hole_encounters"], R["hole_deaths"]
    le, ld = R["longhole_encounters"], R["longhole_deaths"]
    print(f"  hole encounters {he} -> death rate {100*hd/max(he,1):.2f}% | "
          f"long-hole {le} -> {100*ld/max(le,1):.2f}%")
    print(f"  jumps {len(R['jumps'])} ({1000*len(R['jumps'])/max(R['meters'],1):.1f}/1000m) | "
          f"slides {R['slides']} | airtime med {median(R['airtimes']):.2f}s")

    # takeoff cp before gaps, by speed
    by_v = defaultdict(list)
    for j in R["jumps"]:
        if j["before_gap"] and not j["on_gap"] and j["cp"] is not None:
            by_v[vbucket(j["vel"])].append(j["cp"])
    print("  takeoff cp on APPROACH cell before a gap (med [p25,p75], n):")
    for k in ["<130", "130-170", "170-200", "200+"]:
        xs = by_v.get(k, [])
        if xs:
            print(f"    v{k:8}: {median(xs):.2f} [{pctl(xs,0.25):.2f},{pctl(xs,0.75):.2f}] n={len(xs)}")
    on_gap = [j for j in R["jumps"] if j["on_gap"]]
    print(f"  jumps taken ON the gap cell itself (late takeoff): {len(on_gap)} "
          f"({100*len(on_gap)/max(len(R['jumps']),1):.1f}% of jumps)")

    rg = R["rejump_gaps"]
    print(f"  re-jump grounded gap (s): med {median(rg):.2f} p10 {pctl(rg,0.10):.2f} "
          f"p25 {pctl(rg,0.25):.2f} n={len(rg)} | <=0.15s (instant re-jump): "
          f"{100*sum(1 for x in rg if x <= 0.15)/max(len(rg),1):.1f}%")
    cg = R["chain_step_gaps"]
    if cg:
        print(f"  ... inside HOLE CHAINS: med {median(cg):.2f} p10 {pctl(cg,0.10):.2f} n={len(cg)} | "
              f"<=0.15s: {100*sum(1 for x in cg if x <= 0.15)/max(len(cg),1):.1f}%")
    if R["held_frac_d"]:
        print(f"  policy held-jump while airborne: {100*R['held_frac_n']/R['held_frac_d']:.1f}% of airborne frames")
    if R["swipe_lat"]:
        print(f"  human swipe->takeoff latency: med {median(R['swipe_lat']):.3f}s n={len(R['swipe_lat'])}")
    m, tot = R["coin_lane_match"]
    if tot:
        print(f"  human big lane excursions (|lane|>4.5): {100*m/tot:.1f}% have coins on cell/next ({m}/{tot} frames)")
    # death table
    print("  deaths (dist, pt, dt):", sorted(R["deaths"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", nargs="+", required=True)
    ap.add_argument("--policy", nargs="+", required=True)
    a = ap.parse_args()
    hpaths = [p for g in a.human for p in sorted(glob.glob(g))]
    ppaths = [p for g in a.policy for p in sorted(glob.glob(g))]
    print("human files:", hpaths)
    print("policy files:", ppaths)
    RH = analyze(hpaths, "human")
    RP = analyze(ppaths, "policy")
    report("HUMAN", RH)
    report("POLICY", RP)


if __name__ == "__main__":
    main()
