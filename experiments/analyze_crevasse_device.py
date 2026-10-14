#!/usr/bin/env python3
"""Device crevasse forensics from a capture_calib JSONL (hook 16z). Measures the pieces the sim's
crevasse fix needs (v55 lesson: a fixed cap-arc offset failed; the real device has takeoff-timing
VARIANCE -> a fraction grazes). Per crevasse encounter (path_type 33-36):
  * takeoff cell_progress + velocity  -> the VARIANCE distribution (mean/std/percentiles)
  * arc clearance at the cap = min(world_y - ground_height) during the encounter (neg = descended
    below the raised cap top = graze zone)
  * outcome: clear / graze (is_stumbling fired, non-fatal) / kill (is_dead or death_type==4)
grouped BY path_level (high-ground lvl-3 caps graze ~32% on device; lvl-2 ~0%)."""
import json, sys, statistics as S
from pathlib import Path
from collections import defaultdict

path = Path(sys.argv[1] if len(sys.argv) > 1 else "captures/v51_crevasse_16z.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
CREV = {33, 34, 35, 36}


def num(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


# group rows by episode, in step order
eps = defaultdict(list)
for r in rows:
    eps[r.get("ep")].append(r)

by_level = defaultdict(lambda: {"clear": 0, "graze": 0, "kill": 0, "takeoff_cp": [], "takeoff_v": [],
                                "clearance": [], "gh": []})
n_enc = 0
for ep, frames in eps.items():
    frames.sort(key=lambda r: r.get("step", 0))
    i = 0
    while i < len(frames):
        pt = int(num(frames[i].get("path_type"), -1))
        if pt not in CREV:
            i += 1
            continue
        # ---- one crevasse encounter = contiguous crevasse-pt frames ----
        j = i
        while j < len(frames) and int(num(frames[j].get("path_type"), -1)) in CREV:
            j += 1
        enc = frames[i:j]
        n_enc += 1
        # bucket by the CAP level: a step-UP crevasse (first one ~587m) has a level-2 GAP but a level-3
        # raised CAP (ground_height 15). Use the max level seen on a raised (ground_height>0.5) cell;
        # fall back to the max level in the encounter, then enc[0].
        cap_frames = [f for f in enc if num(f.get("ground_height")) > 0.5]
        if cap_frames:
            lvl = max(int(num(f.get("path_level"), 2)) for f in cap_frames)
        else:
            lvl = max(int(num(f.get("path_level"), 2)) for f in enc)
        gh_max = max(num(f.get("ground_height")) for f in enc)
        # takeoff = last is_jumping 0->1 transition in the ~12 frames before the encounter
        pre = frames[max(0, i - 12):i]
        takeoff = None
        for k in range(1, len(pre)):
            if not bool(pre[k - 1].get("is_jumping")) and bool(pre[k].get("is_jumping")):
                takeoff = pre[k]
        if takeoff is None and enc and bool(enc[0].get("is_jumping")):
            takeoff = enc[0]     # already airborne at entry; use entry cp
        if takeoff is not None:
            by_level[lvl]["takeoff_cp"].append(num(takeoff.get("cell_progress")))
            by_level[lvl]["takeoff_v"].append(num(takeoff.get("velocity")))
        # arc clearance over the cap: min(world_y - ground_height) where ground_height>0 (the raised cap)
        clears = [num(f.get("world_y")) - num(f.get("ground_height"))
                  for f in enc if num(f.get("ground_height")) > 0.5]
        if clears:
            by_level[lvl]["clearance"].append(min(clears))
        by_level[lvl]["gh"].append(gh_max)
        # outcome
        grazed = any(bool(f.get("is_stumbling")) or num(f.get("time_since_stumble"), 9) == 0 for f in enc)
        # kill: death within this encounter or the next 6 frames, as a Ledge(4) or generic death
        tail = frames[i:min(len(frames), j + 6)]
        killed = any(bool(f.get("is_dead")) for f in tail)
        if killed:
            by_level[lvl]["kill"] += 1
        elif grazed:
            by_level[lvl]["graze"] += 1
        else:
            by_level[lvl]["clear"] += 1
        i = j

print(f"{path.name}: {len(rows)} frames, {len(eps)} eps, {n_enc} crevasse encounters\n")
print(f"{'lvl':>3} {'n':>4} {'clear':>5} {'graze':>5} {'kill':>4} {'graze%':>6} {'kill%':>5} "
      f"{'takeoff_cp mean±std (min..max)':>34} {'arc_clear med':>13}")
for lvl in sorted(by_level):
    c = by_level[lvl]
    n = c["clear"] + c["graze"] + c["kill"]
    if n == 0:
        continue
    tc = c["takeoff_cp"]
    tcs = (f"{S.mean(tc):.2f}±{(S.pstdev(tc) if len(tc)>1 else 0):.2f} "
           f"({min(tc):.2f}..{max(tc):.2f}) n={len(tc)}") if tc else "n/a"
    cl = c["clearance"]
    cls = f"{S.median(cl):+.2f}" if cl else "n/a"
    ghmax = max(c["gh"]) if c["gh"] else 0.0
    print(f"{lvl:>3} {n:>4} {c['clear']:>5} {c['graze']:>5} {c['kill']:>4} "
          f"{100*c['graze']/n:>5.0f}% {100*c['kill']/n:>4.0f}% {tcs:>34} {cls:>13}  gh<={ghmax:.0f}")

# pooled takeoff-cp variance (the key knob the sim needs)
allcp = [v for c in by_level.values() for v in c["takeoff_cp"]]
if allcp:
    allcp.sort()
    print(f"\nPOOLED takeoff_cp (n={len(allcp)}): mean={S.mean(allcp):.3f} std={S.pstdev(allcp):.3f} "
          f"p10={allcp[len(allcp)//10]:.2f} p50={S.median(allcp):.2f} p90={allcp[9*len(allcp)//10]:.2f}")
    print("  => the sim needs takeoff_cp SPREAD ~this std, not a fixed offset (v55 failed = no variance).")
