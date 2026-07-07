"""Comprehensive device death-cause forensic: classify EVERY death across captures into edge-case
buckets by reconstructing the obstacle sequence + state leading to it. Ranks the buckets by frequency
and strike-distance so we know which UNMODELLED edge case to close next (user: the median is capped by
how many edge cases the sim doesn't cover -> a run dies when it hits one)."""
import json, sys, glob
from collections import Counter, defaultdict
import statistics as st

DT = {1: "Fall", 2: "Tree", 3: "Slide", 4: "Ledge", 5: "Burnt", 7: "dt7"}

def load_eps(path):
    eps = {}
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        try: r = json.loads(l)
        except: continue
        eps.setdefault(r.get("ep"), []).append(r)
    return eps

def is_crev(pt): return isinstance(pt, int) and 33 <= pt <= 36
def is_hole(pt): return isinstance(pt, int) and 11 <= pt <= 16
def is_narrow(pt): return isinstance(pt, int) and 17 <= pt <= 30
def is_flame(pt): return isinstance(pt, int) and 31 <= pt <= 32

def classify(frames):
    """Return (bucket, dist) for the death of this episode from its last frames."""
    last = frames[-1]
    dt = int(last.get("death_type") or 0)
    pt = last.get("path_type")
    dist = last.get("distance") or 0
    win = frames[-14:]
    # recent obstacle context
    recent_pts = [f.get("path_type") for f in win]
    had_crev = any(is_crev(p) for p in recent_pts)
    had_hole = any(is_hole(p) for p in recent_pts)
    was_jumping = any(f.get("is_jumping") for f in win[-6:])
    # velocity slowdown in the window (a prior graze re-timed things)
    vels = [f.get("velocity") or 0 for f in win if (f.get("velocity") or 0) > 0]
    slowdown = (max(vels) - min(vels)) / max(1.0, max(vels)) if len(vels) >= 2 else 0.0

    if pt == -1:
        return "TURN dead-letter (off-track)", dist
    if dt in (3, 5):
        if had_crev or had_hole:
            return "JUMP->SLIDE (airborne at slide after a jump-obstacle)", dist
        return "SLIDE/FLAME standalone (not sliding)", dist
    if is_crev(pt) or (dt in (2, 4) and had_crev):
        return "CREVASSE (apex/graze/timing)", dist
    if is_narrow(pt) or (dt == 1 and is_narrow(pt)):
        return "NARROW fall (off the strip)", dist
    if is_hole(pt) or (dt in (1, 4) and had_hole):
        return "HOLE (fell in / clipped far wall)", dist
    if dt == 1:
        return "FALL on straight (hole/gap, %s)" % ("post-slowdown" if slowdown > 0.05 else "clean"), dist
    if dt == 2:
        return "TREE (jump/stumble)", dist
    return "OTHER dt=%s pt=%s" % (DT.get(dt, dt), pt), dist

def run(patterns):
    buckets = defaultdict(list)
    n = 0
    for pat in patterns:
        for path in glob.glob(pat):
            for ep, frames in load_eps(path).items():
                if not frames or not frames[-1].get("is_dead"):
                    continue
                # skip degraded-artifact repeats (identical last 2 frames w/ same dist)
                b, d = classify(frames)
                buckets[b].append(d)
                n += 1
    print(f"total deaths classified: {n}\n")
    print("%-52s %5s %7s %7s %s" % ("edge-case bucket", "n", "med_d", "mean_d", "distances"))
    for b in sorted(buckets, key=lambda k: -len(buckets[k])):
        ds = buckets[b]
        print("%-52s %5d %7.0f %7.0f  %s" % (
            b, len(ds), st.median(ds), sum(ds)/len(ds),
            sorted(round(x) for x in ds)))

if __name__ == "__main__":
    pats = sys.argv[1:] or ["captures/ab_v51_*.jsonl", "captures/ab2_v51_*.jsonl",
                            "captures/ab3_v51_*.jsonl", "captures/ab3_v53_*.jsonl"]
    run(pats)
