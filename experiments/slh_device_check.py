"""PRE-CHECK: does the SHIPPED encoder's `slide_landing_hazard` column separate real device deaths?

The derivation measured arc slack directly. This asks the different, harder question: does the column
as IMPLEMENTED -- with the real span law, the level scaling and the integer rounding -- reproduce that
separation on device frames? A cue that is right in principle but rounds to the wrong cell is worth
nothing (that is exactly how landing_hazard was wrong by a whole cell at crevasses until 2026-07-27).

Frame convention: logged action(i) is the policy's response to state(i-1), so the state the policy
SAW when it committed to a takeoff is frame i-1.
"""
import sys, glob, json
sys.path.insert(0, "src")
import numpy as np
from templerun_rl import obs as O

IDX = O.OBS_FIELDS.index("slide_landing_hazard")
SLIDE_DEATHS = {3, 5}          # DeathType Slide / Burnt
WINDOW = 15                    # frames (10 Hz) -- a jump is 8

def episodes(path):
    cur, ep = [], None
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "velocity" not in r:
            continue
        if r.get("ep") != ep:
            if cur:
                yield cur
            cur, ep = [], r.get("ep")
        cur.append(r)
    if cur:
        yield cur

files = [f for f in glob.glob("captures/*.jsonl")
         if "human" not in f and "menuonly" not in f]
tab = {0.0: [0, 0], 0.5: [0, 0], 1.0: [0, 0]}      # value -> [n, slide-deaths]
cp_by = {0.0: [], 1.0: []}
for f in files:
    for fr in episodes(f):
        n = len(fr)
        death_i = next((i for i, x in enumerate(fr) if x.get("death_type")), None)
        dtype = fr[death_i].get("death_type") if death_i is not None else None
        for i in range(1, n):
            if fr[i].get("action") != 2:                 # not a JUMP command
                continue
            prev = fr[i - 1]
            if prev.get("is_jumping") or prev.get("is_dead"):
                continue                                  # only real takeoffs
            v = float(O.state_to_obs(prev)[IDX])
            if v not in tab:
                continue
            died = (death_i is not None and int(dtype or 0) in SLIDE_DEATHS
                    and 0 <= death_i - i <= WINDOW)
            tab[v][0] += 1
            tab[v][1] += int(died)
            if v in cp_by:
                cp_by[v].append(float(prev.get("cell_progress") or 0.0))

print("takeoffs pooled over %d device captures\n" % len(files))
print("%-22s %8s %8s %10s" % ("slide_landing_hazard", "takeoffs", "deaths", "rate"))
for v in (0.0, 0.5, 1.0):
    n, d = tab[v]
    print("%-22s %8d %8d %9.2f%%" % (v, n, d, 100 * d / n if n else 0.0))
n1, d1 = tab[1.0]; n0, d0 = tab[0.0]
if n1 and n0 and d0:
    print("\nlift: P(slide death | cue=1) / P(| cue=0) = %.1fx" % ((d1 / n1) / (d0 / n0)))
for v in (0.0, 1.0):
    if cp_by[v]:
        print("takeoff cp @cue=%s: median %.2f (n=%d)"
              % (v, float(np.median(cp_by[v])), len(cp_by[v])))

# ---- honest ceiling: what share of ALL deaths does the cue flag one jump in advance? ----
from collections import Counter
tot, flagged, mix = 0, 0, Counter()
for f in files:
    for fr in episodes(f):
        di = next((i for i, x in enumerate(fr) if x.get("death_type")), None)
        if di is None:
            continue
        tot += 1
        mix[int(fr[di]["death_type"])] += 1
        for i in range(max(1, di - WINDOW), di + 1):
            if fr[i].get("action") != 2:
                continue
            p = fr[i - 1]
            if p.get("is_jumping") or p.get("is_dead"):
                continue
            if float(O.state_to_obs(p)[IDX]) == 1.0:
                flagged += 1
                break
print("\ndeaths: %d total, %d (%.1f%%) preceded within %d frames by a takeoff the cue FLAGGED"
      % (tot, flagged, 100 * flagged / max(1, tot), WINDOW))
print("death-type mix:", dict(mix.most_common()))
