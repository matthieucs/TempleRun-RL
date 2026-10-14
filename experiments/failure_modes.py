"""Classify EVERY device death of a policy into a NAMED failure mode with its mechanism.

Each mode is defined by what was true in the ~10 frames before death, so each one maps to a
distinct fix (recipe knob vs new observation vs neither). Ranked by frequency.
"""
import sys, glob, json, collections
sys.path.insert(0, "src")
from templerun_rl import obs as O

HOLE=set(range(11,17)); CREV=set(range(33,37)); FLAME={31,32}; NARROW=set(range(17,31))
LH = O.OBS_FIELDS.index("landing_hazard")

def classify(fr, di):
    d=fr[di]; pt=int(d.get('path_type') or 0)
    wy,gh=d.get('world_y'),d.get('ground_height')
    below = (wy is not None and gh is not None and float(wy)-float(gh) < -3)
    air=[bool(fr[j].get('is_jumping')) for j in range(max(0,di-9),di+1)]
    acts=[fr[j].get('action') for j in range(max(0,di-9),di+1)]
    dt=int(d.get('death_type') or 0)
    # was it airborne continuously for >=4 frames before dying (an arc that started earlier)?
    long_air = sum(air[-6:])>=4
    if pt in HOLE and below and long_air:            return "A_ARC_ENDS_IN_HOLE"
    if pt in HOLE and below:                          return "B_HOLE_LATE_OR_NO_JUMP"
    if dt in (3,5) or pt in FLAME:
        return "C_AIRBORNE_AT_SLIDE" if any(air[-4:]) else "D_SLIDE_MISSED"
    if pt in CREV:                                    return "E_CREVASSE"
    if pt in NARROW:                                  return "F_NARROW"
    if pt in (3,4,5,6,7,8,9,10) or dt==7:             return "G_TURN"
    if below:                                         return "B_HOLE_LATE_OR_NO_JUMP"
    return "H_OTHER_pt%d_dt%d" % (pt,dt)

def scan(pats,label):
    modes=collections.Counter(); tot=0; seen=set()
    for pat in pats:
        for f in glob.glob(pat):
            rs=[json.loads(l) for l in open(f) if l.strip()]
            eps=collections.defaultdict(list)
            for r in rs:
                if 'velocity' in r: eps[r.get('ep')].append(r)
            for e,fr in eps.items():
                di=next((i for i,x in enumerate(fr) if x.get('death_type')),None)
                if di is None: continue
                key=round(float(fr[di].get('distance') or 0),3)
                if key in seen or key<200: continue      # drop stale duplicates from wedged streams
                seen.add(key); tot+=1
                modes[classify(fr,di)]+=1
    print("\n%s  (%d deaths)" % (label,tot))
    for m,c in modes.most_common():
        print("   %-26s %2d = %3.0f%%" % (m,c,100*c/max(1,tot)))

scan(['captures/w18_v120_device.jsonl','captures/w18_v120_device_b.jsonl','captures/w18_v120_clean.jsonl'],"v120 (CHAMPION)")
scan(['captures/w18_v125_*.jsonl'],"v125")
scan(['captures/w18_v111_base.jsonl'],"v111")
