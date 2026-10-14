"""P(death | my arc is predicted to land on a must-JUMP hazard) -- device vs sim, same encoder.

This is the general form of the mechanism the 2026-07-30 forensic found in 7 of 8 fatal frames
(v120's chain_coast rule only covers the tight-chain subset). Computed with the SHIPPED obs encoder
so the number is the policy's own `landing_hazard` column, not a re-derivation.
"""
import sys, glob, json
sys.path.insert(0, "src")
import numpy as np
from templerun_rl import obs as O

IDX = O.OBS_FIELDS.index("landing_hazard")
WINDOW = 15          # frames @10 Hz ~ 1.5 s

def episodes(path):
    """GROUP by ep, do not split on ep CHANGING: cpp_rollout_dump.py writes rows interleaved by env
    (step-major), so a change-detector yields 1-frame 'episodes' and finds ~0 takeoffs."""
    import collections
    g = collections.defaultdict(list)
    for line in open(path):
        try: r = json.loads(line)
        except Exception: continue
        if "velocity" not in r: continue
        g[r.get("ep")].append(r)
    for k in sorted(g, key=lambda x: (x is None, x)):
        if len(g[k]) > 30: yield g[k]

def scan(pats, label):
    tab = {1.0: [0, 0], 0.0: [0, 0], 0.5: [0, 0]}
    for pat in pats:
        for f in glob.glob(pat):
            for fr in episodes(f):
                di = next((i for i, x in enumerate(fr) if x.get("death_type")), None)
                sim = fr[0].get("_hv") is None      # device captures carry _hv; sim dumps do not
                for i in range(1, len(fr)):
                    if fr[i].get("action") != 2: continue
                    p = fr[i] if sim else fr[i-1]       # device: action(i) answers state(i-1)
                    if p.get("is_jumping") or p.get("is_dead"): continue
                    v = float(O.state_to_obs(p)[IDX])
                    if v not in tab: continue
                    died = di is not None and 0 <= di - i <= WINDOW
                    tab[v][0] += 1; tab[v][1] += int(died)
    tot = sum(tab[k][0] for k in tab)
    print("%-30s takeoffs %5d" % (label, tot))
    for v in (0.0, 0.5, 1.0):
        n, d = tab[v]
        print("     landing_hazard=%.1f  %5d takeoffs  %3d fatal<=1.5s = %5.2f%%" % (v, n, d, 100*d/max(1, n)))
    if tab[0.0][0] and tab[0.0][1]:
        print("     LIFT (1.0 vs 0.0) = %.1fx" % ((tab[1.0][1]/max(1,tab[1.0][0]))/(tab[0.0][1]/tab[0.0][0])))

scan(["captures/w18_v111_base.jsonl"], "DEVICE v111 (today)")
scan(["captures/w17_v1*_device.jsonl"], "DEVICE v107+v111 (pooled)")
scan(["scratchpad/_cc_0.0.jsonl"], "SIM v111 (no chain rule)")
scan(["scratchpad/_v120_10002432.jsonl"], "SIM v120 (chain rule 0.5)")
