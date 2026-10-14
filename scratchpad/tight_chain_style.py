"""HOW does the human clear a tight chain (h2->h3 gap 1-2) when the policy dies 24% of the time?

Hypothesis: at gap 1-2 the two obstacles fit inside ONE jump arc (span ~2-3 cells at speed), so the
human takes ONE jump covering both, while the policy jumps h2 and lands on h3. Measured as jump
ONSETS between h2 and h3 -- a game-state edge, so it is symmetric across human (no action byte) and
policy captures.
"""
import sys, glob
sys.path.insert(0, "scratchpad")
import numpy as np
from chain3_forensic import cell_map, chains
from hole_landing_forensic import load_eps, cell_seq

def scan(pats, label, gap=4):
    onset_n, cont_air, n = [], [], 0
    for pat in pats:
        for f in glob.glob(pat):
            for _k, ep in load_eps(f):
                cells = cell_seq(ep); cmap = cell_map(ep, cells)
                if not cmap: continue
                jon = [i for i in range(1, len(ep))
                       if ep[i].get("is_jumping") and not ep[i-1].get("is_jumping")]
                seen = set()
                for (a, b, c) in chains(cmap, gap):
                    if a[0] in seen: continue
                    seen.add(a[0])
                    if c[0] - b[1] not in (1, 2): continue      # TIGHT only
                    i2 = [i for i, cc in enumerate(cells) if cc == b[0]]
                    i3 = [i for i, cc in enumerate(cells) if cc == c[1]]
                    if not i2 or not i3: continue
                    n += 1
                    lo, hi = i2[0], i3[-1]
                    onset_n.append(sum(1 for j in jon if lo <= j <= hi))
                    seg = ep[lo:hi+1]
                    cont_air.append(1 if seg and all(x.get("is_jumping") for x in seg) else 0)
    if not n: print("%-22s none" % label); return
    print("%-22s tight chains %3d | jump onsets between h2..h3: mean %.2f  (0 onsets: %.0f%%, >=2: %.0f%%) | airborne THROUGHOUT h2..h3: %.0f%%"
          % (label, n, np.mean(onset_n), 100*np.mean([x == 0 for x in onset_n]),
             100*np.mean([x >= 2 for x in onset_n]), 100*np.mean(cont_air)))

scan(["captures/w17_v1*_device.jsonl"], "DEVICE v107+v111")
scan(["captures/human_2026*.jsonl"], "DEVICE human")
scan(["scratchpad/_simpat_v111chain.jsonl"], "SIM v111")
