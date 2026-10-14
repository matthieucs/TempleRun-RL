"""COAST, corrected: NO new input of EITHER kind (jump or slide) between h2 and h3.

The first version counted only JUMP onsets, so a correct SLIDE response at a slide-type h3 was
scored as coasting. That is not symmetric: sim tight pairs are 51% jump->slide vs the device's 22%,
so the contamination hit the sim far harder and manufactured a sim-vs-device gap.
"""
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import sys, glob
sys.path.insert(0, "experiments")
from collections import Counter
from chain3_forensic import cell_map, chains
from hole_landing_forensic import load_eps, cell_seq

def scan(pats, label, gap=4):
    tab = {True: [0, 0], False: [0, 0]}; comp = Counter(); fatal = Counter()
    g2t, g2d = Counter(), Counter(); tot = 0; km = 0.0
    for pat in pats:
        for f in glob.glob(pat):
            for _k, ep in load_eps(f):
                cells = cell_seq(ep); cmap = cell_map(ep, cells)
                if not cmap: continue
                km += max(float(x.get("distance") or 0) for x in ep) / 1000.0
                dc = cells[-1] if ep[-1].get("is_dead") else None
                onset = [j for j in range(1, len(ep))
                         if (ep[j].get("is_jumping") and not ep[j-1].get("is_jumping"))
                         or (ep[j].get("is_sliding") and not ep[j-1].get("is_sliding"))]
                seen = set()
                for (a, b, c) in chains(cmap, gap):
                    if a[0] in seen: continue
                    seen.add(a[0]); tot += 1
                    g2 = c[0] - b[1]; died = dc is not None and a[0] <= dc <= c[1]
                    g2t[g2] += 1; g2d[g2] += int(died)
                    if g2 not in (1, 2): continue
                    i2 = [i for i, cc in enumerate(cells) if cc == b[0]]
                    i3 = [i for i, cc in enumerate(cells) if cc == c[1]]
                    if not i2 or not i3: continue
                    coast = not any(i2[0] < j <= i3[-1] for j in onset)
                    tab[coast][0] += 1; tab[coast][1] += int(died)
                    k = {"J": "jump", "S": "slide"}
                    key = "%s->%s" % (k.get(b[3], "?"), k.get(c[3], "?"))
                    comp[key] += 1; fatal[key] += int(died)
    tight = sum(tab[k][0] for k in tab); td = sum(tab[k][1] for k in tab)
    print("\n%s  %.0f km | chains %d (%.2f/km) | TIGHT %d (%.0f%%) deaths %d = %.1f%%"
          % (label, km, tot, tot/max(1e-9, km), tight, 100*tight/max(1, tot), td, 100*td/max(1, tight)))
    for k, lab in ((True, "  COAST (no input)"), (False, "  acted           ")):
        n, d = tab[k]; print("%s chains %3d deaths %2d = %5.1f%%" % (lab, n, d, 100*d/max(1, n)))
    print("   gaps:", "  ".join("g%d %d/%d=%.0f%%" % (g, g2d[g], g2t[g], 100*g2d[g]/max(1, g2t[g]))
                                for g in sorted(g2t) if g2t[g]))
    print("   comp:", "  ".join("%s %d/%d" % (k, fatal[k], v) for k, v in comp.most_common()))

scan(["captures/w17_v1*_device.jsonl"], "DEVICE v107+v111")
scan(["captures/human_2026*.jsonl"], "DEVICE human")
scan(["experiments/_simchain_v111.jsonl"], "SIM v111 (C++ 192ep)")
