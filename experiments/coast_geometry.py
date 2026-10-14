"""In a COAST, where does the arc actually come down relative to h3 -- sim vs device?

If the sim's arc clears h3 and lands beyond, one jump legitimately solves the tight pair there and
no death rule fires; on device the same arc ends ON h3. That would make this a GEOMETRY gap, not a
death-model dose. Measured in CELLS from takeoff, on both sides with the same code.
"""
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import sys, glob
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
import numpy as np
from collections import Counter
from templerun_rl import obs as O
from chain3_forensic import cell_map, chains
from hole_landing_forensic import load_eps, cell_seq

def scan(pats, label, gap=4):
    rel = Counter(); spans = []; n = 0; land_on_h3 = 0
    for pat in pats:
        for f in glob.glob(pat):
            for _k, ep in load_eps(f):
                cells = cell_seq(ep); cmap = cell_map(ep, cells)
                if not cmap: continue
                seen = set()
                for (a, b, c) in chains(cmap, gap):
                    if a[0] in seen: continue
                    seen.add(a[0])
                    if c[0] - b[1] not in (1, 2): continue
                    i2 = [i for i, cc in enumerate(cells) if cc == b[0]]
                    i3 = [i for i, cc in enumerate(cells) if cc == c[1]]
                    if not i2 or not i3: continue
                    if sum(1 for j in range(i2[0]+1, i3[-1]+1)
                           if ep[j].get("is_jumping") and not ep[j-1].get("is_jumping")): continue
                    jon = [j for j in range(1, i2[0]+1)
                           if ep[j].get("is_jumping") and not ep[j-1].get("is_jumping")]
                    if not jon: continue
                    t = jon[-1]; fr = ep[t-1]
                    v = float(fr.get("velocity") or 0)
                    span = max(0.0, O.LAND_SPAN_A*v + O.LAND_SPAN_B)
                    land_cell = cells[t-1] + int(float(fr.get("cell_progress") or 0) + span)
                    d = land_cell - c[0]        # <0 short of h3, 0 ON h3, >0 past it
                    rel[max(-2, min(2, d))] += 1
                    spans.append(span); n += 1
                    if c[0] <= land_cell <= c[1]: land_on_h3 += 1
    if not n: print("%-26s none" % label); return
    print("%-26s coasts %3d | span median %.2f cells | lands ON h3 %3d = %2.0f%% | rel-to-h3 %s"
          % (label, n, float(np.median(spans)), land_on_h3, 100*land_on_h3/n,
             dict(sorted(rel.items()))))

scan(["captures/w17_v1*_device.jsonl"], "DEVICE v107+v111")
scan(["experiments/_simchain_v111.jsonl"], "SIM v111 (C++)")
