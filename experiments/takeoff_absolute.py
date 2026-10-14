"""Takeoff distance to the cap in CELLS (absolute), vs the cell_progress metric this project has
used for months. cp resets to ~1.0 whenever the takeoff moves one cell earlier, so a policy that
correctly takes off FURTHER OUT at speed can register as a FALLING cp. Absolute cells-to-cap has no
such wrap and is the physically meaningful quantity."""
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import sys, glob
sys.path.insert(0, "experiments")
import numpy as np
from hole_landing_forensic import load_eps, cell_seq

CREV_PT = set(range(33, 37))
def scan(pats, label):
    rows = []
    for pat in pats:
        for f in glob.glob(pat):
            for _k, ep in load_eps(f):
                cells = cell_seq(ep)
                capcell = {}
                for i, fr in enumerate(ep):
                    pt = int(fr.get("path_type") or 0)
                    if pt in CREV_PT and int(fr.get("path_level") or 0) >= 3:
                        capcell.setdefault(cells[i], i)
                jon = [i for i in range(1, len(ep))
                       if ep[i].get("is_jumping") and not ep[i-1].get("is_jumping")]
                for c, ci in capcell.items():
                    cand = [j for j in jon if cells[j] <= c and c - cells[j] <= 4]
                    if not cand: continue
                    j = cand[-1]
                    fr = ep[j-1]
                    v = float(fr.get("velocity") or 0)
                    cp = float(fr.get("cell_progress") or 0)
                    dist = (c - cells[j-1]) - cp          # cells from takeoff point to the cap
                    rows.append((v, cp, dist))
    if not rows: print("%-22s none" % label); return
    a = np.array(rows)
    print("%-22s n=%3d" % (label, len(a)))
    for lo, hi in ((140,180),(180,200),(200,240)):
        m = (a[:,0] >= lo) & (a[:,0] < hi)
        if m.sum() >= 4:
            print("    v%3d-%3d n=%3d  cp %.2f   cells-to-cap %.2f" % (lo,hi,m.sum(),np.median(a[m,1]),np.median(a[m,2])))
    if len(a) > 8:
        for col, nm in ((1,"cp"),(2,"cells-to-cap")):
            s = np.polyfit(a[:,0], a[:,col], 1)[0]
            print("    slope d(%s)/dv = %+.5f" % (nm, s))

scan(["captures/human_2026*.jsonl"], "HUMAN")
scan(["captures/w18_v111_base.jsonl"], "v111 device (today)")
