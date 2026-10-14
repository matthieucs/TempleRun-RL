"""Is being AIRBORNE when the 2nd/3rd chain obstacle arrives what actually kills?

Correlation before action: a behaviour that merely DIFFERS from the human is not automatically the
cause of death. This cross-tabs chain outcome against airborne-at-h2/h3 on the same validated chain
definition, pooling the champion-era policy captures for n.
"""
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import sys, glob
sys.path.insert(0, "experiments")
from chain3_forensic import cell_map, chains
from hole_landing_forensic import load_eps, cell_seq

def scan(pats, label, gap=4):
    tab = {True: [0, 0], False: [0, 0]}      # airborne -> [chains, deaths]
    for pat in pats:
        for f in glob.glob(pat):
            for _k, ep in load_eps(f):
                cells = cell_seq(ep)
                cmap = cell_map(ep, cells)
                if not cmap: continue
                death_cell = cells[-1] if ep[-1].get("is_dead") else None
                seen = set()
                for (a, b, c) in chains(cmap, gap):
                    if a[0] in seen: continue
                    seen.add(a[0])
                    air = False
                    for target in (b[0], c[0]):
                        idx = [i for i, cc in enumerate(cells) if cc == target]
                        if idx and ep[idx[0]].get("is_jumping"): air = True
                    died = death_cell is not None and a[0] <= death_cell <= c[1]
                    tab[air][0] += 1; tab[air][1] += int(died)
    print("%-22s %-10s %8s %8s %8s" % (label, "airborne", "chains", "deaths", "rate"))
    for k in (True, False):
        n, d = tab[k]
        print("%-22s %-10s %8d %8d %7.1f%%" % ("", str(k), n, d, 100*d/max(1, n)))

scan(["captures/w17_v111_device.jsonl"], "v111")
scan(["captures/w17_v1*_device.jsonl"], "v107+v111 pooled")
scan(["captures/human_2026*.jsonl"], "HUMAN")
