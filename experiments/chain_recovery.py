"""WHY does the policy die in chains when the human, equally blind, never does?

Both see 0% of third obstacles inside the +3 horizon at commit, so it is NOT perception. This
measures RECOVERY, reusing chain3_forensic's VALIDATED chain/structure definition (55 v111 / 45
human) rather than a looser one -- mixing two chain definitions is how you get a number that means
nothing.

⚠️ ENCODING-INDEPENDENT BY CONSTRUCTION: human captures have `action = None` on all 78,066 frames
(they play by touch; there is no action byte), so an action-based reaction time is measurable on the
policy only. Both sides DO stream `is_jumping`/`is_sliding`, so this reads the OBSERVABLE CONSEQUENCE
of an input -- symmetric, and what actually matters for survival anyway.
"""
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import sys, glob
sys.path.insert(0, "experiments")
import numpy as np
from chain3_forensic import cell_map, chains
from hole_landing_forensic import load_eps, cell_seq

HZ = None   # use chain3's default hazard rule via cell_map


def onsets(ep, key):
    return [i for i in range(1, len(ep)) if ep[i].get(key) and not ep[i - 1].get(key)]


def analyse(pats, label, gap=4):
    air, react, n_ch = [], [], 0
    for pat in pats:
        for f in glob.glob(pat):
            for _k, ep in load_eps(f):
                cells = cell_seq(ep)
                cmap = cell_map(ep, cells)
                if not cmap:
                    continue
                jump_on = set(onsets(ep, "is_jumping"))
                slide_on = set(onsets(ep, "is_sliding"))
                act_on = sorted(jump_on | slide_on)
                for (a, b, c) in chains(cmap, gap):
                    n_ch += 1
                    for target in (b[0], c[0]):
                        idx = [i for i, cc in enumerate(cells) if cc == target]
                        if idx:
                            air.append(1 if ep[idx[0]].get("is_jumping") else 0)
                    # touchdown after h1 -> next jump/slide ONSET (frames @10 Hz)
                    ia = [i for i, cc in enumerate(cells) if cc == a[0]]
                    if not ia:
                        continue
                    i0 = ia[0]
                    land = next((j for j in range(i0, min(i0 + 30, len(ep)))
                                 if not ep[j].get("is_jumping")), None)
                    if land is None:
                        continue
                    nxt = next((j for j in act_on if j > land), None)
                    if nxt is not None and nxt - land <= 30:
                        react.append(nxt - land)
    if not air:
        print("%-8s no chains" % label)
        return
    print("%-8s chains %3d | AIRBORNE at h2/h3 %5.1f%% (n=%4d) | touchdown->next input: median %4.1f "
          "frames  p25 %.1f p75 %.1f (n=%d)"
          % (label, n_ch, 100 * np.mean(air), len(air),
             np.median(react) if react else -1,
             np.percentile(react, 25) if react else -1,
             np.percentile(react, 75) if react else -1, len(react)))


analyse(["captures/w17_v111_device.jsonl"], "v111")
analyse(["captures/w17_v107_device.jsonl"], "v107")
analyse(["captures/human_2026*.jsonl"], "HUMAN")
