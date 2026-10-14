#!/usr/bin/env python3
"""LEADING INDICATOR for the step-down fix: P(jump | step-DOWN to low ground) measured IN-SIM.

The device-side detector (scripts/analysis/stepdown_forensic.py) needs phone time and 8-episode batches
have ~11% power. This rolls a policy in the CPU sim and reports the SAME per-encounter rate, which
counts hundreds of crossings per rollout and therefore separates cleanly (the project's standing
"prefer rate metrics over medians" rule).

READ IT LIKE THIS. Every policy up to v102 sits at P(jump)=100% on device AND in-sim, because the
sim kills a grounded step-down. With stepdown_grounded_ok on, the crossing becomes free on foot, so
a policy that has actually learnt the choice should fall toward the HUMAN's 29.4% on a 2->1 drop.
If it stays pinned at 100% the world change alone was too weak a signal and the next lever is the
reaction_bonus, which pays 0.1 for a JUMP on any must_jump cell -- including this soft one.

    python experiments/stepdown_sim.py runs/ppo_v103_stepdown_final.zip \
        --config "$(cat experiments/_v103_cfg.json)" --episodes 24
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse, json, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--config", default="")
    p.add_argument("--land-span", default="")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    import templerun_rl.obs as O
    if a.land_span:
        A, B = (float(x) for x in a.land_span.split(","))
        O.LAND_SPAN_A, O.LAND_SPAN_B = A, B

    from hole_landing_sim import rollout, DEVICE_TRUE_CFG
    cfg = dict(DEVICE_TRUE_CFG)
    if a.config:
        cfg.update(json.loads(a.config))
    rows = rollout(a.model, a.episodes, cfg)

    tag = a.tag or Path(a.model).stem
    out = _ROOT / "experiments" / f"_stepdown_{tag}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    from stepdown_forensic import crossings, report
    r, km = crossings(str(out), 10.0)
    report(r, km, f"SIM {tag}")
    # the one number to watch, isolated
    low = [x for x in r if x["delta"] < 0 and x["after"] <= 1]
    if low:
        j = sum(x["jumped"] for x in low)
        d = sum(x["killed"] for x in low)
        print(f"\n  >>> P(jump | step-DOWN to LOW ground) = {100*j/len(low):.1f}%  "
              f"(n={len(low)}, deaths {d})     HUMAN = 29.4%, every policy so far = 100%")


if __name__ == "__main__":
    main()
