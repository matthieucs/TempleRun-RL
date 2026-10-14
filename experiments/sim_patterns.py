#!/usr/bin/env python3
"""Roll a policy in the CPU sim and score it with scripts/analysis/detect_patterns.py — the SIM-side mirror of
the named-pattern scan, so a training arm can be judged on the USER'S reported mistakes mid-flight
instead of waiting for phone time.

--land-span pins obs.LAND_SPAN_A/B for the rollout. Needed because the constants were corrected on
2026-07-27 (0.0204/-0.94 -> 0.0143/-0.11): a model trained before that saw the OLD landing_hazard, so
scoring it under the new one would measure an obs mismatch rather than the policy.

    python experiments/sim_patterns.py runs/ppo_v91a_heldjump_4001792_steps.zip \
        --config "$(cat experiments/_v91a_cfg.json)" --land-span 0.0204,-0.94 --episodes 16
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse, json, subprocess, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--config", default="")
    p.add_argument("--land-span", default="", help="A,B to pin obs.LAND_SPAN_A/B for this rollout")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    import templerun_rl.obs as O
    if a.land_span:
        A, B = (float(x) for x in a.land_span.split(","))
        O.LAND_SPAN_A, O.LAND_SPAN_B = A, B
        print(f"[land-span pinned] {A}, {B}")

    from hole_landing_sim import rollout, DEVICE_TRUE_CFG
    cfg = dict(DEVICE_TRUE_CFG)
    if a.config:
        cfg.update(json.loads(a.config))
    rows = rollout(a.model, a.episodes, cfg)
    tag = a.tag or Path(a.model).stem
    out = _ROOT / "experiments" / f"_simpat_{tag}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    subprocess.run([sys.executable, str(_ROOT / "scripts" / "analysis" / "detect_patterns.py"), str(out)])


if __name__ == "__main__":
    main()
