#!/usr/bin/env python3
"""Rank policies with enough seeds that the answer is actually readable.

    python scripts/validate/eval_paired.py runs/a.zip runs/b.zip --seeds 17 --episodes 128 \
        --sim-config "$(cat experiments/_v114_cfg.json)"

WHY THIS EXISTS. A single n=128 evaluation is worthless for ranking. MEASURED 2026-07-28: the SAME
model (v111) scored 2188 / 2383 / 3222 / 3565 / 3644 across five eval seeds -- a spread of 49 % of
its own mean. So a one-seed comparison of two EQUAL policies can read anywhere from -37 % to +68 %,
and it did: v114 read +29.6 % on seed 0 and -28.9 % on seed 1, while the honest answer over five
seeds was +2.7 % (t = 0.14). Three training arms were called "not resolvable" this week for exactly
this reason, and one nearly got crowned on a phantom.

WHAT TO USE. The statistic is the MEAN OF PER-SEED MEDIANS, with a standard error across seeds.
Measured paired s.d. between two policies is ~1240 m on a ~3000 m mean, so

    seeds needed ~= (1240 / (0.5 * effect_you_care_about * 3000)) ** 2

    effect      seeds   wall clock (~2 min/seed)
      50 %         3        ~6 min
      30 %         8       ~16 min
      20 %        17       ~35 min
      10 %        68      ~2.3 h

Thirty-five minutes to resolve 20 % is cheap next to a 10 M-step arm, and far cheaper than a device
batch (~48 episodes/arm for the same power, at ~2.5 h of phone time each).

WHAT THIS DOES NOT FIX. Track luck is a common random number and could be cancelled outright by
scoring every policy on identical injected tracks, which would need far fewer seeds. That was
attempted and abandoned: `sim_probe_track` in cpp_sim/sim.cpp generates cells fine but its
`distance_real` will not accumulate, so every generated track stays in the early ramp and never
turns slides on. Worth another go -- it would cut the cost roughly tenfold.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = re.compile(r"^(\S+\.zip)\s+median=\s*(\d+)\s+mean=\s*(\d+)")


def run_seed(ckpts: list[str], seed: int, episodes: int, cfg: str, max_steps: int,
             extra: list[str] | None = None) -> dict:
    # eval_cpp.py, NOT eval_ckpt_torch.py: the latter scores in the RETIRED torch sim, which
    # silently lacks anything added to the C++ trainer since (it has no airborne_slide_kill_prob at
    # all). Judge where you train.
    cmd = [sys.executable, str(HERE / "eval_cpp.py"), *ckpts,
           "--episodes", str(episodes), "--seed", str(seed),
           "--max-episode-steps", str(max_steps), *(extra or [])]
    if cfg:
        cmd += ["--sim-config", cfg]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE.parents[1]))
    got = {}
    for ln in out.stdout.splitlines():
        m = LINE.match(ln.strip())
        if m:
            got[m.group(1)] = int(m.group(2))
    if not got:
        print(out.stdout[-500:], out.stderr[-500:])
    return got


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--seeds", type=int, default=17,
                   help="17 resolves ~20%%; see the table in this file's docstring")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--max-episode-steps", type=int, default=8000)
    p.add_argument("--seed-offset", type=int, default=0,
                   help="start the seed range here instead of 0. USE THIS WHENEVER THE CHECKPOINT "
                        "WAS CHOSEN BY LOOKING AT AN EARLIER RUN: re-scoring on the SAME seeds you "
                        "selected on reports a biased-high effect, because part of what you picked "
                        "is that seed set's noise. Fresh seeds give an unbiased estimate.")
    p.add_argument("--sim-config", default="")
    p.add_argument("--curriculum-max-start-dist", type=float, default=0.0,
                   help="must match the arm being judged; forwarded to eval_cpp")
    a = p.parse_args()

    extra = ["--curriculum-max-start-dist", str(a.curriculum_max_start_dist)]
    per: dict[str, list[int]] = {}
    for s in range(a.seed_offset, a.seed_offset + a.seeds):
        got = run_seed(a.checkpoints, s, a.episodes, a.sim_config, a.max_episode_steps, extra)
        for k, v in got.items():
            per.setdefault(k, []).append(v)
        line = "  ".join("%s %5d" % (k.split("_")[1][:8], v) for k, v in got.items())
        print("seed %2d   %s" % (s, line), flush=True)

    names = list(per)
    if not names:
        sys.exit("error: no results parsed")
    print("\n%-46s %8s %8s %8s" % ("checkpoint", "mean", "s.e.", "n"))
    for k in names:
        v = per[k]
        se = st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0
        print("%-46s %8.0f %8.0f %8d" % (k, st.mean(v), se, len(v)))

    # paired comparison against the first checkpoint: pairing cancels the seed's shared difficulty
    if len(names) > 1 and len(per[names[0]]) > 1:
        base = per[names[0]]
        print("\npaired vs %s" % names[0])
        for k in names[1:]:
            v = per[k]
            n = min(len(base), len(v))
            d = [v[i] - base[i] for i in range(n)]
            sd = st.stdev(d) if n > 1 else 0.0
            se = sd / n ** 0.5 if n > 1 else 0.0
            t = st.mean(d) / se if se else 0.0
            verdict = ("RESOLVED" if abs(t) >= 2.0 else
                       "not resolved (need %d seeds for this effect)"
                       % max(n, int((sd / max(1.0, abs(st.mean(d)) / 2.0)) ** 2)))
            print("  %-44s %+7.0f m  (%+5.1f%%)  t=%5.2f  %s"
                  % (k, st.mean(d), 100 * st.mean(d) / st.mean(base), t, verdict))


if __name__ == "__main__":
    main()
