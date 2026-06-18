#!/usr/bin/env python3
"""Statistical generation-fidelity test — does the PROCEDURAL sim generator match reality?

The replay test (scripts/sim_fidelity.py) proves the sim's *physics/collision/scoring* on a
KNOWN track. This script tests the other half: the sim's *procedural track generator* (the
RNG that builds tracks during training). It can't be diffed frame-by-frame against reality
(different RNG), so we compare DISTRIBUTIONS instead.

We extract, from BOTH the real episodes' recorded tracks AND many freshly generated
procedural sim episodes, the same four track-shape distributions:

  * obstacle-spacing      gaps (in path elements) between consecutive obstacles
  * turn-rate             fraction of elements that are turns (a single ratio per source)
  * piece-type mix        histogram over path `type` values
  * coin-run length       lengths of consecutive coin (CoinBonus obstacle) runs

For each, we print a side-by-side comparison table and run a two-sample Kolmogorov-Smirnov
test (implemented here — no scipy dependency). PASS if the KS statistic is below a threshold
(distributions indistinguishable at the chosen confidence); FAIL otherwise.

Usage:
    python scripts/sim_gen_fidelity.py <trace.jsonl> [--n-sim-episodes 200] [--seed 0]
    python scripts/sim_gen_fidelity.py --self-test     # synthetic check of the stats

NOTE: this is the STUB called for in docs/SIM_CONTRACT.md §3 — it is fully working against
the contract's sim interface, but the obstacle-encoding constants (COIN_OBSTACLE, TURN
detection) are best-effort from the dump and may need a one-line tweak once the real
generator lands. Marked clearly inline.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from templerun_rl import constants as C  # noqa: F401  (kept for canonical names / future use)

# --- Track encoding constants (best-effort from dump.cs; see CLAUDE.md §5) ----------------
# Obstacles enum: must_jump in {1 hole, 4 treeJump}; must_slide in {5 treeSlide, 6 flameTower}.
# A non-zero `obstacles` value marks an obstacle element. Coins aren't in `obstacles` in the
# trace schema; if the generator exposes a coin marker we treat obstacles==COIN as a coin run.
OBSTACLE_NONE = 0
# CoinBonus isn't a path Obstacle in the dump — coin runs are a generator concept. We detect a
# "coin run" as a maximal run of elements with the same (type) that the generator tags; absent a
# tag we fall back to runs of obstacle-free straight pieces. TWEAK here when the generator lands.
COIN_TYPE_HINT = None  # set to an int path `type` once known; None -> use obstacle-free-run proxy

# Pass thresholds (two-sample KS statistic D; lower = more similar).
KS_PASS_D = 0.20            # D below this -> distributions match well enough
TURN_RATE_ABS_TOL = 0.05    # turn-rate is a scalar -> compare directly


# --------------------------------------------------------------------------- #
# Track extraction from a trace
# --------------------------------------------------------------------------- #
def _iter_trace_tracks(path: str):
    """Yield each episode's flattened track (list of element dicts) from a trace JSONL.

    We stitch per-step `track` windows into one ordered sequence per episode, de-duplicating
    the overlap: each step reports the K elements AHEAD, which slide forward by ~1 per step.
    We approximate the full track by taking the FIRST element of each step's window (the one
    about to be consumed) plus the final window's tail.
    """
    cur_first_elems: list = []
    last_window: list = []
    have_ep = False
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = obj.get("t")
            if t == "ep_start":
                if have_ep and (cur_first_elems or last_window):
                    yield cur_first_elems + last_window[1:]
                cur_first_elems, last_window, have_ep = [], [], True
            elif t == "step":
                have_ep = True
                window = obj.get("track") or []
                if window:
                    cur_first_elems.append(window[0])
                    last_window = window
            elif t == "ep_end":
                if cur_first_elems or last_window:
                    yield cur_first_elems + last_window[1:]
                cur_first_elems, last_window, have_ep = [], [], False
    if cur_first_elems or last_window:
        yield cur_first_elems + last_window[1:]


# --------------------------------------------------------------------------- #
# Distribution feature extraction (works on any list[element-dict])
# --------------------------------------------------------------------------- #
def _is_turn(el: dict) -> bool:
    return bool(el.get("is_turn") or el.get("can_turn_left") or el.get("can_turn_right"))


def _has_obstacle(el: dict) -> bool:
    return int(el.get("obstacles", OBSTACLE_NONE) or 0) != OBSTACLE_NONE


def obstacle_spacing(track: list) -> list:
    """Gaps (in elements) between consecutive obstacle elements."""
    idxs = [i for i, el in enumerate(track) if _has_obstacle(el)]
    return [b - a for a, b in zip(idxs, idxs[1:])]


def turn_rate(track: list) -> float:
    if not track:
        return 0.0
    return sum(1 for el in track if _is_turn(el)) / len(track)


def piece_type_counts(track: list) -> Counter:
    return Counter(int(el.get("type", -1)) for el in track)


def coin_run_lengths(track: list) -> list:
    """Lengths of consecutive coin runs.

    If COIN_TYPE_HINT is set, a coin element is `type == COIN_TYPE_HINT`. Otherwise we use the
    obstacle-free-straight proxy: maximal runs of non-turn, obstacle-free elements (where coins
    naturally sit). TWEAK COIN_TYPE_HINT once the generator exposes the real marker.
    """
    runs, cur = [], 0
    for el in track:
        if COIN_TYPE_HINT is not None:
            is_coin = int(el.get("type", -1)) == COIN_TYPE_HINT
        else:
            is_coin = (not _is_turn(el)) and (not _has_obstacle(el))
        if is_coin:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


@dataclass
class DistBundle:
    """All four distributions aggregated across many tracks from one source (real or sim)."""

    spacing: list = field(default_factory=list)
    turn_rates: list = field(default_factory=list)
    type_counts: Counter = field(default_factory=Counter)
    coin_runs: list = field(default_factory=list)
    n_tracks: int = 0
    n_elements: int = 0

    def add(self, track: list) -> None:
        if not track:
            return
        self.n_tracks += 1
        self.n_elements += len(track)
        self.spacing += obstacle_spacing(track)
        self.turn_rates.append(turn_rate(track))
        self.type_counts += piece_type_counts(track)
        self.coin_runs += coin_run_lengths(track)


# --------------------------------------------------------------------------- #
# Two-sample Kolmogorov-Smirnov (no scipy)
# --------------------------------------------------------------------------- #
def ks_2samp(a: list, b: list) -> tuple[float, float]:
    """Two-sample KS statistic D and an asymptotic p-value. Returns (D, p).

    D = max |F_a(x) - F_b(x)| over the pooled samples. p uses the Kolmogorov asymptotic
    series (good for n,m >= ~20). Empty inputs -> D=1.0 (maximally different), p=0.
    """
    a = sorted(float(x) for x in a)
    b = sorted(float(x) for x in b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 1.0, 0.0
    # Walk the merged order. At each distinct pooled value x, consume ALL of x from BOTH
    # samples before measuring the CDF gap — otherwise a tie that exists in both samples
    # transiently desyncs the two empirical CDFs and inflates D (a real bug this fixes).
    i = j = 0
    d = 0.0
    while i < n or j < m:
        if j >= m or (i < n and a[i] <= b[j]):
            x = a[i]
        else:
            x = b[j]
        while i < n and a[i] == x:
            i += 1
        while j < m and b[j] == x:
            j += 1
        fa, fb = i / n, j / m
        d = max(d, abs(fa - fb))
    en = (n * m / (n + m)) ** 0.5
    p = _kolmogorov_p((en + 0.12 + 0.11 / en) * d)
    return d, p


def _kolmogorov_p(t: float) -> float:
    """Asymptotic Q_KS(t) = P(D > t/en). Series sum; clamps to [0,1]."""
    if t < 1e-9:
        return 1.0
    s = 0.0
    for k in range(1, 101):
        term = (-1) ** (k - 1) * 2.718281828459045 ** (-2.0 * k * k * t * t)
        s += term
        if abs(term) < 1e-10:
            break
    p = 2.0 * s
    return max(0.0, min(1.0, p))


# --------------------------------------------------------------------------- #
# Sim sampling (procedural episodes)
# --------------------------------------------------------------------------- #
class SimUnavailable(RuntimeError):
    pass


def sample_sim_tracks(n_episodes: int, seed: int, max_steps: int = 2000) -> list:
    """Roll out n procedural sim episodes; return each episode's observed track sequence.

    We reconstruct the track from the per-step raw_state (path_type/path_level/obstacles +
    turn cues) since the sim doesn't necessarily expose its internal track list. One element
    is appended whenever the element ahead changes (path_type/level transition).
    """
    try:
        import gymnasium as gym
        import templerun_rl  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SimUnavailable(f"could not import gymnasium/templerun_rl: {e}")
    if "TempleRunSim-v0" not in gym.envs.registry:
        raise SimUnavailable(
            "TempleRunSim-v0 is not registered yet (the procedural sim is built in parallel "
            "against docs/SIM_CONTRACT.md §2). Re-run once it exists."
        )

    tracks = []
    for ep in range(n_episodes):
        env = gym.make("TempleRunSim-v0", track_source="procedural").unwrapped
        try:
            env.reset(seed=seed + ep)
            track, prev_key = [], None
            done = False
            steps = 0
            while not done and steps < max_steps:
                st = env.raw_state()
                key = (st.get("path_type"), st.get("path_level"),
                       st.get("must_jump"), st.get("must_slide"))
                if key != prev_key:
                    track.append({
                        "type": st.get("path_type", -1),
                        "level": st.get("path_level", -1),
                        # encode must_jump/slide back to an obstacle marker (non-zero => obstacle)
                        "obstacles": 1 if (st.get("must_jump") or st.get("must_slide")) else 0,
                        "is_turn": bool(st.get("is_turn")),
                        "can_turn_left": bool(st.get("can_turn_left")),
                        "can_turn_right": bool(st.get("can_turn_right")),
                    })
                    prev_key = key
                _o, _r, term, trunc, _i = env.step(int(C.Action.NOOP))
                done = term or trunc
                steps += 1
            if track:
                tracks.append(track)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    return tracks


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _summ(vals: list) -> str:
    if not vals:
        return "n=0"
    vals = sorted(float(v) for v in vals)
    n = len(vals)
    mean = sum(vals) / n
    median = vals[n // 2]
    return f"n={n:<5} mean={mean:6.2f} med={median:6.2f} min={vals[0]:5.1f} max={vals[-1]:6.1f}"


def render_gen_report(real: DistBundle, sim: DistBundle) -> tuple[str, bool]:
    L = []
    L.append("=" * 74)
    L.append("  GENERATION FIDELITY REPORT  (real tracks vs procedural sim tracks)")
    L.append("=" * 74)
    L.append(f"  real: {real.n_tracks} tracks / {real.n_elements} elements   "
             f"sim: {sim.n_tracks} tracks / {sim.n_elements} elements")
    L.append("")

    results = []  # (name, passed, detail)

    # 1) obstacle spacing — KS
    d, p = ks_2samp(real.spacing, sim.spacing)
    ok = d <= KS_PASS_D and real.spacing and sim.spacing
    results.append(("obstacle-spacing", bool(ok), f"KS D={d:.3f} p={p:.3f}"))
    L.append("  [1] OBSTACLE SPACING (gaps between obstacles, in elements)")
    L.append(f"      real : {_summ(real.spacing)}")
    L.append(f"      sim  : {_summ(sim.spacing)}")
    L.append(f"      KS   : D={d:.3f}  p={p:.3f}   {'PASS' if ok else 'FAIL'} (D<= {KS_PASS_D})")
    L.append("")

    # 2) turn-rate — scalar compare (mean of per-track rates)
    rr = sum(real.turn_rates) / len(real.turn_rates) if real.turn_rates else 0.0
    sr = sum(sim.turn_rates) / len(sim.turn_rates) if sim.turn_rates else 0.0
    ok = abs(rr - sr) <= TURN_RATE_ABS_TOL and real.turn_rates and sim.turn_rates
    results.append(("turn-rate", bool(ok), f"|{rr:.3f}-{sr:.3f}|"))
    L.append("  [2] TURN RATE (fraction of elements that are turns)")
    L.append(f"      real : {rr:.4f}     sim : {sr:.4f}     |diff| = {abs(rr - sr):.4f}")
    L.append(f"      {'PASS' if ok else 'FAIL'} (tol +/-{TURN_RATE_ABS_TOL})")
    L.append("")

    # 3) piece-type mix — KS on the per-element type samples + a small table
    real_types = _counter_to_samples(real.type_counts)
    sim_types = _counter_to_samples(sim.type_counts)
    d3, p3 = ks_2samp(real_types, sim_types)
    ok = d3 <= KS_PASS_D and real_types and sim_types
    results.append(("piece-type-mix", bool(ok), f"KS D={d3:.3f} p={p3:.3f}"))
    L.append("  [3] PIECE-TYPE MIX (share of each path `type`)")
    L.append(f"      {'type':>5}  {'real%':>8}  {'sim%':>8}")
    all_types = sorted(set(real.type_counts) | set(sim.type_counts))
    rt_tot = sum(real.type_counts.values()) or 1
    st_tot = sum(sim.type_counts.values()) or 1
    for ty in all_types[:15]:
        rp = 100.0 * real.type_counts.get(ty, 0) / rt_tot
        sp = 100.0 * sim.type_counts.get(ty, 0) / st_tot
        L.append(f"      {ty:>5}  {rp:>7.2f}%  {sp:>7.2f}%")
    if len(all_types) > 15:
        L.append(f"      ... and {len(all_types) - 15} more types")
    L.append(f"      KS   : D={d3:.3f}  p={p3:.3f}   {'PASS' if ok else 'FAIL'} (D<= {KS_PASS_D})")
    L.append("")

    # 4) coin-run length — KS
    d4, p4 = ks_2samp(real.coin_runs, sim.coin_runs)
    ok = d4 <= KS_PASS_D and real.coin_runs and sim.coin_runs
    results.append(("coin-run-length", bool(ok), f"KS D={d4:.3f} p={p4:.3f}"))
    L.append("  [4] COIN-RUN LENGTH (consecutive coin/clear-straight elements)")
    L.append(f"      real : {_summ(real.coin_runs)}")
    L.append(f"      sim  : {_summ(sim.coin_runs)}")
    L.append(f"      KS   : D={d4:.3f}  p={p4:.3f}   {'PASS' if ok else 'FAIL'} (D<= {KS_PASS_D})")
    L.append("")

    # verdict
    L.append("=" * 74)
    L.append("  VERDICT (per distribution)")
    L.append("=" * 74)
    all_pass = True
    for name, passed, detail in results:
        all_pass = all_pass and passed
        L.append(f"  {'PASS' if passed else 'FAIL'}  {name:<20} {detail}")
    L.append("-" * 74)
    if all_pass:
        L.append("  >> GENERATOR MATCHES: all track-shape distributions are statistically close.")
    else:
        L.append("  >> GENERATOR DRIFTS: one or more distributions differ — tune the procedural "
                 "generator before relying on procedural training.")
    L.append("=" * 74)
    return "\n".join(L), all_pass


def _counter_to_samples(counter: Counter) -> list:
    """Expand a type->count histogram into a flat sample list (for KS on the type variable)."""
    out = []
    for ty, cnt in counter.items():
        out.extend([ty] * cnt)
    return out


# --------------------------------------------------------------------------- #
# Synthetic self-test
# --------------------------------------------------------------------------- #
def self_test() -> int:
    """Verify the stats/KS logic on hand-made data (no sim/trace needed)."""
    print("[self-test] checking KS + distribution extraction on synthetic tracks ...")
    failures = []

    # KS on identical samples -> D ~ 0.
    d, _p = ks_2samp([1, 2, 3, 4, 5] * 10, [1, 2, 3, 4, 5] * 10)
    if d > 1e-9:
        failures.append(f"KS of identical samples should be ~0, got D={d}")
    # KS on disjoint samples -> D == 1.
    d, _p = ks_2samp([0] * 50, [100] * 50)
    if abs(d - 1.0) > 1e-9:
        failures.append(f"KS of disjoint samples should be 1.0, got D={d}")

    # feature extraction on a tiny known track.
    track = [
        {"type": 1, "obstacles": 0, "is_turn": False},
        {"type": 1, "obstacles": 1, "is_turn": False},   # obstacle @ idx1
        {"type": 2, "obstacles": 0, "is_turn": True},    # turn
        {"type": 1, "obstacles": 0, "is_turn": False},
        {"type": 1, "obstacles": 4, "is_turn": False},   # obstacle @ idx4
    ]
    sp = obstacle_spacing(track)
    if sp != [3]:
        failures.append(f"obstacle_spacing wrong: expected [3] got {sp}")
    tr = turn_rate(track)
    if abs(tr - 0.2) > 1e-9:
        failures.append(f"turn_rate wrong: expected 0.2 got {tr}")
    tc = piece_type_counts(track)
    if tc[1] != 4 or tc[2] != 1:   # idx 0,1,3,4 are type 1 (4 of them); idx 2 is type 2
        failures.append(f"piece_type_counts wrong: {dict(tc)}")

    # build two clearly-different bundles and confirm the report FAILS.
    import random
    rng = random.Random(0)
    real, sim = DistBundle(), DistBundle()
    for _ in range(40):
        # real: obstacles every ~5 elements
        rt = [{"type": 1, "obstacles": (1 if i % 5 == 0 else 0), "is_turn": (i % 13 == 0)}
              for i in range(60)]
        real.add(rt)
        # sim: obstacles every ~2 elements (clearly denser) -> should FAIL spacing
        stk = [{"type": 1, "obstacles": (1 if i % 2 == 0 else 0), "is_turn": (i % 13 == 0)}
               for i in range(60)]
        sim.add(stk)
    text, all_pass = render_gen_report(real, sim)
    print(text)
    if all_pass:
        failures.append("clearly-different generators should FAIL but report said PASS")

    # and confirm a MATCHING pair PASSES.
    real2, sim2 = DistBundle(), DistBundle()
    for s in range(40):
        r1 = random.Random(1000 + s)
        r2 = random.Random(5000 + s)
        t1 = [{"type": 1, "obstacles": (1 if r1.random() < 0.2 else 0),
               "is_turn": (r1.random() < 0.08)} for _ in range(60)]
        t2 = [{"type": 1, "obstacles": (1 if r2.random() < 0.2 else 0),
               "is_turn": (r2.random() < 0.08)} for _ in range(60)]
        real2.add(t1)
        sim2.add(t2)
    _t, match_pass = render_gen_report(real2, sim2)
    if not match_pass:
        failures.append("statistically identical generators should PASS but report said FAIL")

    print()
    if failures:
        print("[self-test] FAILED:")
        for f in failures:
            print("   -", f)
        return 1
    print("[self-test] PASSED — KS edge cases, feature extraction, and PASS/FAIL verdicts all correct.")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="path to a real trace JSONL")
    ap.add_argument("--self-test", action="store_true", help="run the synthetic stats check (no sim/trace)")
    ap.add_argument("--n-sim-episodes", type=int, default=200, help="procedural sim episodes to sample")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed for the sim rollouts")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.trace:
        ap.error("provide a real trace JSONL, or pass --self-test")

    # Real distributions from the trace.
    real = DistBundle()
    try:
        n = 0
        for track in _iter_trace_tracks(args.trace):
            real.add(track)
            n += 1
    except FileNotFoundError:
        print(f"trace file not found: {args.trace}", file=sys.stderr)
        return 2
    if real.n_tracks == 0:
        print(f"no tracks extracted from {args.trace} (does the logger fill `track`?)", file=sys.stderr)
        return 2
    print(f"extracted {real.n_tracks} real track(s), {real.n_elements} elements from {args.trace}")

    # Sim distributions from procedural rollouts.
    try:
        sim_tracks = sample_sim_tracks(args.n_sim_episodes, args.seed)
    except SimUnavailable as e:
        print("\nSIM NOT AVAILABLE:", e, file=sys.stderr)
        print("  (Real distributions extracted fine; re-run once TempleRunSim-v0 is registered.)",
              file=sys.stderr)
        print("  Tip: `python scripts/sim_gen_fidelity.py --self-test` checks the stats logic now.",
              file=sys.stderr)
        return 3
    sim = DistBundle()
    for tk in sim_tracks:
        sim.add(tk)
    print(f"sampled {sim.n_tracks} procedural sim track(s), {sim.n_elements} elements")

    text, all_pass = render_gen_report(real, sim)
    print(text)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
