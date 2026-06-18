#!/usr/bin/env python3
"""Sim-fidelity harness — the measuring stick that proves the sim matches the real game.

This is the deterministic REPLAY test (per docs/SIM_CONTRACT.md §3). It reads a trace
JSONL recorded on a real phone (§1), and for each episode:

  * builds the sim fed with the recorded track:  gym.make('TempleRunSim-v0',
    track_source=<episode.track>)
  * forces the sim to the episode's first real state:  set_state(first.state, first.hidden)
  * replays the recorded action sequence through step(), one action per recorded frame
  * collects the sim's raw_state() at each step

It then diffs sim-vs-real for every canonical field, grouped into three tiers of
strictness, and prints a single scannable REPORT for a human (you do NOT need to read
this code to read the report):

  EXACT      must match every step:
             must_jump, must_slide, is_turn, can_turn_left, can_turn_right,
             path_type, path_level, facing
  TOLERANCE  velocity (+/-2 or +/-2%),  score (per-step delta: sign + ~magnitude)
  +/-1 STEP  is_dead / is_game_over  (death fires within 1 step of the real death)

Exit code is NON-ZERO if any EXACT-tier field is below 100% match, so this doubles as a
CI / unit test for the sim.

Usage:
    python scripts/sim_fidelity.py <trace.jsonl> [--max-divergences 15] [--max-episodes N]
    python scripts/sim_fidelity.py --self-test     # run the built-in synthetic check

The sim (TempleRunSim-v0) is built in parallel against the same contract. If it cannot
be imported yet, this script says so clearly and exits 3 (env not available) rather than
crashing — except --self-test, which uses a hand-made fake sim and always runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

# templerun_rl is installed editable (see pyproject); these always import.
from templerun_rl import constants as C

# Fields grouped by fidelity tier (names from constants.STATE_FIELDS).
EXACT_FIELDS = (
    "must_jump", "must_slide", "is_turn",
    "can_turn_left", "can_turn_right",
    "path_type", "path_level", "facing",
)
TOL_VELOCITY = "velocity"
TOL_SCORE = "score"
STEP1_FIELDS = ("is_dead", "is_game_over")

VEL_ABS_TOL = 2.0      # velocity considered a match within +/-2 ...
VEL_REL_TOL = 0.02     # ... or +/-2% of the real value, whichever is looser
SCORE_REL_TOL = 0.25   # per-step score delta magnitude: within 25% (or +/-2 absolute)
SCORE_ABS_TOL = 2.0


# --------------------------------------------------------------------------- #
# Trace loading
# --------------------------------------------------------------------------- #
@dataclass
class Episode:
    """One recorded episode: ordered steps + (optional) the track fed to the sim."""

    seq: Optional[int] = None
    track: Optional[list] = None         # the FULL traversed track (assembled from per-step windows)
    track_windows: list = field(default_factory=list)  # raw per-step 'track ahead' windows
    actions: list = field(default_factory=list)   # action dispatched at each step (int | None)
    states: list = field(default_factory=list)    # real 22-field state dicts, one per step
    hiddens: list = field(default_factory=list)   # hidden replay info, one per step (may be {})
    end_reason: Optional[str] = None

    @property
    def n_steps(self) -> int:
        return len(self.states)


def _ekey(e: dict) -> tuple:
    return (e.get("type"), e.get("level"), e.get("obstacles"),
            bool(e.get("is_turn")), bool(e.get("can_turn_left")), bool(e.get("can_turn_right")))


def assemble_full_track(windows: list) -> Optional[list]:
    """Reconstruct the full ordered track the player traversed from the per-step 'track ahead'
    windows. Each window is [current, +1, +2, ...]; between steps it shifts left by the number of
    cells crossed. Detect the shift and append the newly-entered tail elements. Robust to identical
    adjacent cells (aligns by position, not value)."""
    ws = [w for w in windows if w]
    if not ws:
        return None
    full = [dict(e) for e in ws[0]]
    prev = [_ekey(e) for e in ws[0]]
    for w in ws[1:]:
        cur = [_ekey(e) for e in w]
        K = len(prev)
        shift = K  # default: no overlap -> treat whole window as new (discontinuity)
        for s in range(0, K + 1):
            overlap = K - s
            if overlap <= 0:
                shift = s
                break
            if prev[s:] == cur[:overlap]:
                shift = s
                break
        if shift > 0:                       # `shift` cells crossed -> append the new tail elements
            for e in w[max(0, len(w) - shift):]:
                full.append(dict(e))
        prev = cur
    return full


def _finalize(ep: Episode) -> Episode:
    if ep.track is None:                    # leave a directly-set track (self-test) intact
        ep.track = assemble_full_track(ep.track_windows)
    return ep


def load_trace(path: str) -> list[Episode]:
    """Parse a trace JSONL into a list of Episodes. Tolerant of missing/null fields.

    Lines: {"t":"ep_start"}, {"t":"step", state/hidden/track/action}, {"t":"ep_end"}.
    A stray 'step' before any 'ep_start' opens an implicit episode (loggers vary).
    """
    episodes: list[Episode] = []
    cur: Optional[Episode] = None
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                print(f"  war: skipping malformed JSON on line {lineno}", file=sys.stderr)
                continue
            t = obj.get("t")
            if t == "ep_start":
                if cur is not None and cur.n_steps:
                    episodes.append(_finalize(cur))
                cur = Episode(seq=obj.get("seq"))
            elif t == "step":
                if cur is None:
                    cur = Episode(seq=obj.get("seq"))
                state = obj.get("state") or {}
                # normalize: fill any missing canonical field from STATE_DEFAULTS so the
                # diff never KeyErrors on a truncated line.
                norm = {f: state.get(f, C.STATE_DEFAULTS.get(f)) for f in C.STATE_FIELDS}
                cur.states.append(norm)
                cur.hiddens.append(obj.get("hidden") or {})
                cur.actions.append(obj.get("action"))
                # collect every per-step 'track ahead' window; the FULL traversed track is
                # reconstructed from them at finalize (the first window alone is only ~8 cells).
                cur.track_windows.append(obj.get("track"))
            elif t == "ep_end":
                if cur is not None:
                    cur.end_reason = obj.get("reason")
                    if cur.n_steps:
                        episodes.append(_finalize(cur))
                    cur = None
            # silently ignore unknown line types
    if cur is not None and cur.n_steps:
        episodes.append(_finalize(cur))
    return episodes


# --------------------------------------------------------------------------- #
# Sim factory (lazy: the sim is built in parallel; tolerate it being absent)
# --------------------------------------------------------------------------- #
class SimUnavailable(RuntimeError):
    pass


def make_sim_factory() -> Callable[[Optional[list]], Any]:
    """Return a callable build_sim(track) -> sim env, or raise SimUnavailable.

    Looked up lazily so --self-test (which injects its own fake) never needs the real sim.
    """
    try:
        import gymnasium as gym
        import templerun_rl  # noqa: F401  (registers TempleRunSim-v0 on import, if present)
    except Exception as e:  # pragma: no cover - import guard
        raise SimUnavailable(f"could not import gymnasium/templerun_rl: {e}")

    if "TempleRunSim-v0" not in gym.envs.registry:
        raise SimUnavailable(
            "TempleRunSim-v0 is not registered yet (the sim is built in parallel "
            "against docs/SIM_CONTRACT.md §2). Re-run once src/templerun_rl/sim_env.py exists."
        )

    def build_sim(track: Optional[list]):
        source = track if track else "procedural"
        return gym.make("TempleRunSim-v0", track_source=source).unwrapped

    return build_sim


# --------------------------------------------------------------------------- #
# Replay one episode through the sim, collecting sim raw_state per step
# --------------------------------------------------------------------------- #
def replay_episode(ep: Episode, build_sim: Callable[[Optional[list]], Any]) -> list[dict]:
    """Run the episode's recorded actions through the sim; return sim raw_state per step.

    Returned list is aligned 1:1 with ep.states (real). The first sim state is taken right
    after set_state (no action), matching the recorded first frame.
    """
    sim = build_sim(ep.track)
    sim_states: list[dict] = []
    try:
        first_state = ep.states[0]
        first_hidden = ep.hiddens[0] if ep.hiddens else {}
        sim.set_state(first_state, first_hidden)
        sim_states.append(dict(sim.raw_state()))
        # Replay the recorded action for every subsequent recorded frame.
        # ep.actions[i] is the action that PRODUCED ep.states[i] (per §1). So to reproduce
        # frame i (i>=1) we step the sim with ep.actions[i].
        for i in range(1, ep.n_steps):
            action = ep.actions[i]
            if action is None:
                action = int(C.Action.NOOP)
            action = int(action)
            if action >= C.N_ACTIONS:           # reset-only bytes never replay as a move
                action = int(C.Action.NOOP)
            sim.step(action)
            sim_states.append(dict(sim.raw_state()))
    finally:
        close = getattr(sim, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    return sim_states


# --------------------------------------------------------------------------- #
# Diffing
# --------------------------------------------------------------------------- #
@dataclass
class FieldStat:
    matches: int = 0
    total: int = 0

    @property
    def pct(self) -> float:
        return 100.0 * self.matches / self.total if self.total else 100.0


@dataclass
class Divergence:
    episode: int
    step: int
    field: str
    real: Any
    sim: Any


def _vel_match(real: Any, sim: Any) -> bool:
    try:
        r, s = float(real), float(sim)
    except (TypeError, ValueError):
        return real == sim
    return abs(r - s) <= max(VEL_ABS_TOL, VEL_REL_TOL * abs(r))


def _score_delta_match(real_d: float, sim_d: float) -> bool:
    """A per-step score delta matches if its sign agrees AND magnitude is close."""
    # sign agreement (0 counts as "no gain"; both zero is a match)
    if (real_d > 0) != (sim_d > 0):
        return False
    if real_d == 0 and sim_d == 0:
        return True
    return abs(real_d - sim_d) <= max(SCORE_ABS_TOL, SCORE_REL_TOL * abs(real_d))


@dataclass
class Report:
    exact: dict = field(default_factory=lambda: {f: FieldStat() for f in EXACT_FIELDS})
    velocity: FieldStat = field(default_factory=FieldStat)
    score: FieldStat = field(default_factory=FieldStat)
    step1: dict = field(default_factory=lambda: {f: FieldStat() for f in STEP1_FIELDS})
    divergences: list = field(default_factory=list)
    n_episodes: int = 0
    n_steps: int = 0
    episodes_replayed: int = 0


def diff_episode(ep_idx: int, ep: Episode, sim_states: list[dict], rep: Report) -> None:
    """Accumulate per-field stats + divergences for one replayed episode into `rep`."""
    n = min(ep.n_steps, len(sim_states))
    rep.n_steps += n

    # --- EXACT tier: every field must match every step ---
    for step in range(n):
        real, sim = ep.states[step], sim_states[step]
        for f in EXACT_FIELDS:
            rv, sv = real.get(f), sim.get(f)
            st = rep.exact[f]
            st.total += 1
            if rv == sv:
                st.matches += 1
            else:
                rep.divergences.append(Divergence(ep_idx, step, f, rv, sv))

    # --- TOLERANCE tier: velocity ---
    for step in range(n):
        rv, sv = ep.states[step].get(TOL_VELOCITY), sim_states[step].get(TOL_VELOCITY)
        rep.velocity.total += 1
        if _vel_match(rv, sv):
            rep.velocity.matches += 1
        else:
            rep.divergences.append(Divergence(ep_idx, step, "velocity", rv, sv))

    # --- TOLERANCE tier: per-step score DELTA (sign + magnitude) ---
    for step in range(1, n):
        real_d = float(ep.states[step].get("score", 0)) - float(ep.states[step - 1].get("score", 0))
        sim_d = float(sim_states[step].get("score", 0)) - float(sim_states[step - 1].get("score", 0))
        rep.score.total += 1
        if _score_delta_match(real_d, sim_d):
            rep.score.matches += 1
        else:
            rep.divergences.append(
                Divergence(ep_idx, step, "score_delta", f"{real_d:+.0f}", f"{sim_d:+.0f}")
            )

    # --- +/-1 STEP tier: death/game-over fires within 1 step of real ---
    for f in STEP1_FIELDS:
        real_fire = _first_true(ep.states, f, n)
        sim_fire = _first_true(sim_states, f, n)
        st = rep.step1[f]
        st.total += 1
        if real_fire is None and sim_fire is None:
            st.matches += 1                       # neither died: fine
        elif real_fire is None or sim_fire is None:
            rep.divergences.append(Divergence(ep_idx, real_fire if real_fire is not None else -1, f,
                                              real_fire, sim_fire))
        elif abs(real_fire - sim_fire) <= 1:
            st.matches += 1
        else:
            rep.divergences.append(Divergence(ep_idx, real_fire, f, real_fire, sim_fire))


def _first_true(states: list[dict], field_name: str, n: int) -> Optional[int]:
    """Index of the first step at which `field_name` is truthy, or None."""
    for i in range(n):
        if states[i].get(field_name):
            return i
    return None


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
def _bar(pct: float, width: int = 20) -> str:
    fill = int(round(pct / 100.0 * width))
    return "#" * fill + "." * (width - fill)


def render_report(rep: Report, max_divergences: int) -> tuple[str, bool, bool, bool]:
    """Render the human report. Returns (text, exact_pass, tol_pass, step1_pass)."""
    L = []
    L.append("=" * 72)
    L.append("  SIM FIDELITY REPORT  (replay real traces through TempleRunSim-v0)")
    L.append("=" * 72)
    L.append(f"  episodes replayed : {rep.episodes_replayed} / {rep.n_episodes}")
    L.append(f"  total steps diffed: {rep.n_steps}")
    L.append("")

    # ---- EXACT tier ----
    L.append("-" * 72)
    L.append("  TIER 1 / EXACT  (must be 100% every step)")
    L.append("-" * 72)
    L.append(f"  {'field':<18}{'%match':>8}  {'matched/total':>16}  bar")
    exact_pass = True
    for f in EXACT_FIELDS:
        st = rep.exact[f]
        ok = st.pct >= 100.0
        exact_pass = exact_pass and ok
        flag = "OK " if ok else "XX "
        L.append(f"  {flag}{f:<15}{st.pct:>7.2f}%  {st.matches:>7}/{st.total:<8} {_bar(st.pct)}")
    L.append("")

    # ---- TOLERANCE tier ----
    L.append("-" * 72)
    L.append("  TIER 2 / TOLERANCE  (velocity +/-2 or 2% ; score delta sign+magnitude)")
    L.append("-" * 72)
    L.append(f"  {'field':<18}{'%match':>8}  {'matched/total':>16}  bar")
    tol_pass = True
    for name, st in (("velocity", rep.velocity), ("score_delta", rep.score)):
        ok = st.pct >= 95.0                      # tolerance tier passes at >=95%
        tol_pass = tol_pass and ok
        flag = "OK " if ok else "XX "
        L.append(f"  {flag}{name:<15}{st.pct:>7.2f}%  {st.matches:>7}/{st.total:<8} {_bar(st.pct)}")
    L.append("")

    # ---- +/-1 STEP tier ----
    L.append("-" * 72)
    L.append("  TIER 3 / +/-1 STEP  (death/game-over fires within 1 step of real)")
    L.append("-" * 72)
    L.append(f"  {'field':<18}{'%ep-ok':>8}  {'ok/episodes':>16}  bar")
    step1_pass = True
    for f in STEP1_FIELDS:
        st = rep.step1[f]
        ok = st.pct >= 100.0
        step1_pass = step1_pass and ok
        flag = "OK " if ok else "XX "
        L.append(f"  {flag}{f:<15}{st.pct:>7.2f}%  {st.matches:>7}/{st.total:<8} {_bar(st.pct)}")
    L.append("")

    # ---- divergences ----
    L.append("-" * 72)
    L.append(f"  FIRST {max_divergences} DIVERGENCES  (total {len(rep.divergences)})")
    L.append("-" * 72)
    if not rep.divergences:
        L.append("  (none — sim reproduced every diffed field within tier tolerance)")
    else:
        L.append(f"  {'ep':>3} {'step':>5}  {'field':<16}{'real':>12} -> {'sim':<12}")
        for d in rep.divergences[:max_divergences]:
            L.append(f"  {d.episode:>3} {d.step:>5}  {d.field:<16}{str(d.real):>12} -> {str(d.sim):<12}")
        if len(rep.divergences) > max_divergences:
            L.append(f"  ... and {len(rep.divergences) - max_divergences} more")
    L.append("")

    # ---- verdict ----
    L.append("=" * 72)
    L.append("  VERDICT")
    L.append("=" * 72)
    L.append(f"  TIER 1  EXACT      : {'PASS' if exact_pass else 'FAIL'}")
    L.append(f"  TIER 2  TOLERANCE  : {'PASS' if tol_pass else 'FAIL'}")
    L.append(f"  TIER 3  +/-1 STEP  : {'PASS' if step1_pass else 'FAIL'}")
    overall = exact_pass and tol_pass and step1_pass
    if overall:
        verdict = "SIM MATCHES REAL GAME — all three tiers pass. Safe to train against the sim."
    elif exact_pass:
        verdict = "Sim track-logic is exact, but physics/scoring/death drift (tier 2/3). Tune the sim."
    else:
        verdict = "Sim DIVERGES on EXACT-tier track logic — do NOT trust it yet. See divergences above."
    L.append(f"  >> {verdict}")
    L.append("=" * 72)
    return "\n".join(L), exact_pass, tol_pass, step1_pass


# --------------------------------------------------------------------------- #
# Top-level run
# --------------------------------------------------------------------------- #
def run_fidelity(
    episodes: Iterable[Episode],
    build_sim: Callable[[Optional[list]], Any],
    max_divergences: int = 15,
    max_episodes: Optional[int] = None,
) -> tuple[str, int]:
    """Replay+diff all episodes, render the report. Returns (report_text, exit_code)."""
    episodes = list(episodes)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    rep = Report(n_episodes=len(episodes))
    for ep_idx, ep in enumerate(episodes):
        if ep.n_steps == 0:
            continue
        try:
            sim_states = replay_episode(ep, build_sim)
        except Exception as e:  # one bad episode shouldn't abort the whole report
            print(f"  war: episode {ep_idx} failed to replay: {e}", file=sys.stderr)
            continue
        rep.episodes_replayed += 1
        diff_episode(ep_idx, ep, sim_states, rep)

    text, exact_pass, _tol, _s1 = render_report(rep, max_divergences)
    # Exit non-zero if ANY exact-tier field < 100% (CI/unit-test behavior, per §3).
    exit_code = 0 if exact_pass else 1
    if rep.episodes_replayed == 0:
        exit_code = max(exit_code, 4)            # nothing diffed -> not a pass
    return text, exit_code


# --------------------------------------------------------------------------- #
# Built-in synthetic self-test (verifies the diff logic with NO real sim/trace)
# --------------------------------------------------------------------------- #
def _fake_state(**over) -> dict:
    s = {f: C.STATE_DEFAULTS.get(f) for f in C.STATE_FIELDS}
    s.update(over)
    return s


class _FakeSim:
    """A 3-step canned sim source for the self-test.

    It ignores set_state/step inputs and just plays back a fixed sequence — one that is
    crafted to be MOSTLY identical to the fake real trace, with a couple of deliberate
    divergences so we can assert the harness flags them.
    """

    def __init__(self, frames: list[dict]):
        self._frames = frames
        self._i = 0

    def set_state(self, raw_state, hidden):
        self._i = 0

    def step(self, action):
        self._i = min(self._i + 1, len(self._frames) - 1)

    def raw_state(self):
        return dict(self._frames[self._i])

    def close(self):
        pass


def self_test() -> int:
    """Hand-made 3-step fake trace + fake sim. Asserts the diff/report logic works."""
    print("[self-test] building a synthetic 3-step trace + fake sim ...")

    # --- the fake REAL trace: 3 steps, run dies at step 2 ---
    real_frames = [
        _fake_state(score=100, velocity=120.0, path_type=3, path_level=0, facing=1,
                    must_jump=False, is_on_ground=True),
        _fake_state(score=130, velocity=125.0, path_type=4, path_level=0, facing=1,
                    must_jump=True, is_jumping=True),
        _fake_state(score=130, velocity=0.0, path_type=4, path_level=0, facing=1,
                    is_dead=True, is_game_over=True),
    ]
    ep = Episode(
        seq=0,
        track=[{"type": 3, "level": 0, "obstacles": 0}],
        actions=[None, int(C.Action.JUMP), int(C.Action.NOOP)],
        states=[dict(f) for f in real_frames],
        hiddens=[{}, {}, {}],
        end_reason="dead",
    )

    # --- the fake SIM playback: identical EXCEPT one EXACT-field flip + velocity within tol ---
    sim_frames = [
        dict(real_frames[0], velocity=121.0),                 # velocity +1 -> within +/-2 (match)
        dict(real_frames[1], path_type=99),                   # path_type 4 -> 99 : EXACT divergence
        dict(real_frames[2]),                                 # dies same step (tier-3 match)
    ]

    def build_sim(_track):
        return _FakeSim([dict(f) for f in sim_frames])

    text, exit_code = run_fidelity([ep], build_sim, max_divergences=15)
    print(text)
    print()

    # --- assertions on the diff logic ---
    # Recompute the stats cleanly (the run_fidelity call above already printed the report).
    rep = Report(n_episodes=1)
    rep.episodes_replayed = 1
    sim_states = replay_episode(ep, build_sim)
    diff_episode(0, ep, sim_states, rep)

    failures = []
    # path_type must show exactly ONE mismatch (step 1: 4 -> 99) and NOT be 100%.
    if rep.exact["path_type"].pct >= 100.0:
        failures.append("path_type should have been flagged as a divergence but read 100%")
    if rep.exact["facing"].pct < 100.0:
        failures.append("facing matched in the fake data but was flagged as diverging")
    # velocity within tolerance -> 100%.
    if rep.velocity.pct < 100.0:
        failures.append(f"velocity within +/-2 should be 100% but is {rep.velocity.pct:.1f}%")
    # death fires same step in both -> tier-3 OK.
    if rep.step1["is_dead"].pct < 100.0:
        failures.append("is_dead fired on the same step but tier-3 flagged it")
    # exactly one EXACT divergence recorded (the path_type flip).
    exact_divs = [d for d in rep.divergences if d.field in EXACT_FIELDS]
    if len(exact_divs) != 1 or exact_divs[0].field != "path_type":
        failures.append(f"expected exactly 1 EXACT divergence on path_type, got {exact_divs}")
    # overall exit code from the harness must be non-zero (EXACT < 100%).
    if exit_code == 0:
        failures.append("exit code should be non-zero because an EXACT field < 100%")

    if failures:
        print("[self-test] FAILED:")
        for f in failures:
            print("   -", f)
        return 1
    print("[self-test] PASSED — diff logic flags EXACT mismatch, honors tolerances, "
          "tracks death within 1 step, and exits non-zero on EXACT failure.")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="path to a trace JSONL recorded on the phone")
    ap.add_argument("--self-test", action="store_true",
                    help="run the built-in synthetic 3-step diff check (no sim/trace needed)")
    ap.add_argument("--max-divergences", type=int, default=15,
                    help="how many of the first divergences to print (default 15)")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="cap how many episodes to replay (default: all)")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.trace:
        ap.error("provide a trace JSONL, or pass --self-test")

    # Load the trace first (fast, no device/sim needed).
    try:
        episodes = load_trace(args.trace)
    except FileNotFoundError:
        print(f"trace file not found: {args.trace}", file=sys.stderr)
        return 2
    if not episodes:
        print(f"no episodes parsed from {args.trace}", file=sys.stderr)
        return 2
    print(f"loaded {len(episodes)} episode(s), "
          f"{sum(e.n_steps for e in episodes)} steps from {args.trace}")

    # Get the sim. If it isn't built yet, say so clearly and stop (don't crash).
    try:
        build_sim = make_sim_factory()
    except SimUnavailable as e:
        print("\nSIM NOT AVAILABLE:", e, file=sys.stderr)
        print("  (The trace loaded fine; run again once TempleRunSim-v0 is registered.)",
              file=sys.stderr)
        print("  Tip: `python scripts/sim_fidelity.py --self-test` verifies the harness "
              "logic without the sim.", file=sys.stderr)
        return 3

    text, exit_code = run_fidelity(
        episodes, build_sim,
        max_divergences=args.max_divergences,
        max_episodes=args.max_episodes,
    )
    print(text)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
