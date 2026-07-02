#!/usr/bin/env python3
"""TURN + LATERAL behavioral gate — a phone-free guard that catches sim turn/lateral bugs BEFORE a
device cycle. Runs fixed hand-coded policies and asserts the sim rewards the DEVICE-CORRECT behavior
and punishes the device-FATAL behavior.

History (each caught only AFTER a wasted device eval — this gate exists so that stops):
  * a BUFFERED early-turn sim survived 99.8% in-sim yet regressed the device (736 -> 226): the policy
    pre-fired turns on the approach cell. Guard: PRE-FIRE must DIE.
  * an auto-recentering lateral model centered the policy for FREE in-sim; on device the lane HOLDS, so
    the policy turned off-center (lane 4.7) and fell off the path (226 -> 201). Guard: OFF-CENTER-TURN
    must DIE + survivors must turn CENTERED (|lane| small).

Expected on a device-faithful sim:
  * oracle / prompt (turn on is_turn, from CENTER)      -> SURVIVE, and turn at |lane| ~ 0
  * PRE-FIRE (swipe on next_is_turn, before the junction) -> DIE
  * OFF-CENTER (drift off-center, then turn)            -> DIE
  * SPAM (fire turns on straights)                      -> DIE

    python scripts/turn_gate.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import templerun_rl  # noqa: E402
import templerun_rl.constants as C  # noqa: E402
import templerun_rl.obs as O  # noqa: E402
from train_sim import _oracle  # noqa: E402

_spam_ctr = [0]
_oc_ctr = [0]
_prompt_mem = {"cooldown": 0, "prev_dist": 0.0}
_TURN_AIM = 0.28                     # swipe at first frame in the survival window (device: too-early
                                     # OR too-late swipe lands off-center -> water; center ~0.44)


def _base(st):
    if st.get("next_wide_gap"):
        return C.Action.JUMP if float(st.get("cell_progress", 1)) >= 0.62 else C.Action.NOOP
    if st["must_jump"] or st.get("next_must_jump"):
        return C.Action.JUMP
    if st["must_slide"] or st.get("next_must_slide"):
        return C.Action.SLIDE
    return None


def _side(st, allow_next=False):
    cl = st["can_turn_left"] or (allow_next and st.get("next_can_turn_left"))
    cr = st["can_turn_right"] or (allow_next and st.get("next_can_turn_right"))
    return C.Action.TURN_LEFT if cl else (C.Action.TURN_RIGHT if cr else C.Action.TURN_LEFT)


def _recenter(st):
    """ANALOG TILT (2026-07-02): NoOp = level phone = the game itself recenters the lane. A centered
    policy just releases the tilt — no counter-strafe needed (a strafe byte would slam toward ±edge)."""
    return C.Action.NOOP


def prompt(st):                     # device-correct: centered, swipe at CELL CENTER (cprog>=aim), once
    b = _base(st)
    if b is not None:
        return b
    d = float(st.get("distance", 0.0))
    if d < _prompt_mem["prev_dist"] - 20.0:
        _prompt_mem["cooldown"] = 0
    _prompt_mem["prev_dist"] = d
    if _prompt_mem["cooldown"] > 0:
        _prompt_mem["cooldown"] -= 1
    elif st["is_turn"] and float(st.get("cell_progress", 0.0)) >= _TURN_AIM:
        _prompt_mem["cooldown"] = 5
        return _side(st)
    return _recenter(st)


def prefire(st):                    # device-FATAL: pre-fire on next_is_turn (the buffer exploit)
    b = _base(st)
    if b is not None:
        return b
    return _side(st, allow_next=True) if (st["is_turn"] or st.get("next_is_turn")) else _recenter(st)


def off_center(st):                 # device-FATAL: HOLD a side tilt, then turn from the edge
    b = _base(st)
    if b is not None:
        return b
    if st["is_turn"]:
        return _side(st)            # turns from the held edge (|lane| ~7) -> should Fall
    return C.Action.STRAFE_RIGHT    # hold the tilt every tick (analog: NoOp would recenter)


def spam(st):                       # device-FATAL: fire turns on straights
    b = _base(st)
    if b is not None:
        return b
    _spam_ctr[0] += 1
    if st["is_turn"] or st.get("next_is_turn") or _spam_ctr[0] % 3 == 0:
        return _side(st, allow_next=True)
    return C.Action.NOOP


def double_turn(st):                # device-FATAL: turn on EVERY is_turn frame (the linger re-fire)
    b = _base(st)
    if b is not None:
        return b
    if st["is_turn"]:               # fires again during the post-turn linger -> over-rotation Fall
        return _side(st)
    return _recenter(st)


def flash_swipe(st):                # device-FATAL: swipe at the is_turn FLASH (cprog too early)
    b = _base(st)
    if b is not None:
        return b
    # swipe the INSTANT is_turn appears (cprog < 0.20) -> lands >10 off-center -> water
    if st["is_turn"] and float(st.get("cell_progress", 0.0)) < 0.20:
        return _side(st)
    return _recenter(st)


def run(pol, n=80, seed0=13000):
    # narrows excluded: this gate isolates TURN/LATERAL semantics; the gate policies have no
    # narrow-lane logic and narrow deaths would drown the turn signal (narrow.semantics covers those).
    # turn_linger 3 ON: models the device post-turn is_turn persistence (facing-lag) so the DOUBLE-TURN
    # policy can re-fire and MUST die — the exact device regression (ppo_tilt_v1: 12 double-bursts,
    # 8/8 turn deaths). Mirrors the training config.
    env = gym.make("TempleRunSim-v0", max_episode_steps=6000,
                   config=dict(domain_rand=True, curriculum_max_start_dist=0.0,
                               narrows_after_distance=1e9, turn_linger_frames=3)).unwrapped
    dists, surv, lane_at_turn = [], 0, []
    for ep in range(n):
        o, i = env.reset(seed=seed0 + ep)
        st = i["raw_state"]
        while True:
            a = int(pol(st))
            if a in (int(C.Action.TURN_LEFT), int(C.Action.TURN_RIGHT)) and st.get("is_turn"):
                lane_at_turn.append(abs(float(st.get("lane", 0.0))))
            o, r, t, tr, info = env.step(a)
            st = info["raw_state"]
            if t or tr:
                dists.append(float(st["distance"]))
                surv += 1 if tr else 0
                break
    ml = float(np.median(lane_at_turn)) if lane_at_turn else float("nan")
    return float(np.median(dists)), surv, n, ml


def run_gates(n=80, seed0=13000) -> list[dict]:
    """Run all gate policies; return [{name, median, survived, n, lane_at_turn, want, ok}, ...].
    Importable by fidelity_scorecard.py; main() keeps the CLI."""
    rows = [("oracle", _oracle, "survive"), ("prompt(centered)", prompt, "survive"),
            ("PRE-FIRE", prefire, "die"), ("OFF-CENTER turn", off_center, "die"),
            ("SPAM(straights)", spam, "die"), ("DOUBLE-TURN(linger)", double_turn, "die"),
            ("FLASH-SWIPE(early)", flash_swipe, "die")]
    out = []
    for name, pol, want in rows:
        m, s, nn, ml = run(pol, n=n, seed0=seed0)
        frac = s / nn
        good = (frac >= 0.7) if want == "survive" else (frac <= 0.05)
        # survivors must ALSO turn centered (the lateral guard): median |lane|@turn < turn_center_tol
        if want == "survive" and ml == ml and ml > 3.0:
            good = False
        out.append({"name": name, "median": m, "survived": s, "n": nn,
                    "lane_at_turn": ml, "want": want, "ok": good})
    return out


def main():
    print(f"  {'policy':20} | {'median':>7} | {'survive':>8} | {'|lane|@turn':>11} | want")
    ok = True
    for r in run_gates():
        ok = ok and r["ok"]
        print(f"  {r['name']:20} | {r['median']:7.0f} | {r['survived']:4}/{r['n']:<3} | "
              f"{r['lane_at_turn']:11.2f} | {r['want']:7} {'OK' if r['ok'] else 'FAIL <<<'}")
    print("\n" + ("PASS — sim rewards centered is_turn turns; punishes pre-fire / off-center / spam."
                  if ok else "FAIL — sim turn/lateral model does NOT match device; do NOT retrain/deploy."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
