#!/usr/bin/env python3
"""Controlled STRAFE-dynamics probe: measure exactly how a strafe moves the runner on device, so the
sim's lateral model can be CALIBRATED (not guessed). Read-only w.r.t. game logic (only sends strafe/NOOP
actions in a live run; no hook rules).

Drives a fixed script on straights and logs lane/player_x_offset/is_strafing per frame:
  NOOP baseline -> single StrafeLeft -> NOOP settle -> single StrafeRight -> NOOP settle -> hold-left x3.
Resets and repeats across runs (dies at the first turn/obstacle; we only need the early straight).

    python scripts/probes/probe_strafe.py --runs 8 --out captures/strafe_probe.jsonl
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import templerun_rl.constants as C  # noqa: E402
from templerun_rl.client import StateClient, ActionClient  # noqa: E402

SL, SR, NOOP = 0, 1, 6
# LEGACY script (discrete-persistent strafe era): single taps + settles.
LEGACY_SCRIPT = (
    [("base", NOOP)] * 6 +
    [("L1", SL)] + [("L1_settle", NOOP)] * 12 +
    [("R1", SR)] + [("R1_settle", NOOP)] * 12 +
    [("R1b", SR)] + [("R1b_settle", NOOP)] * 12 +
    [("Lhold", SL)] * 3 + [("Lhold_settle", NOOP)] * 12
)
# TILT script (lateral-tilt-14b semantics: byte0=hold LEFT, byte1=hold RIGHT, NoOp=recenter).
# Measures: glide rate toward target (holds), symmetric recenter rate (settles), pulse response
# (partial tilt), full-range L->R crossing. fidelity_scorecard.py REPLAYS this same script in the
# sim (lateral.step check), so keep labels/lengths stable.
TILT_SCRIPT = (
    [("base", NOOP)] * 6 +
    [("Lhold", SL)] * 10 + [("Lhold_settle", NOOP)] * 15 +
    [("Rhold", SR)] * 10 + [("Rhold_settle", NOOP)] * 15 +
    [("Lpulse", SL)] * 2 + [("Lpulse_settle", NOOP)] * 15 +
    [("flipL", SL)] * 8 + [("flipR", SR)] * 8 + [("flip_settle", NOOP)] * 15
)
SCRIPTS = {"tilt": TILT_SCRIPT, "legacy": LEGACY_SCRIPT}
LOG = ("distance", "lane", "player_x_offset", "is_strafing", "path_type", "facing", "is_dead",
       "cell_progress", "_hv")


def reset(ac: ActionClient, sc: StateClient, last_seq: int, timeout=8.0):
    """Send byte 8 then 9 (game-over restart / menu start); wait until on the track."""
    for b in (8, 9):
        ac.send(b); time.sleep(0.4)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            st, last_seq = sc.wait_fresh(last_seq, timeout=3.0)
        except TimeoutError:
            continue
        if st and (st.get("is_on_ground") or (st.get("path_type") or -1) >= 0) and not st.get("is_dead"):
            return last_seq
    return last_seq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--runs", type=int, default=8)
    p.add_argument("--out", default="captures/strafe_probe.jsonl")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--script", choices=sorted(SCRIPTS), default="tilt")
    p.add_argument("--start-dist", type=float, default=50.0,
                   help="wait until this run distance before the script (early-run lateral input is "
                        "LOCKED: holds at dist<~25-45 never move the runner — measured 2026-07-02)")
    a = p.parse_args()
    script = SCRIPTS[a.script]
    ports = C.device_ports(a.device_index)
    sc = StateClient(host=a.host, port=ports["state_port"], connect_timeout=8.0).connect(required=True)
    ac = ActionClient(host=a.host, port=ports["action_port"], connect_timeout=8.0).connect(required=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("w")
    last_seq, n = 0, 0
    def react(st, act):
        """Survive obstacles while the lateral script runs: substitute a jump/slide byte when one is
        due. Safe for the measurement — the hook tilt command is LEVEL-triggered (sticky until the
        next byte0/1/6), so one substituted tick doesn't change the commanded tilt."""
        if st is None:
            return act
        if st.get("must_jump") or st.get("next_must_jump") or st.get("next_wide_gap"):
            return 2
        if st.get("must_slide") or st.get("next_must_slide"):
            return 3
        return act

    for run in range(a.runs):
        last_seq = reset(ac, sc, last_seq)
        # ride out the early-run input lock (reactive NoOp) until the runner is steerable
        t0 = time.monotonic()
        st = None
        while time.monotonic() - t0 < 15.0:
            ac.send(react(st, NOOP))
            try:
                st, last_seq = sc.wait_fresh(last_seq, timeout=3.0)
            except TimeoutError:
                break
            if st and float(st.get("distance") or 0) >= a.start_dist:
                break
            if st and st.get("is_dead"):
                break
        for i, (label, act) in enumerate(script):
            sent = react(st, act)
            ac.send(sent)
            try:
                st, last_seq = sc.wait_fresh(last_seq, timeout=3.0)
            except TimeoutError:
                break
            if st is None:
                continue
            rec = {"run": run, "i": i, "label": label, "action": sent}
            for k in LOG:
                rec[k] = st.get(k)
            f.write(json.dumps(rec) + "\n"); n += 1
            if st.get("is_dead"):
                break
        f.flush()
        print(f"run {run}: {n} frames so far", flush=True)
    f.close(); sc.close(); ac.close()
    print(f"wrote {n} frames -> {out}")


if __name__ == "__main__":
    main()
