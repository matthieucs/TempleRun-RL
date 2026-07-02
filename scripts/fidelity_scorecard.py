#!/usr/bin/env python3
"""SIM-FIDELITY SCORECARD — the one-command, phone-free feedback loop this project never had.

Diffs the sim against captures/device_reference.json (frozen device truth) across every mechanic
that has killed a device eval: turn window, turn/lateral behavior gates, lateral step-response,
narrow semantics + rates + safe-lane map, crevasse takeoff window, stumble 2-strike, velocity curve.

    .venv/bin/python scripts/fidelity_scorecard.py [--json runs/scorecard.json] [--fast] [--strict]

Verdicts: PASS / FAIL / SKIP (reference not yet measured — a SKIP is NOT a pass).
Exit 0 = no FAIL (add --strict to also fail on SKIP; the RETRAIN GATE is --strict green).
Workflow rule (CLAUDE.md): no retrain, no deploy, no sim-knob commit without this green.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import templerun_rl  # noqa: E402  (registers env)
import templerun_rl.constants as C  # noqa: E402
from fidelity_audit import sim_frames, turn_window, vel_curve, event_rates  # noqa: E402
from train_sim import _oracle  # noqa: E402
from turn_gate import run_gates  # noqa: E402
from probe_strafe import TILT_SCRIPT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REF_PATH = ROOT / "captures" / "device_reference.json"
HOOK_JS = ROOT / "frida" / "temple_run_hook.js"

# a track with NOTHING on it — lateral/stumble physics in isolation
STRAIGHT_CFG = dict(domain_rand=False, curriculum_max_start_dist=0.0,
                    turn_prob=0.0, turn_prob_late=0.0, obstacle_prob=0.0,
                    narrows_after_distance=1e9, zone_change_every_cells=10**6)


def _env(cfg: dict, max_steps=3000):
    return gym.make("TempleRunSim-v0", max_episode_steps=max_steps, config=cfg).unwrapped


def _rollout_policy(pol, cfg, n_eps, seed0, max_steps=2000):
    """Generic scripted-policy rollout -> (median_dist, survive_frac)."""
    env = _env(cfg, max_steps)
    dists, surv = [], 0
    for ep in range(n_eps):
        _, i = env.reset(seed=seed0 + ep)
        st = i["raw_state"]
        while True:
            _, r, term, trunc, info = env.step(int(pol(st)))
            st = info["raw_state"]
            if term or trunc:
                dists.append(float(st["distance"]))
                surv += 1 if trunc else 0
                break
    return float(np.median(dists)), surv / n_eps


def _pct(hist: dict, key) -> float:
    """turn_window runlen hists carry '91%' strings — parse one entry."""
    v = hist.get(key) or hist.get(str(key)) or "0%"
    return float(str(v).rstrip("%"))


# --------------------------------------------------------------------------- checks
def check_velocity(ref, sim) -> dict:
    edges = ref["velocity_curve"]["edges"]
    rbin = ref["velocity_curve"]["mean_per_bin"]
    sbin = vel_curve(sim, edges)
    diffs = [abs(s - r) for r, s in zip(rbin, sbin) if r is not None and s == s]
    mae = float(np.mean(diffs)) if diffs else float("nan")
    return {"ref": "per-bin means", "sim": round(mae, 2), "tol": "MAE < 5",
            "pass": bool(diffs) and mae < 5.0}


def check_turn_window(ref, sim) -> dict:
    rv, sv = ref["turn_window"]["values"], turn_window(sim)
    r_med, s_med = rv["is_turn_cprog"]["med"], sv["is_turn_cprog"].get("med", float("nan"))
    r_p90, s_p90 = rv["is_turn_cprog"]["p90"], sv["is_turn_cprog"].get("p90", float("nan"))
    s_one = _pct(sv["is_turn_runlen"], 1)
    ok = (abs(s_med - r_med) <= 0.08) and (s_p90 <= r_p90 + 0.10) and (s_one >= 80.0)
    return {"ref": f"med {r_med:.2f} p90 {r_p90:.2f} 1f {_pct(rv['is_turn_runlen'], 1):.0f}%",
            "sim": f"med {s_med:.2f} p90 {s_p90:.2f} 1f {s_one:.0f}%",
            "tol": "med ±0.08, p90 +0.10, 1-frame ≥80%", "pass": bool(ok)}


def check_turn_gates(n=80) -> dict:
    gates = run_gates(n=n)
    bad = [g["name"] for g in gates if not g["ok"]]
    return {"ref": "oracle/prompt survive; prefire/off-center/spam die",
            "sim": "all ok" if not bad else f"FAIL: {bad}",
            "tol": "≥70% / ≤5%", "pass": not bad, "detail": gates}


def check_narrow_rates(ref, sim) -> dict:
    band = ref["event_rates"]["bands"].get("1500-3000", {})
    r = float(band.get("narrow%", 0.0))
    seg = [f for f in sim if 1500 <= float(f.get("distance") or 0) < 3000]
    s = float(event_rates(seg)["narrow%"]) if seg else float("nan")
    ok = r > 0 and s == s and 0.6 <= s / r <= 1.6
    return {"ref": round(r, 2), "sim": round(s, 2), "tol": "ratio in [0.6, 1.6]", "pass": bool(ok)}


def check_narrow_semantics(n=30, seed0=21000) -> dict:
    """Sim must reward riding the safe lane and kill the wrong lane — regardless of which map is
    right. PER-STRIP survival (not per-episode): dense-config throughput deaths (turn-after-narrow,
    back-to-back strips) are a separate concern from the kill semantics under test here."""
    cfg = dict(domain_rand=False, curriculum_max_start_dist=0.0,
               narrows_after_distance=300.0, narrow_prob=0.4)

    def wrong(st):
        a = _oracle(st)
        ntl = st.get("narrow_target_lane")
        if ntl is not None and a in (int(C.Action.STRAFE_LEFT), int(C.Action.STRAFE_RIGHT),
                                     int(C.Action.NOOP)):
            # aim the OPPOSITE lane (center-safe -> aim right): must fall off the strip
            bad = -int(ntl) if int(ntl) != 0 else 1
            lane = float(st.get("lane", 0.0))
            tgt = bad * 4.5
            if lane < tgt - 1.0:
                return C.Action.STRAFE_RIGHT
            if lane > tgt + 1.0:
                return C.Action.STRAFE_LEFT
            return C.Action.NOOP
        return a

    def strip_survival(pol):
        env = _env(cfg, 2000)
        strips = fatal = 0
        for ep in range(n):
            _, i = env.reset(seed=seed0 + ep)
            st = i["raw_state"]
            on_strip = False
            while True:
                _, r, term, trunc, info = env.step(int(pol(st)))
                st = info["raw_state"]
                now = isinstance(st.get("path_type"), int) and 17 <= st["path_type"] <= 30
                if now and not on_strip:
                    strips += 1
                if term and (now or on_strip):
                    fatal += 1
                on_strip = now
                if term or trunc:
                    break
        return strips, fatal

    a_s, a_f = strip_survival(_oracle)
    w_s, w_f = strip_survival(wrong)
    a_rate = a_f / max(1, a_s)
    w_rate = w_f / max(1, w_s)
    ok = a_s >= 10 and a_rate <= 0.10 and w_rate >= 0.90
    return {"ref": "safe lane rides strips / wrong lane falls",
            "sim": f"aim dies {a_f}/{a_s} strips, wrong dies {w_f}/{w_s}",
            "tol": "aim ≤10% strip-death, wrong ≥90%", "pass": bool(ok)}


def check_narrow_map(ref) -> dict:
    """Hook NARROW_SIDE (compass map -> runner-relative side via facing) vs the Region2D-sampled
    bands (game-polygon ground truth, human_narrows.jsonl). 17/18 skipped (bimodal NarrowSmall)."""
    nm = ref.get("narrow_map", {})
    bands = nm.get("region_bands")
    if nm.get("confidence") != "measured" or not bands:
        return {"skip": f"reference confidence={nm.get('confidence')} — run the device band capture"}
    m = re.search(r"NARROW_SIDE\s*=\s*\{([^}]*)\}", HOOK_JS.read_text())
    if not m:
        return {"ref": "measured bands", "sim": "NARROW_SIDE not found in hook", "tol": "sign match",
                "pass": False}
    hook = {int(k): v for k, v in re.findall(r"(\d+)\s*:\s*'(\w)'", m.group(1))}
    # mirror the hook's narrowStrafeDirFor: compass side a left/right strafe reaches, per facing
    l_side = {"N": "W", "E": "N", "S": "E", "W": "S"}
    r_side = {"N": "E", "E": "S", "S": "W", "W": "N"}
    mism = []
    for key, bnds in bands.items():
        sub_s, fac = key.split(",")
        sub = int(sub_s)
        if not (19 <= sub <= 30) or fac not in l_side or sub not in hook:
            continue
        center = float(np.mean([lo + hi for lo, hi in bnds])) / 2.0
        want = -1 if center < 0 else 1                       # runner-relative strip side (ground truth)
        side = hook[sub]
        got = -1 if l_side[fac] == side else (1 if r_side[fac] == side else 0)
        if got != want:
            mism.append(f"{key}: hook->{got:+d} vs band center {center:+.1f}")
    return {"ref": f"{len(bands)} region bands (game polygons)",
            "sim": "map agrees with polygons" if not mism else "; ".join(mism),
            "tol": "side sign match", "pass": not mism}


def check_crevasse_curve(ref, n_eps=8, seed0=31000) -> dict:
    """Jump-at-cprog sweep on a crevasse-dense track -> min cprog that survives; must sit in the
    device takeoff band (fatal-below / clearing-above)."""
    cfg = dict(domain_rand=False, curriculum_max_start_dist=0.0,
               turn_prob=0.0, turn_prob_late=0.0, obstacle_prob=0.0,
               narrows_after_distance=1e9, zone_change_every_cells=8)

    def jump_at(x):
        def pol(st):
            # mirror the device --gap-override driver: x gates the ANTICIPATORY fire (gap still
            # ahead); once ON a must-jump cell (the lip), jump immediately.
            if st["must_jump"]:
                return C.Action.JUMP
            if ((st.get("next_wide_gap") or st.get("next_must_jump"))
                    and float(st.get("cell_progress", 0.0)) >= x):
                return C.Action.JUMP
            return C.Action.NOOP
        return pol

    surv_by_x = {}
    for x in [round(0.1 * k, 1) for k in range(1, 10)]:
        _, frac = _rollout_policy(jump_at(x), cfg, n_eps, seed0, max_steps=1200)
        surv_by_x[x] = round(frac, 2)

    sweep = ref["crevasse"].get("controlled_sweep")
    if not sweep:
        return {"skip": "no controlled --gap-override device sweep yet (Phase 4)"}
    # device truth (controlled sweep 2026-07-02): survivals span takeoff 0.26-0.98 — the window is
    # BROAD. The sim must not punish jump timings the device tolerates: at every swept target where
    # the device majority-survived, the sim's survival must be >=0.6 (10 Hz sampling noise allowed
    # at the extremes, where a target can be skipped before it fires).
    dev_ok_x = []
    for tgt, recs in sweep["curve"].items():
        outs = [r["survived"] for r in recs if r["takeoff"] is not None]
        # STRICT majority: target 0.9 measured 1/2 on device (0.98 cleared, 1.02 died — the
        # boundary is a coin flip); a 50% target is ambiguous, not a "device clears" claim.
        if outs and sum(outs) / len(outs) > 0.6:
            dev_ok_x.append(float(tgt))
    mism = []
    for x in sorted(dev_ok_x):
        xr = round(min(max(round(x, 1), 0.1), 0.9), 1)          # nearest sim sweep bucket
        if surv_by_x.get(xr, 0.0) < 0.6:
            mism.append(f"x={x}: device clears, sim {surv_by_x.get(xr)}")
    return {"ref": f"controlled sweep: device clears at targets {sorted(dev_ok_x)}",
            "sim": f"sweep {surv_by_x}" + ("" if not mism else f" | MISM: {'; '.join(mism)}"),
            "tol": "sim ≥0.6 survival wherever device majority-clears", "pass": not mism}


def check_stumble(ref) -> dict:
    """Scripted 2-strike test on a pure straight: 1 mistimed turn = survivable slowdown; a 2nd
    within the kill window dies; a 2nd after it survives."""
    doc = ref["stumble"]["values"]["documented"]
    win, slow = float(doc["kill_window_s"]), float(doc["slowdown"])

    def episode(gap_s: float):
        env = _env(STRAIGHT_CFG, 800)
        _, i = env.reset(seed=41000)
        st = i["raw_state"]
        # settle a few frames, then mistimed turn #1
        for _ in range(5):
            st = env.step(int(C.Action.NOOP))[4]["raw_state"]
        v_before = st["velocity"]
        st = env.step(int(C.Action.TURN_LEFT))[4]["raw_state"]
        v_trough = st["velocity"]
        for _ in range(3):
            st = env.step(int(C.Action.NOOP))[4]["raw_state"]
            v_trough = min(v_trough, st["velocity"])
        first_survived = not st["is_dead"]
        for _ in range(int(round(gap_s / 0.1)) - 4):
            st = env.step(int(C.Action.NOOP))[4]["raw_state"]
        st = env.step(int(C.Action.TURN_LEFT))[4]["raw_state"]
        for _ in range(3):
            if st["is_dead"]:
                break
            st = env.step(int(C.Action.NOOP))[4]["raw_state"]
        return first_survived, bool(st["is_dead"]), v_trough / max(1.0, v_before)

    s1, died_fast, ratio = episode(win - 0.5)
    s2, died_slow, _ = episode(win + 0.5)
    ok = s1 and s2 and died_fast and not died_slow and abs(ratio - slow) <= 0.03
    return {"ref": f"window {win}s, slowdown {slow}",
            "sim": f"1st survives={s1 and s2}, die@{win - 0.5}s={died_fast}, "
                   f"survive@{win + 0.5}s={not died_slow}, slowdown={ratio:.2f}",
            "tol": "window ±0.5s, slowdown ±0.03", "pass": bool(ok)}


def check_lateral_step(ref) -> dict:
    lat = ref.get("lateral", {})
    if lat.get("status") == "pending_probe" or "values" not in lat:
        return {"skip": "no device tilt probe yet (Phase 1: probe_strafe.py --script tilt)"}
    rv = lat["values"]
    env = _env(STRAIGHT_CFG, 400)
    _, i = env.reset(seed=51000)
    frames = [{"label": "seed", "lane": i["raw_state"]["lane"]}]
    for label, act in TILT_SCRIPT:
        st = env.step(int(act))[4]["raw_state"]
        frames.append({"label": label, "lane": st["lane"]})
        if st["is_dead"]:
            break
    by = {}
    for f in frames[1:]:
        by.setdefault(f["label"], []).append(float(f["lane"]))
    # flipL/flipR excluded: bounce-contaminated on device (wall scrape mid-hold, see reference note);
    # the clean hold/settle/pulse phases pin down the same dynamics (alpha, reach, recenter).
    compare = ("Lhold", "Rhold", "Lhold_settle", "Rhold_settle", "Lpulse", "Lpulse_settle",
               "flip_settle")
    mism = []
    for label, ref_recs in rv["phases"].items():
        if label not in compare:
            continue
        lanes = by.get(label)
        if not lanes:
            mism.append(f"{label}: missing in sim replay")
            continue
        ref_end = float(np.median([r["lane_end"] for r in ref_recs]))
        if abs(lanes[-1] - ref_end) > 1.0:
            mism.append(f"{label}: sim end {lanes[-1]:.1f} vs device {ref_end:.1f}")
    return {"ref": f"alpha hold {rv.get('alpha_hold')} / recenter {rv.get('alpha_recenter')} "
                   f"per 0.1s, reach ±{rv.get('reach')}",
            "sim": "match" if not mism else "; ".join(mism),
            "tol": "phase-end lane ±1.0", "pass": not mism}


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", dest="json_out")
    p.add_argument("--sim-eps", type=int, default=80)
    p.add_argument("--fast", action="store_true", help="small rollouts (smoke, not a gate)")
    p.add_argument("--strict", action="store_true", help="SKIP also fails (the retrain gate)")
    a = p.parse_args()
    if a.fast:
        a.sim_eps = 20

    ref = json.loads(REF_PATH.read_text())
    print(f"reference: {REF_PATH.name} (generated {ref['_meta']['generated']})")
    print(f"rolling out oracle sim ({a.sim_eps} eps) ...", flush=True)
    sim = sim_frames(a.sim_eps)

    n_gate = 20 if a.fast else 80
    n_nar = 10 if a.fast else 30
    checks = {
        "velocity": check_velocity(ref, sim),
        "turn.window": check_turn_window(ref, sim),
        "turn.gates": check_turn_gates(n=n_gate),
        "lateral.step": check_lateral_step(ref),
        "narrow.rates": check_narrow_rates(ref, sim),
        "narrow.semantics": check_narrow_semantics(n=n_nar),
        "narrow.map": check_narrow_map(ref),
        "crevasse.curve": check_crevasse_curve(ref, n_eps=4 if a.fast else 8),
        "stumble.window": check_stumble(ref),
    }

    n_fail = n_skip = 0
    print(f"\n{'check':>18} | verdict | detail")
    for name, c in checks.items():
        if "skip" in c:
            n_skip += 1
            print(f"{name:>18} |  SKIP   | {c['skip']}")
        else:
            ok = c["pass"]
            n_fail += 0 if ok else 1
            print(f"{name:>18} | {'PASS' if ok else 'FAIL':^7} | ref: {c['ref']}")
            print(f"{'':>18} |         | sim: {c['sim']}   (tol: {c['tol']})")

    verdict = "GREEN" if (n_fail == 0 and (n_skip == 0 or not a.strict)) else "RED"
    print(f"\n{verdict}: {n_fail} FAIL, {n_skip} SKIP "
          f"({'retrain gate needs --strict green (0 SKIP)' if n_skip else 'full-measured'})")
    if a.json_out:
        out = {"verdict": verdict, "fails": n_fail, "skips": n_skip, "checks": checks,
               "reference_generated": ref["_meta"]["generated"]}
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(out, indent=1, default=str) + "\n")
        print(f"wrote {a.json_out}")
    sys.exit(1 if (n_fail or (a.strict and n_skip)) else 0)


if __name__ == "__main__":
    main()
