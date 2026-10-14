#!/usr/bin/env python3
"""Freeze device-truth reference numbers from captures -> captures/device_reference.json.

Run ONCE per new capture batch; the committed JSON is what fidelity_scorecard.py diffs the sim
against, so re-running the scorecard needs NO phone. Each section records its source captures +
date + hook semantics so a HOOK_VERSION change makes staleness visible.

    .venv/bin/python scripts/validate/extract_device_reference.py
    .venv/bin/python scripts/validate/extract_device_reference.py --lateral captures/strafe_probe_tilt2.jsonl

Sections:
  turn_window     human_calib_v2 (365 turns): is_turn 1-frame flash stats (fidelity_audit.turn_window)
  velocity_curve  mean velocity per distance bin (human capture)
  event_rates     per-alive-frame event % per distance band (human capture)
  stumble         2-strike mechanic mined from stumble_count/stumble_kill_timer across eval captures
  narrow_map      PREFILL from human captures (confidence: partial until the device sweep)
  crevasse        PREFILL from policy captures with actions (takeoff cprog x outcome, gap spans)
  lateral         pending_probe until the tilt probe capture exists (--lateral <probe.jsonl>)
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from analyze_calib import load, crevasse_stats, narrow_stats  # noqa: E402
from fidelity_audit import turn_window, vel_curve, event_rates  # noqa: E402

CAP = Path(__file__).resolve().parents[2] / "captures"
OUT = CAP / "device_reference.json"

HUMAN = "human_calib_v2.jsonl"                       # 14,956 m human run, widest field set
STUMBLE_SRC = ["eval_stumble.jsonl", "eval_stumble2.jsonl", HUMAN]
CREVASSE_SRC = ["eval_stumble.jsonl", "eval_stumble2.jsonl", "turnfix_eval.jsonl",
                "turnfix2_eval.jsonl", "lateral_eval.jsonl",
                "deploy_faithful.jsonl", "deploy_faithful3.jsonl", "deploy_faithful4.jsonl",
                "deploy_faithful4b.jsonl", "deploy_faithful5.jsonl", "deploy_faithful6.jsonl",
                "deploy_faithful6b.jsonl",
                # human play (passive, is_jumping edges give takeoffs): reliable CLEARING cprogs
                "human_narrows.jsonl", HUMAN]
NARROW_SRC = [HUMAN, "human_calib_2026-06-26.jsonl", "human.jsonl", "human_turn_calib.jsonl"]
# hook 15c+ capture with Region2D band sampling (the ground truth for narrow safe lanes)
NARROW_BAND_SRC = "human_narrows.jsonl"

VEL_EDGES = [0, 200, 500, 1000, 1500, 2000, 3000, 4000, 6000, 10000]
BANDS = [(500, 1500), (1500, 3000), (3000, 6000)]


def _frames(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _load_many(names: list[str]) -> tuple[dict, list[str]]:
    """Merged {global_ep: frames} across files (ep ids offset per file) + list of files found."""
    eps, used, base = {}, [], 0
    for n in names:
        p = CAP / n
        if not p.exists():
            continue
        used.append(n)
        d = load(str(p))
        for k in sorted(d):
            eps[base] = d[k]
            base += 1
    return eps, used


def stumble_stats(eps) -> dict:
    """Mine the 2-strike stumble mechanic. A stumble = stumble_count increment; the gap to the
    PREVIOUS stumble is the prior frame's stumble_kill_timer (it counts UP = time-since-stumble
    and resets on each stumble). Fatal = is_dead within ~1 s of the increment."""
    slowdowns, fatal_gaps, surv_gaps = [], [], []
    n_events = 0
    for frames in eps.values():
        prev_c = None
        seen_one = False
        for i, f in enumerate(frames):
            c = f.get("stumble_count")
            if c is None:
                continue
            if prev_c is not None and c > prev_c:
                n_events += 1
                fatal = any(g.get("is_dead") for g in frames[i:i + 10])
                # velocity dip: pre-stumble velocity vs the trough right after
                vb = frames[i - 1].get("velocity") if i else None
                va = [g.get("velocity") for g in frames[i:i + 6] if g.get("velocity")]
                if vb and va and not fatal:
                    slowdowns.append(round(min(va) / vb, 3))
                if seen_one:
                    gap = frames[i - 1].get("stumble_kill_timer")
                    if gap is not None and gap < 30.0:      # huge timer = stale/first-of-run
                        (fatal_gaps if fatal else surv_gaps).append(round(float(gap), 2))
                seen_one = True
            prev_c = c
    return {
        "events": n_events,
        "slowdown_med": float(np.median(slowdowns)) if slowdowns else None,
        "slowdown_all": sorted(slowdowns),
        "double_stumble_gap_fatal": sorted(fatal_gaps),
        "double_stumble_gap_survived": sorted(surv_gaps),
        # band the kill window sits in: [max fatal gap, min survived gap]
        "kill_window_band": [max(fatal_gaps) if fatal_gaps else None,
                             min(surv_gaps) if surv_gaps else None],
        # device-confirmed session values (probe_turn_mistake, CLAUDE.md §13) kept as the anchor
        "documented": {"kill_window_s": 2.0, "slowdown": 0.85},
    }


def lateral_stats(path: Path) -> dict:
    """Step-response from a probe_strafe TILT capture. Device dynamics (measured 2026-07-02,
    lateral-accel-15a): lane is a FIRST-ORDER LAG toward the commanded target (+/-7 hold, 0 level) —
    lane += (target-lane)*alpha per 10 Hz frame, alpha ~0.45 once moving (~1 frame command latency),
    symmetric for reach and recenter. px snaps to the target instantly; lane follows."""
    frames = _frames(path)
    tgt_of = {"Lhold": -7.0, "flipL": -7.0, "Lpulse": -7.0, "Rhold": 7.0, "flipR": 7.0}
    alphas = {"hold": [], "recenter": []}
    phases: dict[str, list] = {}
    by_phase: dict[str, list] = {}
    prev = None
    for f in frames:
        key = f"{f.get('run', 0)}:{f.get('label', '?')}"
        by_phase.setdefault(key, []).append(f)
        if (prev is not None and prev.get("run") == f.get("run")
                and f.get("i") == prev.get("i", -9) + 1 and not f.get("is_dead")):
            lab = f.get("label")
            tgt = tgt_of.get(lab, 0.0)
            l0, l1 = prev.get("lane"), f.get("lane")
            if l0 is not None and l1 is not None and abs(tgt - l0) > 0.5:
                a = (l1 - l0) / (tgt - l0)
                if -0.2 < a < 1.2:
                    alphas["hold" if lab in tgt_of else "recenter"].append(a)
        prev = f
    for key, fs in by_phase.items():
        lanes = [f.get("lane") for f in fs if f.get("lane") is not None]
        if len(lanes) < 2:
            continue
        label = key.split(":", 1)[1]
        # drop INPUT-LOCKED reps (early-run lateral lock: a hold phase where the lane never moved) —
        # they would pollute the phase medians the scorecard diffs against.
        if label in tgt_of and max(abs(v) for v in lanes) < 0.2:
            continue
        # drop truncated reps (run died mid-phase, <60% of the phase captured)
        full_n = {"base": 6, "Lhold": 10, "Rhold": 10, "Lpulse": 2, "flipL": 8, "flipR": 8}.get(
            label, 15)
        if len(lanes) < 0.9 * full_n:
            continue
        phases.setdefault(label, []).append(
            {"n": len(lanes), "lane_start": round(lanes[0], 2), "lane_end": round(lanes[-1], 2)})
    def med(v):
        return round(float(np.median(v)), 3) if v else None
    # "moving" alpha: exclude the ~0 latency frames (first frame after a command flip)
    mov_h = [a for a in alphas["hold"] if a > 0.1]
    mov_r = [a for a in alphas["recenter"] if a > 0.1]
    return {
        "model": "first-order lag, alpha per 0.1s control frame",
        "phases": phases,
        "alpha_hold": med(mov_h), "alpha_recenter": med(mov_r),
        "alpha_hold_all": med(alphas["hold"]), "alpha_recenter_all": med(alphas["recenter"]),
        "reach": 7.0,
        # holding INTO the wall can scrape/bounce (lane snaps -6.6 -> +1..+3 then re-glides; seen in
        # flipL runs 4/6) — a wall-stumble recenter. flip phases are bounce-contaminated; the scorecard
        # compares the clean hold/settle/pulse phases only.
        "wall_bounce_observed": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lateral", help="probe_strafe tilt capture to fill the lateral section")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()

    ref: dict = {"_meta": {"generated": _dt.date.today().isoformat(),
                           "note": "device truth frozen from captures; scorecard diffs sim vs THIS"}}

    human = _frames(CAP / HUMAN)
    ref["turn_window"] = {"source": HUMAN, "values": turn_window(human)}
    ref["velocity_curve"] = {"source": HUMAN, "edges": VEL_EDGES,
                             "mean_per_bin": [None if v != v else round(v, 1)
                                              for v in vel_curve(human, VEL_EDGES)]}
    bands = {}
    for lo, hi in BANDS:
        seg = [f for f in human if lo <= float(f.get("distance") or 0) < hi]
        if seg:
            bands[f"{lo}-{hi}"] = {k: round(v, 2) if isinstance(v, float) else v
                                   for k, v in event_rates(seg).items()}
    ref["event_rates"] = {"source": HUMAN, "bands": bands}

    st_eps, st_used = _load_many(STUMBLE_SRC)
    ref["stumble"] = {"source": st_used, "values": stumble_stats(st_eps)}

    na_eps, na_used = _load_many(NARROW_SRC)
    nm = {"source": na_used, "confidence": "partial",
          "note": "human-play mined; needs the --narrow-override device sweep",
          "values": narrow_stats(na_eps)}
    bp = CAP / NARROW_BAND_SRC
    if bp.exists():
        # Region2D GROUND TRUTH (hook 15c): per (next subtype, facing) the game-polygon safe band,
        # validated against human on-strip outcomes (every strip death rode OUTSIDE its band,
        # captures/human_narrows.jsonl 2026-07-02). Long-narrow strip center ~ +/-4.5, half-width
        # ~2.0-2.5; NarrowSmall 17/18 is BIMODAL (two side strips), lo/hi spans one side per sample.
        bands: dict = {}
        for f in _frames(bp):
            lo = f.get("narrow_band_lo")
            if lo is None:
                continue
            key = f"{f.get('next_path_type')},{ {1:'N',2:'E',3:'S',4:'W'}.get(f.get('facing'),'?') }"
            bands.setdefault(key, set()).add((int(lo), int(f["narrow_band_hi"])))
        nm["region_bands"] = {k: sorted(v) for k, v in sorted(bands.items())}
        nm["confidence"] = "measured"
        nm["source"] = na_used + [NARROW_BAND_SRC]
    ref["narrow_map"] = nm

    cr_eps, cr_used = _load_many(CREVASSE_SRC)
    ref["crevasse"] = {"source": cr_used, "confidence": "partial",
                       "values": crevasse_stats(cr_eps)}
    csp = CAP / "crevasse_sweep.jsonl"
    if csp.exists():
        # CONTROLLED jump-at-cprog device sweep (capture_calib --gap-override): unambiguous
        # takeoff -> outcome. 2026-07-02: survivals span 0.26-0.98; deaths = late from ON the gap
        # cell (0.92/1.02 fired from pt36) + 2 mid outliers -> the device window is BROAD, no
        # early-jump kill (stepup gate refuted).
        eps: dict = {}
        for f in _frames(csp):
            eps.setdefault(f["ep"], []).append(f)
        curve: dict = {}
        for e, fr in eps.items():
            for i, f in enumerate(fr):
                cl = f.get("narrow_cmd") or ""
                if not (cl.startswith("gapjump") or cl.startswith("gapmissed")):
                    continue
                tgt = cl.split("tgt")[1]
                actual = round(float(cl.split("@")[1].split("/")[0]), 2) if "@" in cl else None
                died = any(g.get("is_dead") for g in fr[i:i + 15])
                curve.setdefault(tgt, []).append({"takeoff": actual, "survived": not died})
        ref["crevasse"]["controlled_sweep"] = {"source": "crevasse_sweep.jsonl", "curve": curve}
        ref["crevasse"]["confidence"] = "measured"

    if a.lateral:
        lp = Path(a.lateral)
        ref["lateral"] = {"source": lp.name, "hook": "lateral-accel-15a",
                          "values": lateral_stats(lp)}
    else:
        ref["lateral"] = {"status": "pending_probe",
                          "note": "run probe_strafe.py --script tilt, then re-run with --lateral"}

    Path(a.out).write_text(json.dumps(ref, indent=1) + "\n")
    print(f"wrote {a.out}")
    for k, v in ref.items():
        if k == "_meta":
            continue
        src = v.get("source") or v.get("status")
        print(f"  {k:>15}: {src}")


if __name__ == "__main__":
    main()
