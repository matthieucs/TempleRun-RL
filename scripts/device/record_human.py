#!/usr/bin/env python3
"""Passively record a HUMAN-played Temple Run run for sim calibration / fidelity.

Unlike ``record_traces.py`` (which drives a reactive-random policy via byte sends), this
script sends NO actions — the human plays on the phone by touch. It injects
``trace_logger.js``, then logs every ``tr_frame`` (state + hidden + 8-cell track) to a
SIM_CONTRACT JSONL, wrapping episodes at run-start / death. Because the human's input
arrives as touches (not action bytes), the hook tags frames with ``action=null``; we INFER
the per-step action from state rising-edges (jump/slide/strafe/turn) so the trace is still
replay-able by ``sim_fidelity.py``. The inference is best-effort — the un-actioned raw state
(velocity curve, path_type sequence, obstacle spacing, boost) is exact either way.

Usage:
    python scripts/device/record_human.py --serial e4de25a1 --device-index 0 \
        --out traces/human.jsonl [--episodes N] [--max-seconds S]

Play normally on the phone. Each death ends an episode; press play again for the next.
Stop with Ctrl-C (the current file is flushed). Network-off / keep-awake are the caller's job
(``setup_device.py`` already does both).
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import templerun_rl.constants as C  # noqa: E402
# reuse the proven channel + line helpers from the policy recorder
from record_traces import (  # noqa: E402
    TraceChannel, state_line, hidden_line, track_line, alive, run_started,
)

# PathDirection: None0 N1 E2 S3 W4. Left/right neighbor of a heading (matches sim_env).
_LEFT_OF = {1: 4, 4: 3, 3: 2, 2: 1}
_RIGHT_OF = {1: 2, 2: 3, 3: 4, 4: 1}


def infer_action(prev: dict, cur: dict):
    """Infer the gameplay action that produced ``cur`` from ``prev``, via rising edges.

    Returns an action int 0..6, or None if nothing changed (idle frame). Priority follows
    what is observable + unambiguous; strafe direction uses the hidden ``lane`` delta if present.
    """
    if prev is None:
        return None
    ps, cs = prev.get("state", {}), cur.get("state", {})
    # Jump / Slide: flag went False -> True this frame.
    if not ps.get("is_jumping") and cs.get("is_jumping"):
        return int(C.Action.JUMP)
    if not ps.get("is_sliding") and cs.get("is_sliding"):
        return int(C.Action.SLIDE)
    # Turn: facing changed -> left or right relative to the old heading.
    pf, cf = int(ps.get("facing", -1)), int(cs.get("facing", -1))
    if pf > 0 and cf > 0 and pf != cf:
        if _LEFT_OF.get(pf) == cf:
            return int(C.Action.TURN_LEFT)
        if _RIGHT_OF.get(pf) == cf:
            return int(C.Action.TURN_RIGHT)
    # Strafe: is_strafing rising edge; direction from lane delta if available.
    if not ps.get("is_strafing") and cs.get("is_strafing"):
        pl = (prev.get("hidden") or {}).get("lane")
        cl = (cur.get("hidden") or {}).get("lane")
        if pl is not None and cl is not None:
            if cl < pl:
                return int(C.Action.STRAFE_LEFT)
            if cl > pl:
                return int(C.Action.STRAFE_RIGHT)
        return int(C.Action.STRAFE_LEFT)  # unknown direction: default left
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL", "e4de25a1"))
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--host", default=C.STATE_HOST)
    p.add_argument("--out", default="traces/human.jsonl")
    p.add_argument("--episodes", type=int, default=0, help="stop after N episodes (0 = until Ctrl-C)")
    p.add_argument("--max-seconds", type=float, default=0.0, help="overall time budget (0 = none)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    ports = C.device_ports(args.device_index)
    host_gadget = f"{args.host}:{ports['gadget_port']}"
    print(f"recording HUMAN play from serial={args.serial} (gadget {host_gadget}) -> {args.out}",
          flush=True)
    print("Play on the phone now. Ctrl-C to stop.", flush=True)

    chan = TraceChannel(host_gadget, debug=args.debug).connect()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    global_seq = 0
    episodes = 0
    in_episode = False
    prev_frame = None
    steps_this_ep = 0
    last_seq = -1

    def write(fh, obj):
        fh.write(__import__("json").dumps(obj, separators=(",", ":")) + "\n")
        fh.flush()

    fh = open(out_path, "w", encoding="utf-8")
    try:
        while True:
            if args.episodes and episodes >= args.episodes:
                break
            if args.max_seconds and (time.monotonic() - t0) >= args.max_seconds:
                break
            frame, seq = chan.wait_next(last_seq, timeout=2.0)
            if frame is None:
                continue
            last_seq = seq
            st = frame.get("state", {})

            if not in_episode:
                if run_started(st):
                    in_episode = True
                    steps_this_ep = 0
                    prev_frame = None
                    global_seq += 1
                    write(fh, {"t": "ep_start", "seq": global_seq})
                    # first frame of the run: action=null
                    global_seq += 1
                    write(fh, {"t": "step", "seq": global_seq, "action": None,
                               "state": state_line(st),
                               "hidden": hidden_line(frame.get("hidden", {})),
                               "track": track_line(frame.get("track", []))})
                    prev_frame = frame
                    steps_this_ep += 1
                continue

            # in an episode: log every fresh frame
            act = infer_action(prev_frame, frame)
            global_seq += 1
            write(fh, {"t": "step", "seq": global_seq, "action": act,
                       "state": state_line(st),
                       "hidden": hidden_line(frame.get("hidden", {})),
                       "track": track_line(frame.get("track", []))})
            prev_frame = frame
            steps_this_ep += 1

            if not alive(st):
                write(fh, {"t": "ep_end", "reason": "dead"})
                episodes += 1
                in_episode = False
                print(f"  episode {episodes}: {steps_this_ep} steps, "
                      f"final score={st.get('score')}", flush=True)
    except KeyboardInterrupt:
        if in_episode:
            write(fh, {"t": "ep_end", "reason": "interrupt"})
        print("\nstopped.", flush=True)
    finally:
        fh.flush()
        fh.close()
        chan.close()
        print(f"done: {episodes} episode(s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
