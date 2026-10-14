#!/usr/bin/env python3
"""HUMAN-DEMONSTRATION recorder on the MAIN hook (17b), 2026-07-08 USER-directed pivot.

Records the full 30 Hz state stream (temple_run_hook.js on 9999) while a HUMAN plays by hand, in the
SAME JSONL shape as the policy captures (capture_calib) so every forensic analyzer runs on both
(decode_highground_death / analyze_crevasse_device / device_jumphold / rejump_lethality / ...).

It first sends action byte 10 (hook 17b HUMAN MODE) which REVERTS the accelerometer replacement —
without it the human's tilt is dead (the replacement returns the policy's tiltCmd). Swipes
(jump/slide/turn) were never intercepted. After a human session, RE-INJECT the hook before any
policy capture (the revert is one-way per injection).

    python scripts/record_human_state.py --out captures/human_r1.jsonl
    (PLAY on the phone; Ctrl+C to stop — per-episode summaries print live)

action=null on every frame: the human's actions are recovered from state rising-edges
(is_jumping/is_sliding/facing/lane) by the analyzers.
"""
from __future__ import annotations
import argparse, json, socket, time
from pathlib import Path

STATE_PORT, ACTION_PORT = 9999, 9998


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None,
                   help="output JSONL; default = captures/human_<UTCstamp>.jsonl (NEVER overwrites)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-human-mode", action="store_true",
                   help="skip sending byte 10 (only if tilt passthrough is already active)")
    a = p.parse_args()
    # CLOBBER-PROOF (2026-07-08 lesson: restarts with a fixed --out truncated a dozen human runs):
    # default to a unique timestamped file; if --out exists, append a numeric suffix instead of truncating.
    if a.out is None:
        a.out = time.strftime("captures/human_%Y%m%d_%H%M%S.jsonl", time.gmtime())
    op = Path(a.out)
    if op.exists():
        i = 1
        while Path(f"{op.with_suffix('')}_{i}.jsonl").exists():
            i += 1
        a.out = f"{op.with_suffix('')}_{i}.jsonl"
        print(f"(--out existed; writing {a.out} instead — never overwriting)")

    act_sock = None
    if not a.no_human_mode:
        try:
            # KEEP the action socket OPEN for the whole session — per-action connect/close DROPS bytes
            # (docs/GADGET.md; the same reason ActionClient holds one persistent socket).
            act_sock = socket.create_connection((a.host, ACTION_PORT), timeout=5)
            act_sock.sendall(bytes([10]))
            time.sleep(0.5)
            print("HUMAN MODE enabled (byte 10) — your tilt now controls the runner.")
        except OSError as e:
            print(f"WARNING: could not enable human mode ({e}) — tilt may be dead.")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    st = socket.create_connection((a.host, STATE_PORT), timeout=8)
    st.settimeout(10.0)
    f = out.open("w")
    buf = b""
    ep = 0; step = 0; n = 0
    was_dead = None
    ep_t0 = time.time()
    print(f"RECORDING -> {out}   (PLAY NOW; Ctrl+C to stop)")
    try:
        while True:
            try:
                chunk = st.recv(65536)
            except socket.timeout:
                print("(stream quiet 10s — menu/pause? still recording)")
                continue
            if not chunk:
                print("stream closed (hook unloaded?)"); break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                dead = bool(d.get("is_dead"))
                if was_dead is None:
                    was_dead = dead
                if was_dead and not dead:
                    ep += 1; step = 0; ep_t0 = time.time()
                    print(f"--- ep {ep} started ---", flush=True)
                if not was_dead and dead:
                    print(f"=== ep {ep} END: dist={float(d.get('distance') or 0):.0f} "
                          f"pt={d.get('path_type')} dt={d.get('death_type')} "
                          f"({time.time()-ep_t0:.0f}s) ===", flush=True)
                was_dead = dead
                rec = {"ep": ep, "step": step, "action": None, "t": round(time.time(), 3)}
                rec.update(d)
                f.write(json.dumps(rec) + "\n")
                step += 1; n += 1
                if n % 3000 == 0:
                    f.flush()
                    d_now = float(d.get("distance") or 0)
                    print(f"  [{n} frames on disk | ep {ep} | dist {d_now:.0f} | dead={dead}]", flush=True)
    except KeyboardInterrupt:
        print("\nstopped by user.")
    finally:
        f.close(); st.close()
        if act_sock is not None:
            act_sock.close()
        print(f"wrote {n} frames over {ep} episode(s) -> {out}")


if __name__ == "__main__":
    main()
