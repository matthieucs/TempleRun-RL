#!/usr/bin/env python3
"""Record REAL Temple Run phone play to a trace JSONL file for offline sim-fidelity testing.

This is component (A) of docs/SIM_CONTRACT.md: it injects ``frida/trace_logger.js`` into the
Frida gadget (exactly like ``frida/gadget_inject.py``), drives the running game with a built-in
REACTIVE-RANDOM policy, and writes the recorded trajectory as JSONL in the SIM_CONTRACT §1 schema
(``ep_start`` / ``step`` / ``ep_end`` lines).

Why the gadget channel (not just the 9999 stream): ``trace_logger.js`` emits a rich per-tick
``tr_frame`` over Frida ``send()`` — the 22 raw state fields + a ``hidden`` block + the next K
PathElements (``track``) + the ACTION BYTE the Update hook dispatched on that tick. Tagging the
frame with the dispatched action *inside the hook* is race-free: we log exactly the action that
produced each frame (the host never has to guess the alignment). So the recorder consumes those
``tr_frame`` messages for the logged data, and only uses ``ActionClient`` (port 9998) to push the
reactive-random policy's chosen actions into the game.

Resilience: brief stalls (screen blip, gadget re-inject) just pause the loop; the recorder keeps
the last frame and resumes when frames return, up to ``--stall-timeout`` per wait.

Usage:
    python scripts/record_traces.py --episodes 20 --out traces/run.jsonl
    python scripts/record_traces.py --serial e4de25a1 --episodes 50 --out traces/topaz.jsonl
    python scripts/record_traces.py --device-index 1 --episodes 10 --out traces/phone1.jsonl

Ports default to device slot 0 (in-phone 9999/9998/27042). Pass ``--device-index N`` (host ports
offset by N*100, matching constants.device_ports) or override individual ports. ``adb forward`` for
those ports must already be in place — run ``python scripts/setup_device.py --no-inject`` first (this
script does its own injection, so use --no-inject there) or just rely on prior forwards.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path

import frida

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from templerun_rl import constants as C  # noqa: E402
from templerun_rl.client import ActionClient  # noqa: E402

TRACE_SCRIPT = REPO_ROOT / "frida" / "trace_logger.js"


# ---------------------------------------------------------------------------
# Reactive-random policy
# ---------------------------------------------------------------------------
class ReactiveRandomPolicy:
    """Mostly-surviving, varied policy driven off the raw state dict.

    Priority: react to obstacles/turns (so episodes survive and exercise jump/slide/turn), then
    fall back to a mostly-NoOp idle with occasional strafes (variety without suiciding).

    - ``must_jump``  -> JUMP
    - ``must_slide`` -> SLIDE
    - ``is_turn`` with a valid side -> TURN toward a random VALID side (forcing an invalid turn
      kills the run, so we only turn when can_turn_* says it's allowed)
    - else: mostly NOOP, occasionally a random strafe (lateral variety, low death risk)
    """

    def __init__(self, rng: random.Random, strafe_prob: float = 0.08):
        self.rng = rng
        self.strafe_prob = strafe_prob

    def act(self, s: dict) -> int:
        if s.get("must_jump"):
            return int(C.Action.JUMP)
        if s.get("must_slide"):
            return int(C.Action.SLIDE)
        if s.get("is_turn"):
            sides = []
            if s.get("can_turn_left"):
                sides.append(int(C.Action.TURN_LEFT))
            if s.get("can_turn_right"):
                sides.append(int(C.Action.TURN_RIGHT))
            if sides:
                return self.rng.choice(sides)
            # is_turn but neither side valid yet -> hold (NoOp); we'll turn when a side opens
            return int(C.Action.NOOP)
        if self.rng.random() < self.strafe_prob:
            return self.rng.choice((int(C.Action.STRAFE_LEFT), int(C.Action.STRAFE_RIGHT)))
        return int(C.Action.NOOP)


# ---------------------------------------------------------------------------
# Frida trace-frame channel: inject trace_logger.js and surface tr_frame messages.
# ---------------------------------------------------------------------------
class TraceChannel:
    """Connects to the gadget, loads trace_logger.js, and exposes the latest tr_frame.

    The frames arrive on the Frida message thread; we keep the newest in a slot (with a seq
    counter) and also push every frame to a queue so the recorder can consume them in order
    when it wants the per-tick stream. The recorder only logs frames it has decided to keep
    (between ep_start and ep_end), so we expose both ``latest()`` and ``drain()``.
    """

    def __init__(self, host_gadget: str, debug: bool = False):
        self.host_gadget = host_gadget
        self.debug = debug
        self._device = None
        self._session = None
        self._script = None
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=4096)
        self._ready = threading.Event()

    def connect(self, connect_timeout: float = 10.0) -> "TraceChannel":
        with open(TRACE_SCRIPT, "r", encoding="utf-8") as f:
            source = f.read()
        dm = frida.get_device_manager()
        self._device = dm.add_remote_device(self.host_gadget)
        # the gadget exposes a single process named "Gadget"
        self._session = self._device.attach("Gadget")
        self._script = self._session.create_script(source)
        self._script.on("message", self._on_message)
        self._script.load()
        print(f"=== trace_logger.js loaded via {self.host_gadget} ===", flush=True)
        # wait for the agent to signal 'ready' (libil2cpp resolved, hooks installed)
        if not self._ready.wait(timeout=connect_timeout):
            print("WARN: agent did not report 'ready' within "
                  f"{connect_timeout:.0f}s (frames may still arrive once a run starts)", flush=True)
        return self

    def _on_message(self, message, data):
        mtype = message.get("type")
        if mtype == "send":
            payload = message.get("payload")
            if isinstance(payload, dict) and "tr_frame" in payload:
                frame = payload["tr_frame"]
                with self._lock:
                    self._latest = frame
                    self._seq += 1
                try:
                    self._q.put_nowait(frame)
                except queue.Full:
                    # drop the oldest to keep up (recorder fell behind briefly)
                    try:
                        self._q.get_nowait()
                        self._q.put_nowait(frame)
                    except queue.Empty:
                        pass
            elif isinstance(payload, dict) and "tr" in payload:
                tr = payload["tr"]
                if isinstance(tr, list) and tr and tr[0] == "ready":
                    self._ready.set()
                if self.debug:
                    print("DBG:", tr, flush=True)
        elif mtype == "log":
            print(message.get("payload", ""), flush=True)
            if "ready:" in str(message.get("payload", "")):
                self._ready.set()
        elif mtype == "error":
            print("FRIDA ERROR:", message.get("stack", message), flush=True)

    def latest(self):
        with self._lock:
            return self._latest, self._seq

    def drain(self):
        """Pop all queued frames currently available (in arrival order)."""
        frames = []
        while True:
            try:
                frames.append(self._q.get_nowait())
            except queue.Empty:
                break
        return frames

    def wait_next(self, after_seq: int, timeout: float):
        """Block until a frame newer than ``after_seq`` arrives (or timeout). Returns (frame, seq)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._seq > after_seq:
                    return self._latest, self._seq
            time.sleep(0.005)
        return None, after_seq

    def close(self):
        try:
            if self._session is not None:
                self._session.detach()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Frame -> SIM_CONTRACT line helpers
# ---------------------------------------------------------------------------
def state_line(state: dict) -> dict:
    """Project a tr_frame state onto exactly the 22 canonical STATE_FIELDS (order + defaults)."""
    out = {}
    for f in C.STATE_FIELDS:
        v = state.get(f, C.STATE_DEFAULTS.get(f))
        out[f] = v
    return out


def hidden_line(hidden: dict) -> dict:
    """Pass through the hidden block, normalizing the documented keys (missing -> null)."""
    h = hidden or {}
    keys = ("distance", "lane", "ground_height", "slide_t", "strafe_t", "stumble_t", "death_type")
    out = {k: h.get(k, None) for k in keys}
    # keep the extra best-effort location (the harness ignores unknown keys)
    if "location" in h:
        out["location"] = h["location"]
    return out


def track_line(track) -> list:
    """Pass through the track block as a list of {type,level,obstacles,is_turn,can_turn_*} dicts."""
    if not track:
        return []
    keys = ("type", "level", "obstacles", "is_turn", "can_turn_left", "can_turn_right")
    return [{k: el.get(k) for k in keys} for el in track]


def alive(state: dict) -> bool:
    return not bool(state.get("is_dead")) and not bool(state.get("is_game_over"))


def run_started(state: dict) -> bool:
    return (alive(state)
            and (bool(state.get("is_on_ground")) or int(state.get("path_type", -1)) >= 0))


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self, chan: TraceChannel, actions: ActionClient, policy: ReactiveRandomPolicy,
                 out_path: Path, step_dt: float, max_steps: int, stall_timeout: float,
                 reset_timeout: float):
        self.chan = chan
        self.actions = actions
        self.policy = policy
        self.out_path = out_path
        self.step_dt = step_dt
        self.max_steps = max_steps
        self.stall_timeout = stall_timeout
        self.reset_timeout = reset_timeout
        self._fh = None
        self._global_seq = 0  # monotonic frame counter across the whole file

    def __enter__(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out_path, "w", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._fh.flush()

    def _wait_fresh(self, after_seq: int, timeout: float):
        """Wait for the next trace frame after ``after_seq`` (tolerates brief stalls)."""
        frame, seq = self.chan.wait_next(after_seq, timeout)
        return frame, seq

    def _reset(self) -> "tuple[dict, int]":
        """Start a fresh run (bytes 8 then 9, gated gadget-side) and wait until it's progressing.

        Returns (first_started_frame, seq). Raises TimeoutError if no run starts in time.
        """
        deadline = time.monotonic() + self.reset_timeout
        _, seq = self.chan.latest()
        last_started = None
        consecutive = 0
        # nudge into a run; re-send across the wait because a relaunched app lands on the menu
        for a in C.RESET_ACTIONS:
            self.actions.try_send(int(a))
        while time.monotonic() < deadline:
            frame, seq = self._wait_fresh(seq, timeout=min(self.stall_timeout, self.reset_timeout))
            if frame is None:
                # stalled: re-nudge and keep waiting within the reset budget
                for a in C.RESET_ACTIONS:
                    self.actions.try_send(int(a))
                continue
            st = frame.get("state", {})
            if run_started(st):
                consecutive += 1
                last_started = (frame, seq)
                if consecutive >= C.DEFAULT_RESET_SETTLE_STEPS:
                    return last_started
            else:
                consecutive = 0
                # if we've drifted back to a non-gameplay screen, re-nudge
                for a in C.RESET_ACTIONS:
                    self.actions.try_send(int(a))
        if last_started is not None:
            return last_started
        raise TimeoutError("no fresh run within reset_timeout")

    def record_episode(self, ep_index: int) -> str:
        """Record one episode. Returns the ep_end reason ('dead'|'reset'|'timeout')."""
        # --- start a fresh run ---
        try:
            frame, seq = self._reset()
        except TimeoutError:
            # couldn't start a run (stuck screen / ad). Emit a 0-step episode marked 'reset'.
            self._global_seq += 1
            self._write({"t": "ep_start", "seq": self._global_seq})
            self._write({"t": "ep_end", "reason": "reset"})
            return "reset"

        self._global_seq += 1
        self._write({"t": "ep_start", "seq": self._global_seq})
        # the first (reset) frame: action = null (no gameplay action produced it)
        self._emit_step(frame, action_override=None)

        # --- drive the run ---
        reason = "timeout"
        steps = 0
        last_action = None
        while steps < self.max_steps:
            st = frame.get("state", {})
            if not alive(st):
                reason = "dead"
                break
            # choose + send the reactive action
            a = self.policy.act(st)
            self.actions.try_send(int(a))
            last_action = a
            # pace one control tick, then take the newest frame produced AFTER the action
            time.sleep(self.step_dt)
            nframe, nseq = self._wait_fresh(seq, timeout=self.stall_timeout)
            if nframe is None:
                # stall: the device paused (screen blip / re-inject). Treat as a soft reset.
                reason = "reset"
                break
            frame, seq = nframe, nseq
            # The JS already tags the frame with the action the hook dispatched on that tick.
            # Prefer that authoritative tag; fall back to the action we sent if the tag is null
            # (e.g. several frames between dispatches — the action still "produced" this frame).
            tag = frame.get("action", None)
            self._emit_step(frame, action_override=tag if tag is not None else last_action)
            steps += 1
            st2 = frame.get("state", {})
            if not alive(st2):
                reason = "dead"
                break

        self._write({"t": "ep_end", "reason": reason})
        return reason

    def _emit_step(self, frame: dict, action_override) -> None:
        self._global_seq += 1
        line = {
            "t": "step",
            "seq": self._global_seq,
            "action": action_override,
            "state": state_line(frame.get("state", {})),
            "hidden": hidden_line(frame.get("hidden", {})),
            "track": track_line(frame.get("track", [])),
        }
        self._write(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL", "e4de25a1"),
                   help="device serial (default: e4de25a1 = topaz). Informational only; ports drive the connection.")
    p.add_argument("--episodes", type=int, default=10, help="number of episodes to record")
    p.add_argument("--out", default="traces/run.jsonl", help="output JSONL path")
    p.add_argument("--device-index", type=int, default=0,
                   help=f"0-based device slot; host ports = in-phone port + index*{C.DEVICE_PORT_STRIDE}")
    p.add_argument("--host", default=C.STATE_HOST, help="host the forwarded ports live on")
    p.add_argument("--gadget-port", type=int, default=None, help="override host gadget port")
    p.add_argument("--state-port", type=int, default=None, help="override host state port (unused; frames come via gadget)")
    p.add_argument("--action-port", type=int, default=None, help="override host action port")
    p.add_argument("--step-dt", type=float, default=C.DEFAULT_STEP_DT, help="seconds per control tick (~10 Hz)")
    p.add_argument("--max-steps", type=int, default=C.DEFAULT_MAX_EPISODE_STEPS, help="per-episode step ceiling")
    p.add_argument("--stall-timeout", type=float, default=C.DEFAULT_STEP_STALL_TIMEOUT,
                   help="seconds to wait for the next frame before treating a stall as a soft reset")
    p.add_argument("--reset-timeout", type=float, default=15.0, help="seconds to wait for a fresh run to start")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the reactive-random policy")
    p.add_argument("--strafe-prob", type=float, default=0.08, help="idle-frame strafe probability")
    p.add_argument("--debug", action="store_true", help="print agent DBG: diagnostics")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ports = C.device_ports(args.device_index)
    gadget_port = args.gadget_port if args.gadget_port is not None else ports["gadget_port"]
    action_port = args.action_port if args.action_port is not None else ports["action_port"]
    host_gadget = f"{args.host}:{gadget_port}"

    print(f"recording {args.episodes} episode(s) from serial={args.serial} "
          f"(gadget {host_gadget}, action {args.host}:{action_port}) -> {args.out}", flush=True)

    chan = TraceChannel(host_gadget, debug=args.debug).connect()
    actions = ActionClient(args.host, action_port, connect_timeout=C.DEFAULT_CONNECT_TIMEOUT,
                           retry_timeout=10.0).connect(required=False)
    policy = ReactiveRandomPolicy(random.Random(args.seed), strafe_prob=args.strafe_prob)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path)

    summary = []
    try:
        with Recorder(chan, actions, policy, out_path, args.step_dt, args.max_steps,
                      args.stall_timeout, args.reset_timeout) as rec:
            for ep in range(args.episodes):
                t0 = time.monotonic()
                reason = rec.record_episode(ep)
                dt = time.monotonic() - t0
                summary.append(reason)
                print(f"  episode {ep + 1}/{args.episodes}: reason={reason} ({dt:.1f}s)", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted; flushing what we have.", flush=True)
    finally:
        actions.close()
        chan.close()

    deaths = summary.count("dead")
    print(f"\ndone: {len(summary)} episode(s) -> {out_path}  "
          f"(dead={deaths}, reset={summary.count('reset')}, timeout={summary.count('timeout')})", flush=True)


if __name__ == "__main__":
    main()
