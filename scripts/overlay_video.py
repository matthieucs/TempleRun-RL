#!/usr/bin/env python3
"""Burn the action rail onto a scrcpy screen recording, and cut it into one clip per run.

    python scripts/overlay_video.py --video run.mp4 --actions run.jsonl

By default this writes ONE VIDEO PER EPISODE, named with the distance reached:

    run_ep00_1730m.mp4   run_ep01_7799m.mp4   run_ep02_15002m.mp4

Add --full for the whole recording as a single file, or --no-split for only that.

Offline and re-runnable: phone time is expensive and unrepeatable, rendering is cheap. You never
re-run the phone to adjust a colour.

THE HARD PART IS ALIGNMENT. `capture_calib.py` logs `t_host` (wall clock, at the moment the action
byte goes out) but the video has its own timeline, and scrcpy's start is not synchronous with the
policy process. The refinement that makes this exact: we CONTROL the game -- `reset()` sends byte 8
then 9, and the instant a run starts the death card is replaced by gameplay, an enormous
frame-to-frame difference. So every episode start is simultaneously an exact host timestamp and an
unmistakable visual event. Matching those two sequences yields offset AND clock drift.

Self-checks that need no phone:
    --render-rail sheet.png          the seven rail states, to eyeball the icons
    --preview 8                      sample frames with the overlay, to eyeball alignment
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlay_rail import BYTE_TO_KEY, Rail  # noqa: E402

NOOP_BYTE = 6


class Episode(NamedTuple):
    index: int
    t_reset: float      # host time of reset(), i.e. when the run starts
    t_last: float       # host time of the final action, i.e. the death
    distance: float


# ---------------------------------------------------------------- external tools
def _tool(name: str) -> str:
    env = os.environ.get(name.upper())
    if env and Path(env).exists():
        return env
    found = shutil.which(name)
    if not found:
        sys.exit(f"error: {name} not found on PATH (set the {name.upper()} environment variable "
                 f"to its full path)")
    return found


def probe(video: Path) -> dict:
    out = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"error: ffprobe failed on {video}\n{out.stderr.strip()}")
    d = json.loads(out.stdout)
    st = d["streams"][0]
    num, _, den = st["avg_frame_rate"].partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {"width": int(st["width"]), "height": int(st["height"]),
            "duration": float(d["format"]["duration"]), "fps": fps}


_NVENC: bool | None = None


def nvenc_works(ffmpeg: str) -> bool:
    """Actually try it. Being listed in -encoders says nothing about the driver working."""
    global _NVENC
    if _NVENC is None:
        try:
            r = subprocess.run([ffmpeg, "-hide_banner", "-v", "error", "-f", "lavfi",
                                "-i", "testsrc=s=256x256:r=30", "-t", "0.2",
                                "-c:v", "h264_nvenc", "-f", "null", "-"],
                               capture_output=True, timeout=90)
            _NVENC = r.returncode == 0
        except Exception:                                        # noqa: BLE001
            _NVENC = False
    return _NVENC


# ---------------------------------------------------------------- capture
def load_actions(path: Path) -> tuple[list[tuple[float, int]], list[Episode], list[float], bool]:
    """-> (actions, episodes, host times at which the game's `facing` changed, has-real-clock).

    The facing changes are the sync anchors that matter. A run produces ~35 of them where it
    produces one reset, and each is a 90-degree camera rotation -- the least ambiguous event this
    game can give us in both streams at once.
    """
    events: list[tuple[float, int]] = []
    per_ep: dict[int, dict] = {}
    turns: list[float] = []
    prev_facing = None
    synthetic = False
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            act = int(r.get("action", NOOP_BYTE))
            t = r.get("t_host")
            if t is None:
                # pre-2026-07-28 capture: no clock at all, only a nominal-10 Hz step counter
                synthetic = True
                t = float(r.get("step", 0)) / 10.0
            t = float(t)
            events.append((t, act))
            fac = r.get("facing")
            if fac is not None:
                if prev_facing is not None and float(fac) != prev_facing:
                    turns.append(t)
                prev_facing = float(fac)
            ep = int(r.get("ep", 0))
            e = per_ep.setdefault(ep, {"t_reset": float(r.get("t_reset", t)),
                                       "t_last": t, "distance": 0.0})
            e["t_last"] = max(e["t_last"], t)
            try:
                e["distance"] = max(e["distance"], float(r.get("distance") or 0.0))
            except (TypeError, ValueError):
                pass
    events.sort(key=lambda e: e[0])
    eps = [Episode(i, per_ep[i]["t_reset"], per_ep[i]["t_last"], per_ep[i]["distance"])
           for i in sorted(per_ep)]
    return events, eps, turns, not synthetic


# ---------------------------------------------------------------- visual transitions
def frame_diff_signal(video: Path, sample_fps: float, width: int,
                      src_w: int, src_h: int) -> tuple[np.ndarray, np.ndarray]:
    h = max(2, int(round(width * src_h / src_w / 2)) * 2)
    cmd = [_tool("ffmpeg"), "-v", "error", "-i", str(video),
           "-vf", f"fps={sample_fps},scale={width}:{h}", "-pix_fmt", "gray",
           "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nbytes = width * h
    prev = None
    diffs: list[float] = []
    assert proc.stdout is not None
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        cur = np.frombuffer(buf, dtype=np.uint8).astype(np.int16)
        if prev is not None:
            diffs.append(float(np.abs(cur - prev).mean()))
        prev = cur
    proc.stdout.close()
    proc.wait()
    d = np.asarray(diffs, dtype=np.float64)
    t = (np.arange(len(d)) + 1) / sample_fps
    return t, d


def find_peaks(t: np.ndarray, d: np.ndarray, max_peaks: int = 240) -> list[float]:
    if len(d) < 3:
        return []
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) or 1e-6
    thresh = med + 5.0 * 1.4826 * mad
    idx = [i for i in range(1, len(d) - 1)
           if d[i] > thresh and d[i] >= d[i - 1] and d[i] >= d[i + 1]]
    idx.sort(key=lambda i: -d[i])
    idx = sorted(idx[:max_peaks])
    return [float(t[i]) for i in idx]


# ---------------------------------------------------------------- alignment
class Align:
    """t_video = a * (t_host - t0) + b."""

    def __init__(self, t0: float, a: float, b: float, matched: int, how: str):
        self.t0, self.a, self.b, self.matched, self.how = t0, a, b, matched, how

    def to_video(self, t_host: float) -> float:
        return self.a * (t_host - self.t0) + self.b

    def to_host(self, t_video: float) -> float:
        return (t_video - self.b) / self.a + self.t0

    def __str__(self) -> str:
        return (f"align[{self.how}] matched={self.matched} "
                f"offset={self.b:.3f}s drift={(self.a - 1.0) * 1e6:+.0f} ppm")


def low_confidence(al: "Align", n_resets: int) -> bool:
    """Should the caller warn that this alignment may be wrong?

    Every episode start ought to produce a visual transition, so a shortfall means the anchors and
    the peaks do not really correspond. Two anchors is also the minimum at which they corroborate
    each other at all.
    """
    if al.how == "forced" or n_resets == 0:
        return False
    return al.matched < 2 or al.matched < 0.75 * n_resets


def align(resets: list[float], peaks: list[float], tol: float) -> Align | None:
    if not resets or not peaks:
        return None
    t0 = resets[0]
    R = [r - t0 for r in resets]
    P = sorted(peaks)

    best = (-1, float("inf"), 0.0)
    for r in R:
        for p in P:
            off = p - r
            n, err = 0, 0.0
            for rr in R:
                target = rr + off
                j = bisect.bisect_left(P, target)
                cand = [P[k] for k in (j - 1, j) if 0 <= k < len(P)]
                if not cand:
                    continue
                e = min(abs(c - target) for c in cand)
                if e <= tol:
                    n += 1
                    err += e
            if n > best[0] or (n == best[0] and err < best[1]):
                best = (n, err, off)

    n_match, _, off = best
    if n_match < 1:
        return None

    pairs: list[tuple[float, float]] = []
    for rr in R:
        target = rr + off
        j = bisect.bisect_left(P, target)
        cand = [P[k] for k in (j - 1, j) if 0 <= k < len(P)]
        if cand:
            c = min(cand, key=lambda x: abs(x - target))
            if abs(c - target) <= tol:
                pairs.append((rr, c))

    # Fit the slope only when it is actually identifiable: at least three anchors spread over a
    # meaningful span. Otherwise a tiny timing error at two nearby points becomes a huge fake drift.
    if len(pairs) >= 3 and (pairs[-1][0] - pairs[0][0]) > 30.0:
        xs = np.array([p[0] for p in pairs])
        ys = np.array([p[1] for p in pairs])
        a, b = np.polyfit(xs, ys, 1)
        if abs(a - 1.0) > 0.02:
            a, b = 1.0, float(np.mean(ys - xs))
            how = "offset (drift rejected)"
        else:
            how = "offset+drift"
        return Align(t0, float(a), float(b), len(pairs), how)

    b = float(np.mean([p[1] - p[0] for p in pairs]))
    return Align(t0, 1.0, b, len(pairs), "offset")


def refine_with_turns(al: Align, turn_hosts: list[float], t: np.ndarray, d: np.ndarray,
                      search: float = 3.0) -> tuple[Align, str]:
    """Correct the residual sync error by cross-correlating turns against the frame difference.

    WHY THIS EXISTS. Anchoring on episode starts is only as good as the semantics of the anchor,
    and those turned out to be shaky in both directions: `t_reset` is stamped after
    `_restart_and_wait()` has waited for the run to be *progressing*, while the frame-difference
    peak lands somewhere inside the start-of-run camera animation rather than at its onset. Neither
    error is knowable a priori, and with three episodes there is no way to detect them.

    A turn is immune to both. The game's own `facing` field flips, and the whole scene rotates 90
    degrees -- a huge, sharp, unmistakable event, and a six-minute capture has ~100 of them instead
    of 3. Measured on the first real recording, the reset-anchored fit was 1.1 s out; the two halves
    of the recording independently agreed on 1.10 s and 1.10 s.

    Returns the corrected alignment and a human-readable note. Refuses to apply a correction the
    two halves disagree about, because a confident wrong shift is worse than none.
    """
    if len(turn_hosts) < 20 or len(d) < 120:
        return al, f"not enough anchors to refine ({len(turn_hosts)} turns)"

    step = float(t[1] - t[0]) if len(t) > 1 else 0.1
    rate = 1.0 / step

    def impulses(a: Align) -> np.ndarray:
        s = np.zeros_like(d)
        for x in turn_hosts:
            i = int(round(a.to_video(x) * rate))
            if 0 <= i < len(s):
                s[i] = 1.0
        k = np.hanning(7)
        return np.convolve(s, k / k.sum(), mode="same")

    vid = d - np.convolve(d, np.ones(31) / 31, mode="same")     # high-pass
    dev = impulses(al)
    lags = np.arange(-search, search + 1e-9, step)

    def best_over(sl: slice) -> tuple[float, float]:
        out = (0.0, -9.0)
        for lag in lags:
            x = np.roll(dev, int(round(lag * rate)))
            v, xx = vid[sl], x[sl]
            if v.std() < 1e-9 or xx.std() < 1e-9:
                continue
            c = float(np.corrcoef(v, xx)[0, 1])
            if c > out[1]:
                out = (float(lag), c)
        return out

    m = len(vid) // 2
    lag, corr = best_over(slice(40, -40))
    lag_a, _ = best_over(slice(40, m))
    lag_b, _ = best_over(slice(m, -40))
    if abs(lag_a - lag_b) > 0.3:
        return al, (f"turn refinement REJECTED: the two halves disagree "
                    f"({lag_a:+.2f}s vs {lag_b:+.2f}s) -- leaving the reset-anchored fit alone")
    if abs(lag) < step / 2:
        return al, f"turn refinement: already aligned ({len(turn_hosts)} turns, corr {corr:.2f})"
    out = Align(al.t0, al.a, al.b + lag, al.matched, al.how + "+turns")
    return out, (f"turn refinement: {lag:+.2f}s from {len(turn_hosts)} turns "
                 f"(corr {corr:.2f}, halves {lag_a:+.2f}/{lag_b:+.2f})")


def death_residuals(al: Align, episodes: list[Episode], peaks: list[float],
                    duration: float | None = None, window: float = 2.5,
                    edge: float = 5.0) -> list[float]:
    """How far each death card sits from where we predict the run ended.

    An INDEPENDENT check: deaths are never used as anchors, so this cannot flatter a bad fit. The
    card follows the last logged action almost immediately, so a correct alignment puts these near
    zero. On the first real recording they were +1.17/+1.26 s before the turn refinement and
    +0.07/+0.16 s after it.

    Two guards, both learned from a false alarm on that same recording, where the last run ended
    383 s into a 384 s file: its card was never recorded, and an unrelated peak 6 s away got matched
    instead, reporting a 6 s error that did not exist.
      * `window` -- a candidate further than this is not a death card, so report nothing.
      * `edge`   -- a run ending this close to the end of the recording has no card to find.
    """
    out = []
    for e in episodes:
        v = al.to_video(e.t_last)
        if duration is not None and v > duration - edge:
            continue
        near = [p for p in peaks if abs(p - v) < window]
        if near:
            out.append(min(near, key=lambda p: abs(p - v)) - v)
    return out


def synchronise(video: Path, info: dict, episodes: list[Episode], turn_hosts: list[float],
                events: list[tuple[float, int]], *, tol: float, auto_sync: bool,
                bias_ms: float, offset_ms: float | None,
                no_auto_align: bool, has_clock: bool) -> Align:
    """Work out the video<->capture mapping, and say plainly how much to trust it.

    Three layers, because no single one is dependable on its own:
      1. COARSE   episode starts matched to visual transitions. Good to a second or two, and its
                  anchor semantics are shaky at both ends (see the spec).
      2. REFINE   cross-correlate ~100 turns against the frame difference. This is what makes it
                  right; split halves must agree or it is discarded.
      3. VALIDATE death cards, which layer 2 never touched. Purely a check on the other two.
    """
    resets = [e.t_reset for e in episodes]
    if offset_ms is not None:
        al = Align(resets[0] if resets else events[0][0], 1.0, offset_ms / 1000.0, 0, "forced")
        print(f"sync    {al}  (forced, no checks run)")
        return al

    diff_t = diff_d = None
    peaks: list[float] = []
    al: Align | None = None
    if not no_auto_align and has_clock and resets:
        diff_t, diff_d = frame_diff_signal(video, 10.0, 160, info["width"], info["height"])
        peaks = find_peaks(diff_t, diff_d)
        al = align(resets, peaks, tol)

    if al is None:
        meta = video.with_suffix(".meta.json")
        if meta.exists():
            al = Align(float(json.loads(meta.read_text())["t_rec_start"]), 1.0, 0.0, 0,
                       "recorder start")
        else:
            al = Align(events[0][0], 1.0, 0.0, 0, "first action (GUESS)")
        print(f"sync    coarse stage FAILED, using {al.how}")
    else:
        print(f"sync    coarse: {al.matched}/{len(resets)} starts matched, offset {al.b:.2f}s")

    if auto_sync and turn_hosts and diff_d is not None:
        al, note = refine_with_turns(al, turn_hosts, diff_t, diff_d)
        print(f"        {note}")

    if bias_ms:
        al = Align(al.t0, al.a, al.b + bias_ms / 1000.0, al.matched, al.how + "+bias")
        print(f"        manual bias {bias_ms:+.0f} ms applied")

    res = death_residuals(al, episodes, peaks, duration=info["duration"])
    if res:
        med = sorted(res)[len(res) // 2]
        detail = ", ".join(f"{r:+.2f}" for r in res)
        if max(abs(r) for r in res) <= 0.4:
            print(f"        CHECK OK: {len(res)} death card(s) land {detail} s from prediction "
                  f"(independent of everything above)")
        else:
            print(f"WARNING: death cards land {detail} s from prediction, so the overlay is "
                  f"probably out by about {med:+.1f}s. Re-run with --sync-bias-ms {med * 1000:+.0f}")
    else:
        print("        no death card was usable for validation; sync is UNVERIFIED")
    print(f"        {al}")
    return al


# ---------------------------------------------------------------- lit-cell sequence
def lit_per_frame(events: list[tuple[float, int]], n_frames: int, fps: float,
                  al: Align, min_lit: float, t0_video: float = 0.0) -> list[str | None]:
    """Which cell is lit on each overlay frame, starting at video time `t0_video`.

    One rule: every non-NoOp action lights its cell for `min_lit`, most recent wins. Sustain falls
    out for free -- control is 10 Hz (100 ms) and min_lit is 280 ms, so a HELD tilt re-lights before
    it can decay and reads as continuous, while an isolated tap decays after 280 ms. A single 100 ms
    action would be 6 frames at 60 fps, far too brief to see; stretching it is deliberate.
    """
    fired = [(t, BYTE_TO_KEY[a]) for t, a in events if a in BYTE_TO_KEY]
    times = [t for t, _ in fired]
    out: list[str | None] = []
    for i in range(n_frames):
        th = al.to_host(t0_video + i / fps)
        j = bisect.bisect_right(times, th) - 1
        out.append(fired[j][1] if j >= 0 and th - times[j] < min_lit else None)
    return out


# ---------------------------------------------------------------- output
def composite(video: Path, out: Path, rail: Rail, lit: list[str | None], *,
              rail_fps: float, scale: float, crf: int, encoder: str,
              t_start: float = 0.0, duration: float | None = None,
              gpu_cq: int = 27, gpu_maxrate: str = "12M", quiet: bool = False) -> None:
    """Overlay `lit` onto [t_start, t_start+duration] of `video`.

    The overlay is fed at `rail_fps`, NOT the video's frame rate. ffmpeg's overlay filter holds the
    most recent overlay frame, and the rail can only change at the 10 Hz control rate, so feeding it
    60 times a second just pushes three times the bytes through the pipe for an identical result.
    At 1080x253 RGBA that is the difference between ~65 MB/s and ~22 MB/s.
    """
    W, SH = rail.frame_w, rail.strip_h
    raw = {k: img.tobytes() for k, img in rail.states.items()}
    ffmpeg = _tool("ffmpeg")

    if encoder == "auto":
        encoder = "nvenc" if nvenc_works(ffmpeg) else "x264"
    if encoder == "nvenc":
        # NVENC is a dedicated ASIC on the card, NOT the CUDA cores, so this offloads the encode
        # without competing with training. Its -cq scale is NOT x264's -crf: passing crf 18 through
        # produced a 2.6 GB file where x264 produced 755 MB, hence a separate quality knob and a
        # hard bitrate ceiling.
        vcodec = ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
                  "-rc", "vbr", "-cq", str(gpu_cq), "-b:v", "0",
                  "-maxrate", gpu_maxrate, "-bufsize", "24M"]
    else:
        vcodec = ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]

    chain = (f"[0:v]fps={rail_fps * 3},format=rgba[bg];"
             f"[bg][1:v]overlay=0:{rail.strip_y}:eof_action=repeat[ov]")
    if abs(scale - 1.0) > 1e-6:
        ow = int(round(rail.frame_w * scale)) // 2 * 2
        oh = int(round(rail.frame_h * scale)) // 2 * 2
        chain += f";[ov]scale={ow}:{oh}[o]"
    else:
        chain += ";[ov]null[o]"

    cmd = [ffmpeg, "-y", "-v", "error"]
    if not quiet:
        cmd += ["-stats"]
    if t_start > 0:
        cmd += ["-ss", f"{t_start:.3f}"]
    cmd += ["-i", str(video),
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{SH}",
            "-r", str(rail_fps), "-i", "pipe:0",
            "-filter_complex", chain, "-map", "[o]", "-an"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [*vcodec, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for key in lit:
            proc.stdin.write(raw[key])
    except BrokenPipeError:
        pass    # ffmpeg reached the end of the window first; harmless
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
    rc = proc.wait()
    if rc != 0:
        sys.exit(f"error: ffmpeg exited {rc}")


def write_previews(video: Path, rail: Rail, al: Align, events, min_lit: float,
                   n: int, duration: float, out_dir: Path) -> None:
    import io

    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        ts = duration * (k + 0.5) / n
        key = lit_per_frame(events, 1, 1.0, al, min_lit, t0_video=ts)[0]
        r = subprocess.run([_tool("ffmpeg"), "-v", "error", "-ss", f"{ts:.3f}", "-i", str(video),
                            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
                           capture_output=True)
        if r.returncode != 0 or not r.stdout:
            print(f"  preview {k}: could not extract a frame at {ts:.2f}s")
            continue
        frame = Image.open(io.BytesIO(r.stdout)).convert("RGBA")
        frame.alpha_composite(rail.frame(key), (0, rail.strip_y))
        p = out_dir / f"preview_{k:02d}_t{ts:07.2f}s_{key or 'none'}.png"
        frame.convert("RGB").save(p)
        print(f"  {p.name}")


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path)
    p.add_argument("--actions", type=Path)
    p.add_argument("--out", type=Path, help="output path (per-run clips are named from its stem)")
    p.add_argument("--fps", type=float, default=60.0, help="output frame rate")
    p.add_argument("--rail-fps", type=float, default=20.0,
                   help="rate the overlay is fed at; the rail only changes at 10 Hz")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--crf", type=int, default=18, help="CPU encoder quality")
    p.add_argument("--gpu-cq", type=int, default=27, help="GPU encoder quality (different scale)")
    p.add_argument("--gpu-maxrate", default="12M", help="GPU encoder bitrate ceiling")
    p.add_argument("--encoder", choices=("auto", "nvenc", "x264"), default="auto",
                   help="auto uses the GPU's NVENC block when it works, else libx264 on the CPU")
    p.add_argument("--min-lit-ms", type=float, default=280.0)
    p.add_argument("--no-split", dest="split", action="store_false",
                   help="do not cut per-run clips (implies --full)")
    p.add_argument("--full", action="store_true", help="also write the whole recording as one file")
    p.add_argument("--pre-roll", type=float, default=1.0, help="seconds kept before a run starts")
    p.add_argument("--post-roll", type=float, default=3.0, help="seconds kept after the death")
    p.add_argument("--rail-width", type=float, default=None)
    p.add_argument("--rail-bottom", type=float, default=None)
    p.add_argument("--offset-ms", type=float, default=None,
                   help="force this video-minus-host offset and skip auto-alignment")
    p.add_argument("--no-auto-align", action="store_true")
    p.add_argument("--align-tol", type=float, default=0.6)
    p.add_argument("--no-auto-sync", dest="auto_sync", action="store_false",
                   help="skip the turn-based sync refinement (see refine_with_turns)")
    p.add_argument("--sync-bias-ms", type=float, default=0.0,
                   help="nudge the rail; POSITIVE moves it LATER in the video")
    p.add_argument("--preview", type=int, default=0, metavar="N",
                   help="write N sample PNGs instead of any video")
    p.add_argument("--render-rail", type=Path, metavar="PNG",
                   help="write the seven rail states as a sheet and exit")
    p.add_argument("--frame-size", default="1080x2340")
    a = p.parse_args()

    rail_kw = {}
    if a.rail_width is not None:
        rail_kw["rail_w_frac"] = a.rail_width
    if a.rail_bottom is not None:
        rail_kw["bottom_frac"] = a.rail_bottom

    if a.render_rail:
        if a.video and a.video.exists():
            info = probe(a.video)
            w, h = info["width"], info["height"]
        else:
            w, h = (int(x) for x in a.frame_size.lower().split("x"))
        Rail(w, h, **rail_kw).sheet().save(a.render_rail)
        print(f"wrote {a.render_rail}  ({w}x{h} frame)")
        return

    if not (a.video and a.actions):
        p.error("--video and --actions are required (or use --render-rail)")
    for f in (a.video, a.actions):
        if not f.exists():
            sys.exit(f"error: no such file: {f}")

    info = probe(a.video)
    print(f"video   {a.video.name}  {info['width']}x{info['height']}  "
          f"{info['duration']:.1f}s  ~{info['fps']:.1f} fps (variable)")

    events, episodes, turn_hosts, has_clock = load_actions(a.actions)
    if not events:
        sys.exit("error: capture contains no action records")
    if not has_clock:
        print("WARNING: capture has no t_host -- falling back to step/10 Hz. This capture predates "
              "the timestamp change, so drift against the video is likely.")
    print(f"actions {len(events)} records, {len(episodes)} episodes, "
          f"{events[-1][0] - events[0][0]:.1f}s of host time")

    al = synchronise(a.video, info, episodes, turn_hosts, events,
                     tol=a.align_tol, auto_sync=a.auto_sync, bias_ms=a.sync_bias_ms,
                     offset_ms=a.offset_ms, no_auto_align=a.no_auto_align, has_clock=has_clock)

    rail = Rail(info["width"], info["height"], **rail_kw)
    print(f"rail    {rail.frame_w}x{rail.strip_h} strip at y={rail.strip_y}")

    if a.preview:
        out_dir = a.out if a.out else a.video.parent / "preview"
        print(f"writing {a.preview} preview frames to {out_dir}")
        write_previews(a.video, rail, al, events, a.min_lit_ms / 1000.0,
                       a.preview, info["duration"], out_dir)
        return

    out = a.out or a.video.with_name(a.video.stem + "_overlay.mp4")
    enc = "nvenc" if (a.encoder == "auto" and nvenc_works(_tool("ffmpeg"))) else \
          ("x264" if a.encoder == "auto" else a.encoder)
    print(f"encode  {enc}" + ("  (GPU, dedicated encoder block)" if enc == "nvenc" else "  (CPU)"))
    common = dict(rail_fps=a.rail_fps, scale=a.scale, crf=a.crf, encoder=enc,
                  gpu_cq=a.gpu_cq, gpu_maxrate=a.gpu_maxrate)

    if a.split and episodes:
        print(f"cutting {len(episodes)} runs")
        for ep in episodes:
            v0 = max(0.0, al.to_video(ep.t_reset) - a.pre_roll)
            v1 = min(info["duration"], al.to_video(ep.t_last) + a.post_roll)
            dur = v1 - v0
            if dur <= 0.5:
                print(f"  ep{ep.index:02d}: window is {dur:.2f}s, skipped "
                      f"(the recording probably does not cover this run)")
                continue
            name = f"{out.stem}_ep{ep.index:02d}_{int(round(ep.distance))}m.mp4"
            clip = out.with_name(name)
            n = int(round(dur * a.rail_fps)) + 4
            lit = lit_per_frame(events, n, a.rail_fps, al, a.min_lit_ms / 1000.0, t0_video=v0)
            print(f"  ep{ep.index:02d}  {ep.distance:>7.0f} m  "
                  f"{v0:7.1f}s -> {v1:7.1f}s  ({dur:5.1f}s)  {name}")
            composite(a.video, clip, rail, lit, t_start=v0, duration=dur, quiet=True, **common)

    if a.full or not a.split:
        n = int(round(info["duration"] * a.rail_fps)) + 20
        lit = lit_per_frame(events, n, a.rail_fps, al, a.min_lit_ms / 1000.0)
        composite(a.video, out, rail, lit, **common)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
