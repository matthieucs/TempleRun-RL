#!/usr/bin/env python3
"""Self-checks for the action-overlay pipeline. Needs no phone.

    python scripts/device/test_overlay_video.py

Covers the two things that can silently produce a plausible-but-wrong video: the video/capture
alignment, and the rule that decides which cell is lit on a given frame. Also runs the whole
pipeline end to end against a synthetic recording whose scene cuts are at known times.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from overlay_rail import Rail                                   # noqa: E402
from overlay_video import (Align, align, lit_per_frame,         # noqa: E402
                           low_confidence, probe)

FAILURES: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


# ---------------------------------------------------------------- alignment
def test_align_offset() -> None:
    print("alignment: constant offset, noisy anchors, spurious peaks")
    resets = [1000.0, 1060.0, 1125.0, 1190.0]
    true_off = 3.70
    peaks = [r - resets[0] + true_off + n for r, n in
             zip(resets, (0.03, -0.05, 0.02, -0.01))]
    peaks += [1.0, 17.3, 40.0, 44.9, 88.0, 150.0, 205.0]        # decoys
    al = align(resets, sorted(peaks), tol=0.6)
    check(al is not None, "an alignment is found")
    assert al is not None
    check(al.matched == 4, f"all four episodes matched (got {al.matched})")
    check(abs(al.b - true_off) < 0.06, f"offset recovered ({al.b:.3f} vs {true_off})")
    err = max(abs(al.to_video(r) - (r - resets[0] + true_off)) for r in resets)
    check(err < 0.08, f"round trip within 80 ms (max {err * 1000:.0f} ms)")


def test_align_drift() -> None:
    print("alignment: clock drift over a long recording")
    resets = [2000.0 + 70.0 * i for i in range(6)]
    a_true, b_true = 1.0006, 5.2
    peaks = [a_true * (r - resets[0]) + b_true for r in resets]
    al = align(resets, sorted(peaks + [3.0, 61.0, 300.0]), tol=0.6)
    assert al is not None
    check("drift" in al.how, f"drift term used (how={al.how!r})")
    check(abs(al.a - a_true) < 1e-4, f"slope recovered ({al.a:.6f} vs {a_true})")
    worst = max(abs(al.to_video(r) - (a_true * (r - resets[0]) + b_true)) for r in resets)
    check(worst < 0.02, f"long-recording error under 20 ms (max {worst * 1000:.0f} ms)")


def test_align_inconsistent_peaks() -> None:
    """Peaks that no single offset can explain must NOT yield a confident answer.

    This is the failure that matters in practice: if the visual transitions do not correspond to
    the episode starts at all, the search will still return *something* (its best single-anchor
    guess). What must hold is that it reports a low match count, so the caller can warn instead of
    shipping a silently misaligned video.
    """
    print("alignment: inconsistent peaks yield low confidence, not a confident wrong answer")
    resets = [3000.0 + 60.0 * i for i in range(4)]
    peaks = [1.5 * (r - resets[0]) + 2.0 for r in resets]       # 50 % drift = impossible
    al = align(resets, sorted(peaks), tol=0.6)
    assert al is not None
    check(abs(al.a - 1.0) < 1e-9, f"slope stays 1.0 rather than fitting nonsense (got {al.a})")
    check(al.matched < len(resets),
          f"not all episodes claimed as matched ({al.matched}/{len(resets)})")
    check(low_confidence(al, len(resets)),
          f"flagged low confidence, so the caller warns ({al.matched}/{len(resets)} matched)")
    good = Align(t0=0.0, a=1.0, b=1.0, matched=4, how="offset")
    check(not low_confidence(good, 4), "a fully matched alignment is not flagged")
    check(low_confidence(Align(0.0, 1.0, 1.0, 1, "offset"), 3), "a single anchor is flagged")


def test_death_validation() -> None:
    """The independent check must actually catch a bad fit -- that is its whole job."""
    print("validation: death cards catch a misaligned fit")
    from overlay_video import Episode, death_residuals
    eps = [Episode(0, 1000.0, 1100.0, 3094.0),
           Episode(1, 1120.0, 1230.0, 2915.0),
           Episode(2, 1250.0, 1390.0, 4294.0)]
    good = Align(t0=1000.0, a=1.0, b=5.0, matched=3, how="test")
    cards = [good.to_video(e.t_last) + 0.10 for e in eps]      # card follows the last action
    r = death_residuals(good, eps, cards)
    check(len(r) == 3 and max(abs(x) for x in r) < 0.15,
          f"a correct alignment shows near-zero residuals ({[round(x, 2) for x in r]})")

    bad = Align(t0=1000.0, a=1.0, b=5.0 - 1.10, matched=3, how="test")   # the real bug, 1.1 s out
    r2 = death_residuals(bad, eps, cards)
    check(len(r2) == 3 and min(x for x in r2) > 0.9,
          f"a 1.1 s error is exposed, not hidden ({[round(x, 2) for x in r2]})")
    check(max(abs(x) for x in r2) > 0.4,
          "the error exceeds the warning threshold, so the user is told")

    # the false alarm this guard was written for: a run ending at the very end of the recording
    # has no card to find, and an unrelated distant peak must not be matched to it
    far = [good.to_video(eps[0].t_last) + 0.10, good.to_video(eps[1].t_last) + 0.10, 300.0]
    dur = good.to_video(eps[2].t_last) + 1.0          # recording stops 1 s after the last death
    r3 = death_residuals(good, eps, far, duration=dur)
    check(len(r3) == 2, f"the run cut off by the end of the recording is skipped ({len(r3)} kept)")
    check(max(abs(x) for x in r3) < 0.15, f"and no bogus residual is reported ({r3})")
    r4 = death_residuals(good, eps[:1], [good.to_video(eps[0].t_last) + 4.0], duration=9e9)
    check(r4 == [], "a peak 4 s away is not a death card and is ignored")


def test_align_no_anchors() -> None:
    print("alignment: gives up rather than inventing an answer")
    check(align([], [1.0, 2.0], 0.6) is None, "no episodes -> None")
    check(align([1.0], [], 0.6) is None, "no peaks -> None")


# ---------------------------------------------------------------- lit rule
def test_lit_rule() -> None:
    print("lit rule: taps stretch, holds sustain, no-op stays dark")
    al = Align(t0=0.0, a=1.0, b=0.0, matched=0, how="test")
    fps, min_lit = 60.0, 0.28

    # a single JUMP tap at t=1.0, surrounded by no-ops
    ev = [(0.0, 6), (1.0, 2), (1.1, 6), (1.2, 6), (2.0, 6)]
    lit = lit_per_frame(ev, int(3 * fps), fps, al, min_lit)
    check(lit[int(0.5 * fps)] is None, "dark before the tap")
    check(lit[int(1.05 * fps)] == "JP", "lit just after the tap")
    check(lit[int(1.25 * fps)] == "JP", "still lit at 250 ms (a 100 ms action would be invisible)")
    check(lit[int(1.35 * fps)] is None, "dark again after 280 ms")

    # a held tilt: repeated at 10 Hz for 1 s, which must read as continuous
    ev = [(round(0.5 + 0.1 * i, 2), 0) for i in range(10)] + [(1.6, 6)]
    lit = lit_per_frame(ev, int(3 * fps), fps, al, min_lit)
    span = [lit[i] for i in range(int(0.55 * fps), int(1.35 * fps))]
    check(all(x == "SL" for x in span), "held tilt is continuously lit, never flickers")
    check(lit[int(1.75 * fps)] is None, "hold decays 280 ms after the last press")

    # the most recent action wins even inside a previous action's window
    ev = [(1.0, 2), (1.1, 3)]
    lit = lit_per_frame(ev, int(2 * fps), fps, al, min_lit)
    check(lit[int(1.15 * fps)] == "SD", "a newer action overrides the older one's tail")

    # turn bytes map to the outer cells
    ev = [(1.0, 4), (1.5, 5)]
    lit = lit_per_frame(ev, int(2 * fps), fps, al, min_lit)
    check(lit[int(1.05 * fps)] == "TL" and lit[int(1.55 * fps)] == "TR", "turn bytes map correctly")


# ---------------------------------------------------------------- geometry
def test_geometry() -> None:
    print("geometry: the rail fits inside the frame at any aspect ratio")
    for w, h, label in ((1080, 2340, "9:19.5 this phone"), (1080, 1920, "9:16"),
                        (720, 1560, "downscaled"), (1920, 1080, "landscape"),
                        (1080, 1440, "3:4")):
        r = Rail(w, h)
        ok = (0 <= r.strip_y and r.strip_y + r.strip_h <= h
              and r.states[None].size == (w, r.strip_h)
              and all(im.size == (w, r.strip_h) for im in r.states.values()))
        check(ok, f"{label} ({w}x{h}) -> strip {w}x{r.strip_h} at y={r.strip_y}")
    check(len(Rail(540, 1170).states) == 7, "seven rail states are pre-rendered")


# ---------------------------------------------------------------- end to end
def build_scene(tmp: Path, seg: float = 4.0, n_pairs: int = 3) -> tuple[Path, list[float]]:
    """A synthetic recording that alternates two very different scenes.

    Each switch to the 'gameplay' scene is a huge frame difference, standing in for the death card
    being replaced by a run starting. Returns the video and the true switch times.
    """
    ffmpeg = shutil.which("ffmpeg")
    quiet = tmp / "quiet.mp4"
    busy = tmp / "busy.mp4"
    subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "lavfi",
                    "-i", f"color=c=0x0d0d0d:s=360x780:r=30", "-t", str(seg),
                    "-pix_fmt", "yuv420p", str(quiet)], check=True)
    subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "lavfi",
                    "-i", f"testsrc=s=360x780:r=30", "-t", str(seg),
                    "-pix_fmt", "yuv420p", str(busy)], check=True)

    lst = tmp / "list.txt"
    parts, switches = [], []
    for i in range(n_pairs):
        parts += [quiet, busy]
        switches.append((2 * i + 1) * seg)          # 'gameplay' starts here
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    scene = tmp / "scene.mp4"
    subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", str(scene)], check=True)
    return scene, switches


def write_capture(path: Path, resets: list[float], marks: list[tuple[int, float, int]]) -> None:
    """marks: (episode, seconds after that episode's reset, action byte)."""
    recs = []
    for ep, t_reset in enumerate(resets):
        by_t = {round(t, 2): b for e, t, b in marks if e == ep}
        for step in range(60):                       # 6 s at 10 Hz
            t = round(step * 0.1, 2)
            recs.append({"ep": ep, "step": step, "action": by_t.get(t, 6),
                         "t_host": t_reset + t, "t_reset": t_reset,
                         "distance": step * 7, "is_dead": False})
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def cell_centre_x(rail: Rail, index: int) -> float:
    from overlay_rail import CELL_FIRST_FRAC, CELL_PITCH_FRAC
    x0 = (rail.frame_w - rail.rail_w) / 2.0
    return x0 + rail.rail_w * (CELL_FIRST_FRAC + index * CELL_PITCH_FRAC)


def orange_centroid(video: Path, t: float, rail: Rail) -> tuple[int, float]:
    """How many ring-orange pixels the finished frame has inside the rail strip, and their mean x.

    Reading the ring back out of the encoded video tests alignment, rendering and compositing
    together -- a mistake in any one of them moves or removes it.
    """
    import io

    import numpy as np
    from PIL import Image

    r = subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-ss", f"{t:.3f}",
                        "-i", str(video), "-frames:v", "1",
                        "-f", "image2pipe", "-vcodec", "png", "pipe:1"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return 0, -1.0
    im = np.asarray(Image.open(io.BytesIO(r.stdout)).convert("RGB"), dtype=np.int16)
    strip = im[rail.strip_y:rail.strip_y + rail.strip_h]
    R, G, B = strip[..., 0], strip[..., 1], strip[..., 2]
    mask = (R > 170) & (G > 70) & (G < 200) & (B < 110) & (R - B > 90)
    n = int(mask.sum())
    if n == 0:
        return 0, -1.0
    return n, float(np.argwhere(mask)[:, 1].mean())


def test_end_to_end() -> None:
    print("end to end: synthetic recording -> aligned overlay")
    if not shutil.which("ffmpeg"):
        check(False, "ffmpeg available")
        return
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        seg = 4.0
        scene, switches = build_scene(tmp, seg=seg)
        info = probe(scene)
        check(abs(info["duration"] - 6 * seg) < 0.5,
              f"scene is {6 * seg:.0f}s (got {info['duration']:.1f}s)")

        # episodes start, in host time, exactly when the busy scene starts in the video
        host0 = 1_700_000_000.0
        resets = [host0 + (s - switches[0]) for s in switches]
        cap = tmp / "run.jsonl"
        # one JUMP 1.0 s into episode 0, one SLIDE 2.0 s into episode 1
        write_capture(cap, resets, [(0, 1.0, 2), (1, 2.0, 3)])

        out = tmp / "prev"
        r = subprocess.run([sys.executable, str(HERE / "overlay_video.py"),
                            "--video", str(scene), "--actions", str(cap),
                            "--out", str(out), "--preview", "4", "--fps", "30"],
                           capture_output=True, text=True)
        check(r.returncode == 0, "preview mode runs clean", r.stderr.strip()[-600:])
        check(len(list(out.glob("*.png"))) == 4, "four preview frames written")
        recovered = [ln for ln in r.stdout.splitlines() if "align[" in ln]
        check(bool(recovered), "an alignment line is reported", r.stdout)
        if recovered:
            line = recovered[0].strip()
            print(f"        {line}")
            check("matched=3" in line, f"all three episode starts matched: {line}")
            # switches[0] is where episode 0 begins in the video
            check(abs(float(line.split("offset=")[1].split("s")[0]) - switches[0]) < 0.15,
                  f"offset lands on the first scene cut ({switches[0]:.1f}s): {line}")

        # and the composited video actually encodes
        vid = tmp / "out.mp4"
        r2 = subprocess.run([sys.executable, str(HERE / "overlay_video.py"),
                             "--video", str(scene), "--actions", str(cap),
                             "--out", str(vid), "--fps", "30", "--crf", "30",
                             "--no-split", "--encoder", "x264"],
                            capture_output=True, text=True)
        check(r2.returncode == 0 and vid.exists(), "composited mp4 is produced",
              r2.stderr.strip()[-600:])
        if not vid.exists():
            return
        oi = probe(vid)
        check(oi["width"] == 360 and oi["height"] == 780,
              f"output keeps the source resolution ({oi['width']}x{oi['height']})")

        # The strongest check available without a phone: the ring must be lit on the RIGHT CELL at
        # the RIGHT TIME in the finished video. Episode 0 starts at video t=4.0 and the capture
        # fires JUMP 1.0 s in, so t=5.0 must show the ring on cell index 2 and t=4.4 must show none.
        rail = Rail(360, 780)
        n_hit, x_hit = orange_centroid(vid, 5.0, rail)
        n_off, _ = orange_centroid(vid, 4.4, rail)
        want_x = cell_centre_x(rail, 2)
        check(n_hit > 40, f"ring is lit at t=5.0s ({n_hit} orange pixels)")
        check(n_off < 10, f"nothing lit at t=4.4s, before the jump ({n_off} orange pixels)")
        check(abs(x_hit - want_x) < 12,
              f"ring sits on the jump cell (x={x_hit:.0f}, expected {want_x:.0f})")

        # ...and the slide 2.0 s into episode 1, which starts at video t=12.0
        n2, x2 = orange_centroid(vid, 14.0, rail)
        check(n2 > 40 and abs(x2 - cell_centre_x(rail, 3)) < 12,
              f"slide lights cell 3 at t=14.0s (n={n2}, x={x2:.0f}, "
              f"expected {cell_centre_x(rail, 3):.0f})")

        # ---- one clip per run -------------------------------------------
        split_out = tmp / "clips" / "run.mp4"
        split_out.parent.mkdir()
        r3 = subprocess.run([sys.executable, str(HERE / "overlay_video.py"),
                             "--video", str(scene), "--actions", str(cap),
                             "--out", str(split_out), "--fps", "30", "--crf", "30",
                             "--encoder", "x264", "--pre-roll", "0.5", "--post-roll", "1.0"],
                            capture_output=True, text=True)
        check(r3.returncode == 0, "per-run splitting runs clean", r3.stderr.strip()[-600:])
        clips = sorted(split_out.parent.glob("run_ep*.mp4"))
        check(len(clips) == 3, f"one clip per run ({len(clips)} written)")
        check(all("m.mp4" in c.name for c in clips),
              f"clips are named with the distance reached ({[c.name for c in clips]})")
        if len(clips) == 3:
            # episode 0 covers host [reset, reset+5.9s]; with 0.5 pre and 1.0 post that is ~7.4 s
            d0 = probe(clips[0])["duration"]
            check(6.5 < d0 < 8.5, f"clip 0 spans the run plus its roll-in/out ({d0:.1f}s)")
            check(all(probe(c)["width"] == 360 for c in clips), "clips keep the source resolution")


# ---------------------------------------------------------------- main
def main() -> None:
    for fn in (test_align_offset, test_align_drift, test_align_inconsistent_peaks,
               test_death_validation, test_align_no_anchors, test_lit_rule,
               test_geometry, test_end_to_end):
        fn()
        print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
