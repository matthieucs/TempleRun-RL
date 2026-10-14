# Action-overlay capture video — design

**Date:** 2026-07-28
**Status:** approved, implementing
**Purpose:** record the phone while the policy plays, and burn an overlay onto the recording that
shows which action the policy is taking.

---

## 1. Goal and non-goals

**Goal.** A shareable video of the policy playing Temple Run, with a bottom rail that lights the
action being taken — jump, slide, tilt left/right, turn left/right.

**Non-goals**, all decided explicitly by the user:

* No distance or score readout. The game draws its own score; a second number competes with it.
* No input history / ribbon.
* No death treatment. The overlay simply stops drawing; the game's death card speaks for itself.
* Not a forensic debugging tool. A dense telemetry view is a separate job with a different design.

---

## 2. Device facts (measured, not assumed)

Read off device `3f80a3a` on 2026-07-28 with the game live in the foreground:

| Fact | Value |
|---|---|
| Resolution | **1080 × 2340** — that is 9:19.5, **not** 9:16 |
| Density | 440 |
| Model / OS | Redmi Note 7 / Android 9 |
| Letterboxing | **None.** The game paints full-bleed, so there is no free black bar to hide an overlay in |
| Game HUD | Own score top-right during a run; `Distance: … m` on the death card |

**The aspect ratio is never hardcoded.** The compositor reads the recording's real dimensions with
`ffprobe` and lays the rail out in fractions of width and height. A different phone, a different
scrcpy `--max-size`, or a rotated capture all work without a code change.

---

## 3. The overlay

Six icons in the game's own control-hint visual language, on one rounded translucent pill, centred
horizontally at the bottom of the frame. A ring lights around the active action. Nothing else.

### Order (mirror-symmetric)

```
turn-left | tilt-left | jump | slide | tilt-right | turn-right
```

Turns outermost, tilts inboard, jump and slide in the middle.

### Icons

Drawn as **vector primitives**, not font glyphs. Two reasons: no font dependency at render time, and
a tilting phone is not a character in any font.

| Cell | Action byte | Shape |
|---|---|---|
| turn-left | 4 `TurnLeft` | hook arrow, tail down-right, head pointing left |
| tilt-left | 0 `StrafeLeft` | rounded rect rotated −20°, two motion arcs on its left |
| jump | 2 `Jump` | straight up arrow |
| slide | 3 `Slide` | straight down arrow |
| tilt-right | 1 `StrafeRight` | mirror of tilt-left |
| turn-right | 5 `TurnRight` | mirror of turn-left |

`6 NoOp` lights nothing. It is the most common action and the rail should look calm.

The tilt icons are why no text labels are needed: a leaning phone cannot be confused with a turn
arrow. They are also honest to the mechanism — a strafe really is an analog accelerometer **hold**
(byte 0/1 held, byte 6 recenters), not a tap.

### Geometry — all fractions, resolved against the real frame

All values below are measured off the user's reference image (1834 px wide, bar 1708 × 305).

| Property | Value | From the reference |
|---|---|---|
| Rail width | `0.78 × frame_width`, centred | — |
| Rail height | `0.179 × rail_width` | 305 / 1708 |
| Bottom margin | `0.034 × frame_height` | — |
| Corner radius | `0.16 × rail_height` | — |
| First icon centre | `0.095 × rail_width` | 163 / 1708 |
| Icon pitch | `0.162 × rail_width` | 276 / 1708 |
| Icon height | `0.377 × rail_height` | 115 / 305 |
| Icon stroke | `0.056 × rail_height` | 17 / 305 |
| Ring diameter | `0.705 × rail_height` | 215 / 305 |

The phone body is stroked at `0.105 × icon_height`, deliberately *thinner* than the arrows'
`0.148`. At the arrow weight the two side strokes very nearly meet across a body that narrow and
the icon reads as a filled blob — the reference keeps about 55 % of the body width as interior.

### Colour

| Element | Value |
|---|---|
| Panel fill | `rgba(18, 18, 18, 0.91)` |
| Panel border | `rgba(255, 255, 255, 0.25)`, scaled |
| Icon | `#FFFFFF` |
| Ring | `#FF961A`, plus a twice-composited blurred bloom |

### Timing

One action at 10 Hz lasts 100 ms, which is **6 frames at 60 fps** — too brief to read. A fired action
therefore stays lit for a minimum of **280 ms**. This is a deliberate divergence from the raw data
and is recorded here so nobody later mistakes it for a bug.

The rule is a single one: *each non-NoOp action lights its cell for `min_lit`; the most recent action
wins.* Sustained behaviour falls out for free — control is 10 Hz (100 ms) and `min_lit` is 280 ms, so
a held tilt re-lights before it can decay and reads as continuous, while an isolated tap decays
after 280 ms.

---

## 4. Recording

One scrcpy instance that **both mirrors and records**. Flags verified against the installed
scrcpy 3.3.4:

```
scrcpy -s <serial> --record run.mp4 --video-bit-rate 16M --max-fps 60 --no-audio
```

> **Operational rule.** Do not run a second scrcpy alongside it. A second instance spawns a second
> encoder on the device — real extra load on a phone this project has measured to be
> thermally fragile. One instance both displays and records, so nothing is lost.

**The recording is variable frame rate.** scrcpy encodes *on screen change*: gameplay produces ~60
frames a second, the death card produces almost none. Overlaying a fixed-rate rail onto a
variable-rate video would let the two drift apart, so the compositor normalises to constant 60 fps
before overlaying.

**Clean shutdown matters.** An mp4 needs its trailer written. On Windows `Popen.terminate()` is a hard
kill and leaves an unplayable file, so scrcpy is started in a new process group and stopped with
`CTRL_BREAK_EVENT`. The result is verified with `ffprobe`; if it fails, the tool says so loudly
rather than handing over a corrupt file.

---

## 5. Aligning the two streams

This is the only genuinely hard part.

`capture_calib.py` currently logs `{ep, step, action, …}` with **no clock at all** — `step` is a
nominal-10 Hz counter, and the control loop drains to the newest stream line, so it jitters and
occasionally stalls. Assuming 10 Hz would drift visibly within a minute.

### Change to the capture

Add two fields per record:

* `t_host` — `time.time()` captured immediately before `env.step(act)`, i.e. when the action byte
  goes out.
* `t_reset` — `time.time()` at the episode's `reset()`, constant for all records of that episode.

Wall clock rather than `perf_counter` because the value must be comparable across processes.

### Automatic alignment

We *control* the game. `reset()` sends byte 8 then byte 9, and the instant a run starts the death
card is replaced by gameplay — an enormous frame-to-frame difference. Every episode start is
therefore simultaneously an exact host timestamp and an unmistakable visual event.

1. Decode the video downscaled to greyscale (~160 px wide, sampled at 10 Hz) and compute the mean
   absolute difference between consecutive samples.
2. Take local maxima above a threshold as candidate transitions `V`.
3. Take the episode reset times `R` from the capture.
4. For every candidate pairing `(r_i, v_j)` propose `offset = v_j − r_i`, and score it by how many of
   `R + offset` land within tolerance of some element of `V`. Keep the best.
5. Refine on the matched pairs by least squares, fitting `t_video = a·t_host + b`.

Fitting `a` as well as `b` corrects **clock drift** across a long recording, not just a constant
offset. This needs no knowledge of when scrcpy started, which makes it robust to launch races.

**Fallbacks.** If fewer than two episodes match, fit `b` only with `a = 1`. If nothing matches, fall
back to the recorder's own start timestamp and warn loudly. `--offset-ms` overrides everything.

### The reset anchor is not good enough on its own — measured

On the first real recording the reset-anchored fit was **1.1 s out**, which is plainly visible.
Both ends of the anchor are unreliable and neither error is knowable in advance:

* `t_reset` is stamped *after* `_restart_and_wait()` has waited for the run to be **progressing**,
  so it is later than the true start by an unknown amount.
* The frame-difference **peak** is the argmax of the start-of-run camera animation, not its onset,
  so it too sits somewhere inside the transition.

With three episodes there is no way to detect either. **A turn is immune to both**: the game's own
`facing` field flips and the entire scene rotates 90°, and a six-minute capture contains ~100 of
them. So a second pass cross-correlates turn times against the high-passed frame-difference signal
and corrects the residual.

Measured on the first recording: **+1.10 s, with the two halves independently agreeing at +1.10 and
+1.10.** Independently confirmed against the death card, which is never used as an anchor — the gap
between the last logged action and the card appearing fell from +1.17/+1.26 s to +0.07/+0.16 s.

The refinement refuses to apply a correction the two halves disagree about by more than 0.3 s: a
confident wrong shift is worse than none. `--sync-bias-ms` nudges manually (positive = later),
`--no-auto-sync` disables it.

> A dead end worth recording: the first attempt tracked the character's hair colour and correlated
> its on-screen height against the device's `world_y`. It reported "+1.00 s" and was **noise** —
> across five windows it gave +2.30, +1.20, −0.70, +1.20, −0.40 with correlations of 0.10–0.15.
> Reproducibility across independent windows is what separated the real measurement from that one.

---

## 6. Rendering and compositing

**The rail has only seven possible appearances** — one per lit cell, plus nothing lit. So the
renderer draws seven RGBA strips once and then reuses them. Per-frame work becomes a buffer
selection and a pipe write.

Drawing is done at **4× supersample and downscaled with Lanczos**, which gives clean antialiasing on
the ring, the arrowheads and the rotated phone without any special-casing. Cost is irrelevant for
seven images.

Only the **bottom strip** is composited — the rail's bounding box plus a small pad for the glow. The
rest of the frame is copied untouched. This matters: piping full-frame RGBA at 1080 × 2340 × 60 fps
would be ~600 MB/s, whereas the strip is ~30 MB/s.

```
ffmpeg -i run.mp4 -f rawvideo -pix_fmt rgba -s WxSTRIP -r 60 -i -
       -filter_complex "[0:v]fps=60[bg];[bg][1:v]overlay=0:Y:eof_action=repeat[out]"
       -map "[out]" -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4
```

---

## 7. Files

| File | Status | Job |
|---|---|---|
| `scripts/device/capture_calib.py` | +20 lines | log `t_host` and `t_reset` |
| `scripts/device/overlay_rail.py` | new | draws the rail: icon vectors, geometry, the seven states |
| `scripts/device/overlay_video.py` | new | offline compositor: probe, align, render, composite |
| `scripts/device/record_overlay.py` | new | orchestrator: start scrcpy, run the policy, stop cleanly, render |
| `scripts/device/test_overlay_video.py` | new | self-checks, all phone-free |

The split is deliberate: **phone time is expensive and unrepeatable, rendering is cheap and
repeatable.** You never re-run the phone to adjust a colour.

### `overlay_video.py`

```
--video PATH --actions PATH --out PATH
--fps 60 --scale 1.0 --min-lit-ms 280
--rail-width 0.78 --rail-bottom 0.034 --rail-height-ratio 0.12
--offset-ms N          bypass auto-alignment
--no-auto-align        use --offset-ms / recorder start only
--preview N            write N sample PNGs instead of a video
--render-rail PATH     write the seven rail states as one sheet and exit
```

### `record_overlay.py`

```
MODEL --episodes N --out-dir DIR
--serial 3f80a3a --bitrate 16M --fps 60
--no-render            record only
(remaining args forwarded to capture_calib.py)
```

---

## 8. Failure modes

| Failure | Handling |
|---|---|
| scrcpy or ffmpeg missing | fail immediately, name the resolved path |
| second scrcpy already running | warn — extra device encode load |
| hard kill leaves corrupt mp4 | `CTRL_BREAK_EVENT` shutdown, then `ffprobe` verification |
| device drops mid-run | `capture_calib` already reports `degraded`; recording continues; a capture shorter than the video renders an empty rail for the remainder |
| alignment finds no peaks | fall back to recorder start time, warn loudly, still render |
| capture has no `t_host` (older file) | fall back to `step / 10.0`, warn that drift is likely |

---

## 9. Verification

Everything below runs **without the phone**:

1. `--render-rail` writes the seven states as a PNG sheet — the icons are inspected directly.
2. Alignment is unit-tested against synthetic `R`/`V` sequences with a known offset, known drift,
   spurious peaks, and a missing episode.
3. End-to-end on a synthetic video: `ffmpeg testsrc` plus a hand-written jsonl, then `--preview`
   frames are checked for the ring being lit on the expected cell at the expected time.

On the phone, the first real recording checks one thing specifically: **centring puts the rail under
the pause button and idol** in the bottom-right corner — precisely why the game offsets its own hint
bar to the left. If it collides, narrow the rail rather than un-centring it.

---

## 10. Open

* Icon shapes are a reading of the user's reference image; coordinates are trivial to adjust.
* Rail width may need narrowing after the first real recording (see above).
