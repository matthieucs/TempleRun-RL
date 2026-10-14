#!/usr/bin/env python3
"""Record the phone while the policy plays, then burn the action rail onto the recording.

    python scripts/record_overlay.py runs/ppo_v111_lh030_final.zip --episodes 5

Produces, in --out-dir:
    run.mp4            the raw screen recording
    run.jsonl          the capture (actions + state), as capture_calib writes it
    run.meta.json      recorder metadata, incl. the host time the recording started
    run_overlay.mp4    the recording with the rail burned in

ONE scrcpy INSTANCE. It both mirrors and records, so you still watch live. Do NOT run a second
scrcpy alongside: each instance spawns its own encoder on the device, and this phone is measured to
be thermally fragile (throttling collapses policy distance by 2-3x).

CLEAN SHUTDOWN MATTERS. An mp4 needs its trailer written, and on Windows Popen.terminate() is a hard
kill that leaves the file unplayable. scrcpy is therefore started in its own process group and
stopped with CTRL_BREAK_EVENT, with --time-limit as a backstop so a failed signal cannot leave a
recording running forever. The result is verified with ffprobe and repaired if possible.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# documented location on this box; scrcpy 3.3.4 lives one level deeper than the extracted folder
SCRCPY_FALLBACKS = (
    r"D:\téléchargements\scrcpy-win64-v3.3.4\scrcpy-win64-v3.3.4\scrcpy.exe",
    r"D:\telechargements\scrcpy-win64-v3.3.4\scrcpy-win64-v3.3.4\scrcpy.exe",
)


def find_scrcpy(explicit: str | None) -> str:
    for cand in (explicit, os.environ.get("SCRCPY"), shutil.which("scrcpy"), *SCRCPY_FALLBACKS):
        if cand and Path(cand).exists():
            return str(cand)
    sys.exit("error: scrcpy not found. Pass --scrcpy <path> or set the SCRCPY environment "
             "variable.\n       Tried: " + ", ".join(str(c) for c in SCRCPY_FALLBACKS))


def scrcpy_already_running() -> bool:
    if os.name != "nt":
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq scrcpy.exe"],
                             capture_output=True, text=True, timeout=10)
        return "scrcpy.exe" in out.stdout
    except Exception:
        return False


def playable(video: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not video.exists() or video.stat().st_size == 0:
        return False
    r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() not in ("", "N/A")


def stop_scrcpy(proc: subprocess.Popen, timeout: float = 25.0) -> None:
    if proc.poll() is not None:
        return
    print("stopping scrcpy...")
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  could not signal scrcpy ({exc}); falling back to terminate")
        proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("  scrcpy did not exit on the stop signal; terminating "
              "(the mp4 may need repair)")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def repair(video: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    fixed = video.with_name(video.stem + "_fixed.mp4")
    r = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(video), "-c", "copy", str(fixed)],
                       capture_output=True, text=True)
    if r.returncode == 0 and playable(fixed):
        video.unlink(missing_ok=True)
        fixed.rename(video)
        return True
    fixed.unlink(missing_ok=True)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="policy .zip passed straight to capture_calib.py")
    p.add_argument("--episodes", type=int, default=4)
    # videos live on the big HDD, not next to the repo: a 6-minute raw capture is ~730 MB
    p.add_argument("--out-dir", type=Path,
                   default=Path(os.environ.get("TR_VIDEO_DIR",
                                               r"D:\Robotique\Temple-run RL\captures\video")))
    p.add_argument("--serial", default="3f80a3a")
    p.add_argument("--bitrate", default="16M")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--max-size", type=int, default=0, help="0 = native resolution")
    p.add_argument("--scrcpy", default=None)
    p.add_argument("--time-limit", type=int, default=0,
                   help="hard cap in seconds (default: 360 per episode plus 300)")
    p.add_argument("--allow-existing", action="store_true",
                   help="proceed even if another scrcpy is already running")
    p.add_argument("--no-render", action="store_true", help="record only, do not composite")
    p.add_argument("--encoder", choices=("auto", "nvenc", "x264"), default="auto",
                   help="auto uses the GPU's NVENC block when it works, else the CPU")
    p.add_argument("--startup-timeout", type=float, default=25.0)
    a, extra = p.parse_known_args()

    scrcpy = find_scrcpy(a.scrcpy)
    if scrcpy_already_running() and not a.allow_existing:
        sys.exit("error: another scrcpy is already running.\n"
                 "       Close it first -- a second instance spawns a second encoder on the phone, "
                 "which costs real device performance.\n"
                 "       This script's own instance both mirrors and records, so you lose nothing.\n"
                 "       Use --allow-existing to override.")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    video = a.out_dir / "run.mp4"
    actions = a.out_dir / "run.jsonl"
    meta = a.out_dir / "run.meta.json"
    for stale in (video, actions):
        if stale.exists():
            stale.unlink()

    limit = a.time_limit or (a.episodes * 360 + 300)
    cmd = [scrcpy, "-s", a.serial, "--record", str(video),
           "--video-bit-rate", a.bitrate, "--max-fps", str(a.fps),
           "--no-audio", "--time-limit", str(limit)]
    if a.max_size:
        cmd += ["--max-size", str(a.max_size)]

    print("scrcpy  " + " ".join(cmd[1:]))
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    t_rec_start = time.time()
    proc = subprocess.Popen(cmd, creationflags=flags)

    # wait for the recording file to appear before letting the policy start, so the opening
    # seconds of episode 0 are actually captured
    deadline = time.time() + a.startup_timeout
    while time.time() < deadline and not video.exists():
        if proc.poll() is not None:
            sys.exit(f"error: scrcpy exited immediately with code {proc.returncode}")
        time.sleep(0.2)
    if not video.exists():
        stop_scrcpy(proc)
        sys.exit(f"error: scrcpy produced no recording within {a.startup_timeout:.0f}s")
    time.sleep(1.0)      # let the first frames land
    print(f"recording -> {video}")

    meta.write_text(json.dumps({
        "t_rec_start": t_rec_start,
        "serial": a.serial,
        "bitrate": a.bitrate,
        "max_fps": a.fps,
        "episodes": a.episodes,
        "model": str(a.model),
        "scrcpy": scrcpy,
    }, indent=2))

    cap_cmd = [sys.executable, str(HERE / "capture_calib.py"), a.model,
               "--episodes", str(a.episodes), "--out", str(actions), *extra]
    print("capture " + " ".join(cap_cmd[1:]))
    rc = 0
    try:
        rc = subprocess.run(cap_cmd, cwd=str(REPO)).returncode
    except KeyboardInterrupt:
        print("\ninterrupted -- stopping the recording cleanly so the mp4 stays usable")
        rc = 130
    finally:
        stop_scrcpy(proc)

    if not playable(video):
        print("mp4 has no valid trailer; attempting repair...")
        print("  repaired" if repair(video) else "  REPAIR FAILED -- the recording is unusable")
    if rc != 0:
        print(f"WARNING: capture_calib exited {rc}; rendering whatever was captured")
    if not actions.exists():
        sys.exit("error: no capture file was written; nothing to overlay")

    if a.no_render:
        print(f"done (no render). {video} + {actions}")
        return

    out = a.out_dir / "run_overlay.mp4"
    render = [sys.executable, str(HERE / "overlay_video.py"),
              "--video", str(video), "--actions", str(actions),
              "--out", str(out), "--fps", str(a.fps), "--encoder", a.encoder]
    print("render  " + " ".join(render[1:]))
    subprocess.run(render, cwd=str(REPO))


if __name__ == "__main__":
    main()
