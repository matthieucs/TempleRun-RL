# Opening prompt for the GPU-box Claude Code session (phone ATTACHED — full autonomous loop)

Copy-paste everything below the line into the first message.

---

You have a GPU **and the phone is plugged in**, so you run the full autonomous loop end-to-end: model
the sim → train → gate offline → device-eval on the phone → iterate — exactly the loop the prior session
ran. Start by reading `CLAUDE.md` in full; it's the handoff doc and it's current. One-breath context:
this is an RL agent that plays real Temple Run; the 10,000 m goal was REACHED yesterday (device run
**10,545 m**, policy v36). You are in the **RELIABILITY phase — the job is a 10,000 m MEDIAN, not just a
best.** Yesterday's device runs ranged 589 → 10,545 m; killing that variance is the whole task.

## First: verify the rig
1. Read CLAUDE.md (especially §Setup, §Run it, §Protocol, §STATUS, and the GOAL-REACHED block).
2. `torch.cuda.is_available()` must be True.
3. `adb devices` — **the phone serial is probably NOT `3f80a3a`** (that was the old machine's phone). Note
   the real serial and use it everywhere (or update the setup scripts). Confirm it's the SAME APK build:
   the gadget-injected `TempleRun-gadget.apk` must be installed (Frida gadget on `libmain.so`) and the
   dump.cs offsets must match — if it's a different phone/build, rebuild the APK (`tools/build_gadget_apk.py`)
   and re-dump offsets before trusting any device read.
4. `fidelity_scorecard.py --strict` GREEN on this checkout = your training baseline.

## The training task: a stable FROM-SCRATCH 256-net
Yesterday's wall (see the v37–v40 campaign in CLAUDE.md): a 128-net can't hold reach-center-narrows AND
centered-crevasses at once — they interfere; warm-start retrains just trade one skill for another. A
256-net *widened* from v36 broke the interference offline (narrow 0/50 + crevasse 3/40) but drifts
because it inherited v36's 128-net optimum. The clean shot is a 256-net trained **from scratch** on the
full scenario stack. `scratchpad/train_gpu_256.sh` is the recipe — raise `--num-envs` to your core count
(bottleneck is the CPU-bound Python sim, not the tiny MLP; the GPU alone buys little). Expect >6M steps
to reach v36's quality from scratch; keep the BEST checkpoint, not `_final`.

## Fold in the uncovered edge cases (the variance sources)
Model each in the sim, keep `--strict` GREEN, don't break existing mechanics. Device-GROUND each like the
crevasse/narrow calibrations were (measure on the phone, then fit the sim — never blind-tune; the prior
session burned a day proving offline-green ≠ device-safe for un-modeled mechanics):
1. **Flame-tower slide-under at speed** — the 10,545 m death (Burnt, pt31/32). Probe the slide-under
   window on the phone, then model it.
2. **Multi-hole jump rhythm** — 4-cell must-jump chains; jump-spam fails, needs staggered land-then-rejump.
   (May be partly game-hard at low speed: a 4-cell hole needs v>200 to span.)
3. **Narrows-at-speed** — already modeled (`strafe_exec_lat_frames` + strict band [3,7]); the 256-net is
   the capacity to hold it alongside crevasses.
4. **Stumble → rejump recovery.**

## Reliability workflow (this is different from chasing a best)
- **Measure the MEDIAN, not the max.** 8-episode device evals are NOISY — run more episodes per candidate
  and report the median + the death-cause histogram, not the top run.
- **Same-session, fresh-phone comparisons only.** Restart the game (force-stop → relaunch → 12 s → re-forward
  → re-inject) before each model's eval; phone state degrades and stale injectors corrupt evals.
- A winner beats v36 on the MEDIAN across ALL axes (narrows, crevasses, holes, compounds, flame) — not one.

## Device-ops playbook (hard-won — follow exactly)
- **Run EXACTLY ONE injector.** The host process must stay alive (if it exits, the in-phone agent unloads).
- **NEVER `pkill -f`** — it self-kills the shell (exit 144). Kill the OLD injector by numeric PID only
  (`pgrep -f gadget_inject` → `kill <PID>`) BEFORE re-injecting; stale injectors re-attach after a
  force-stop+restart → multiple agents, port contention, corrupted evals.
- **Never chain `sleep`** in one command (harness blocks it); use `until <cond>; do sleep N; done` in a
  `run_in_background` job, and background long jobs with `setsid nohup … &`.
- Keep-awake: `appops set com.android.shell android:write_settings allow` + `settings put system
  screen_off_timeout 2147483647`. Recover a degraded phone with the force-stop→relaunch recipe in CLAUDE.md.
- **Device-true offline eval config includes `strafe_exec_lat_frames:1`** (the phone lags strafe too) —
  `scratchpad/eval_swipe_deaths.py` is the offline gate; deaths-by-mechanic is the signal, not the median
  (which caps at the sim step limit).

## Hard constraints (unchanged)
NN decides ALL actions — no hook/host action rules (perception/execution translations only); power-ups
stay off; hook stays strictly read-state + inject-tap (the GAME is unmodified — device runs are NOT
seedable); `--strict` GREEN before a model is shippable; keep BEST checkpoints; be honest about what the
scorecard can and can't see. Deploy today = v36 (`runs/ppo_best.zip`, gitignored — it won't be on the
fresh clone; you'll rebuild candidates from scratch anyway, but if you need v36 as a baseline, copy it
over from the phone machine).
