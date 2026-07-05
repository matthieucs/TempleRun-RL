# Temple Run RL — project memory (minimal)

Goal: an RL policy that reaches **10,000 m** on real **Temple Run** (`com.imangi.templerun` v1.39.0),
trained in a headless sim, deployed on a physical Android phone via **Frida** (read game state + inject
actions over TCP — no screen-scraping, no synthetic input). Full history: `docs/CLAUDE_FULL_HISTORY.md`.

## Setup
- **Engine** Unity 2022.3 IL2CPP, arm64 only → needs a REAL arm64 phone (no emulator: Frida can't hook
  the ARM lib under x86 translation). Device **3f80a3a** (Redmi, Android 15, NOT rooted).
- **Frida Gadget** (not frida-server — no root): `libgadget.so` injected via LIEF `DT_NEEDED` on
  `libmain.so`; listens 127.0.0.1:**27042**. Build: `tools/build_gadget_apk.py` → `TempleRun-gadget.apk`.
- **Reverse eng.**: Il2CppDumper → `il2cpp_work/output/dump.cs` (signatures + field offsets + RVAs;
  bodies are native, not readable). All offsets/RVAs below are from it (re-dump if the APK changes).

## Run it
`python scripts/setup_device.py` forwards 27042 + state **9999** + action **9998**, then injects.
Inject one-liner (run EXACTLY ONE; it exits → in-phone agent unloads):
`setsid nohup .venv/bin/python frida/gadget_inject.py frida/temple_run_hook.js 127.0.0.1:27042 &`
Recover a flaky/crash-looped phone: `adb -s 3f80a3a shell am force-stop com.imangi.templerun` →
`am start -n com.imangi.templerun/com.imangi.unityactivity.ImangiUnityActivity` → wait 12 s → re-forward
→ re-inject. Verify `_hv` on :9999. Keep-awake (§HyperOS): `appops set com.android.shell
android:write_settings allow` + `settings put system screen_off_timeout 2147483647`.

## Protocol
- **State (9999)**: newline JSON @ **10 Hz** (`POLL_MS=100`; game Updates @60Hz, so faster IS possible).
- **Actions (9998)**: 1 byte over ONE persistent socket. `0`=StrafeLeft `1`=StrafeRight `2`=Jump
  `3`=Slide `4`=TurnLeft `5`=TurnRight `6`=NoOp; reset-only `8`=RunAgainClicked `9`=OnPlayClicked
  (`reset()` sends 8 then 9). **Lateral = analog tilt** (hook replaces `GetAccelerometerForceX`):
  byte0/1 = hold ±, byte6 = level→game recenters.
- **Turn = the game's own swipe-queue (2026-07-04, hook 16o+):** the turn byte writes `GI_ShouldTurnLeft/
  Right`+`TimeSinceShouldTurn=0` on GameInput (HandleDelayedTurning) → the GAME fires the turn AT THE
  CORNER with native timing, exactly like a real early swipe. This SIDESTEPS the ~2-frame control latency
  that made plain immediate `fn.TurnLeft` overshoot (see STATUS). NN still decides WHEN to turn; the hook
  only queues it (like the analog-tilt translation). `TURN_NATIVE_BUFFER=true`; anti-double-turn debounce +
  valid-dir guard. (The every-frame flag-HOLD variant looked "exploity" — removed; single write only.)
- **Small narrows (pt17/18) → `must_jump`** (hook 16p): human data = jumps them 82%. The policy's existing
  must_jump→jump generalizes → v14 jumps them, no retrain. Set on both `cachedMustJump` and streamed `s.must_jump`.
- **Hook** `frida/temple_run_hook.js`, version in `HOOK_VERSION` (`_hv` on the stream). Current
  **16r** (`leveldir-16r`) = 16p + streams `next_path_level`/`next2_path_level` (upcoming-cell PE_Level).
  Device-VERIFIED 2026-07-05 (10/10): future PE_Level IS readable — the old "reads 0" assumption was WRONG;
  `sign(next2_path_level − path_level)` = the crevasse direction (up/down). Streams raw state + derived
  obs; executes action bytes. NO action-DECISION rules (NN decides); the swipe-queue + must_jump-flag are
  perception/execution translations, not decisions.

## Key offsets (this build; from dump.cs)
- **CharacterPlayer**: IsDead 0xD8, IsJumping 0xF8, IsSliding 0x100, Score 0x138, Velocity(Vec3) 0x34,
  Facing 0x78, JumpVelocity 0x80, GroundHeight 0x88 (=raised ledge height), IsGroundHeightChangeHigher.
- **PathElement**: Type 0x30, Level 0x34, Obstacles 0x38, neighbors 0x50/58/60/68, Rectangle 0xA0.
- **GameController.GameInput @0x78** (HandleDelayedTurning buffer — the turn swipe-queue): GI_JustTurned
  0x20, GI_ShouldTurnLeft **0x21**, GI_ShouldTurnRight **0x22**, GI_TimeSinceShouldTurn 0x24,
  GI_TimeSinceLastTurn 0x28, GI_ShouldJump 0x2C, GI_TimeSinceShouldJump 0x30. (Hook writes ShouldTurnL/R.)
- **Enums**: ObstacleType {Hole1, Narrow2, TreeStumble3, TreeJump4, TreeSlide5, FlameTower6,
  LevelChange7}. PathType: 1-2 straight, 3-10 turns, 11-16 holes (12/13/15/16=HoleLong=2-cell),
  17-30 narrows, 31-32 flame, **33-36 = SingleLevelStepDownCap (zone crevasse, both up/down)**, 37 temple.
  DeathType {Fall1, Tree2, Slide3, **Ledge4**, Burnt5, ...}. MaxRunVelocity **210** (start 100).
- RVAs (Start/Update/Turn/Jump/Slide/Strafe/CanTurn/IsTurn/GetPosition2D/RunAgain/OnPlay/…): see
  `docs/CLAUDE_FULL_HISTORY.md` §5 / `docs/OFFSETS.md`.

## Sim + training
- `src/templerun_rl/sim_env.py` = fast headless model (`TempleRunSim-v0`); obs `obs.py` (**OBS_LEN 30**,
  bools + scalars + cell_progress/lane/narrow_target_lane/**predicted_landing**/next_ground_change_higher
  + **predicted_jump** (velocity-invariant step-up takeoff cue, host-derived) + **next_path_level/
  next2_path_level** (upcoming-altitude direction, hook 16r) + K=4 action history → 58-wide; power-ups
  REMOVED 2026-07-04); reward `reward.py`; PPO `scripts/train_sim.py` (net 128,128, linger 3).
  Deploy = same obs on the phone via the hook.
- Sim turn-buffer config: `hook_turn_buffer=true` models the device swipe-queue (turn near a junction lands
  CENTERED, timing-free, `from_buffer` offset 0) + **late gate `turn_buffer_late_cp=0.56`** (swipe past it =
  silent MISS; device-measured CanTurn window end, velocity-INDEPENDENT — v14 turned 46/46 up to v207).
  Warm-start base **`runs/ppo_v14_obs30_leveldir.zip`** (EXACT surgical equivalent of v14, new obs cols
  zero-init; `scratchpad/surgery_v24_leveldir.py`).
- **JUNCTION LEVEL DROPOUT (2026-07-05 night, device-measured):** the hook's PE neighbor walk BREAKS through
  turns → on-junction frames stream `next_path_level=next2_path_level=-1` (178/202), approach frames
  `(lvl,-1)`. Sim `_snapshot` reproduces it (commit f8d7eff). Obs30 finetunes trained pre-fix saw a
  junction-frame input spike on device they never met in training. v14 immune (zero-init cols).
- **Capture-log alignment (capture_calib):** logged `action(i)` = policy's response to `state(i-1)` (predict
  runs BEFORE env.step; the record holds the POST-step state). And a FIRED turn resets cell_progress → the
  swipe record's cp is POST-turn — don't read "swipe cp" stats off fired turns (artifact; burned us twice).
- **Jump exec latency (sim, 2026-07-05):** `jump_exec_lat_frames`/`jump_exec_lat_jitter` buffer the JUMP
  firing. DEVICE-MEASURED (`scripts/probe_jump_exec.py`, 25/25): **exactly 1 frame @10Hz, zero jitter**
  (uniform with the turn-side 1-frame exec). Constant latency is ABSORBED by policies (window shifts);
  only JITTER creates training pressure (v14 hiMed 2246→130 under 2±1; a predicted_jump pre-aim oracle
  survives). Train at **1±1** (measured mean + TCP obs-staleness variance); harness/replay at 0.
- **Warm-start recipe that does NOT collapse: `--learning-rate 5e-5`** (new knob; default 3e-4 ALWAYS
  collapses a v14 warm-start regardless of ent). Keep the BEST checkpoint — drift resumes past ~500k-1M.
- **THE FEEDBACK LOOPS (use before any retrain/deploy):**
  - `scripts/fidelity_scorecard.py --strict` — ~13 phone-free checks vs `captures/device_reference.json`.
    GREEN required before retrain. (Blind for un-checked mechanics — green ≠ complete.)
  - `scripts/replay_diff.py <capture>` — policy-free ONE-STEP sim-vs-device transition diff on a real
    capture. Great for turns/obstacles; CANNOT test multi-frame mechanics (the crevasse jump arc).
  - Device eval: `scripts/capture_calib.py <model> --episodes 8 --turn-cooldown 0`.
- Checkpoints are noisy + late-collapse → **keep the BEST checkpoint, not `_final`**. Registry `runs/MODELS.md`.

## STATUS — what worked / what failed / where we are

**WORKED (device-verified):**
- **Turns FIXED — via the game's own swipe-queue (2026-07-04, the real fix).** ROOT CAUSE of the turn wall
  = **~2-frame control latency** over Frida-TCP (obs is ~1 frame stale: policy acts at device-cp ~0.50 vs
  sim-cp 0.30; + the turn executes ~1 frame late → device-cp ~0.73), which EATS the tight survivable swipe
  window. `predicted_landing` obs (v10) helped but did NOT close it; retrains kept overshooting (lane
  15-22). THE FIX: route the turn byte through the game's native HandleDelayedTurning (`GI_ShouldTurnLeft`
  write) → the GAME times the turn at the corner, immune to the lag. v14+hook-16k device: **turn-overshoots
  0/8** (was the dominant death). See Protocol.
- **Small narrows JUMPED (hook 16p)** — human jumps them 82%; hook flags pt17/18 `must_jump` → v14 jumps
  (device-verified), no retrain.
- **Narrows (long side-strips)** — cracked via a NARROW-ONLY isolation curriculum → warm-start full task.
- **Jump-into-slide death**, **turn-direction** (native CanTurn opposite-origin, hook 16f), **lateral =
  analog tilt** (alpha 0.446/0.1s), **stumble** 2-strike — all modeled + scorecard-guarded.

- **Crevasses LARGELY FIXED (v14, device-verified 2026-07-03)** — was the top device death after turns
  (v10: 5/8 @~600 m, DeathType 4 = Ledge). replay_diff proved it was the **SIM, not the reward** (reward
  rewards jumping, never timing; the sim was too lenient). Fix = model the crevasse as a **2-cell step**:
  gap on cell 1, raised ledge (`ground_height=15`) on cell 2 where a velocity-dependent APEX check fires
  (clip = Ledge if not near jump apex there). Retrained (v14) → **device median 1652, best 2908, crevasse
  deaths 2/8** (down from 5/8). `next_ground_change_higher` obs (pt 33-36, hook 16i) **verified firing on
  device** (7/7 crevasse-approach frames) — prior "0 fires" was a capture-logging artifact, not a bug.

**FAILED / LESSONS (2026-07-03/05):**
- **12 retrain regressions (v15→v22) at LR 3e-4.** Every warm-start+finetune from v14 drifted BELOW it on
  device (high-speed curriculum, off_center_penalty [device-TOXIC], turn_lane_carry "fidelity" [it's a
  load-bearing REGULARIZER, keep 3.0], tail-risk, action_delay, low-ent, buffer models). Offline metrics do
  NOT predict device turns. **AMENDED 2026-07-05: the collapse mechanism was the LR** — 3e-4 knocks the
  policy off v14's sharp optimum; **5e-5 fine-tunes stably** (v25c/v26 verified: no collapse, skills adopt).
- **v25 (2-phase: crevasse-only isolation → full) DEVICE-REGRESSED (median ~650): 6/8 deaths = MISSED
  TURNS.** Crevasse-ONLY training ATROPHIES device turns even after a full-task Phase B restores them
  in-sim. **Isolation curricula: OK for narrows, WRONG for turns — keep turns practiced in every phase.**
  Also 2/8 crevasse jumped too EARLY = the guessed lat 2±1 over-modeled (real = 1). Both fixed in v26.
- Ruled out (device-measured): over-strafe lateral gap (device caps lane ±7, same as sim); 20Hz sampling
  (WORSE); crevasse apex miscalibration (sim clears the human's v210 0.58 takeoff).

**2026-07-05 NIGHT (autonomous session) — the retrain curse + turn mechanics fully decoded:**
- **v14 FRESH-PHONE: median 2010, best 5462, 0 missed turns in 212 junctions** (`captures/
  v14_freshphone.jsonl`). The old 1652 was the DEGRADED phone. ⚠️ ALL cross-model comparisons must be
  same-session fresh-phone (v26/v27's "missed turns" were measured on the tired phone).
- **Turn mechanics (205+ junctions, frame-level):** every policy decides the swipe on the junction's FIRST
  frame; byte goes out 1 control-tick later. Early-phase onsets (cp<0.2) ALWAYS survive; late-phase onsets
  (cp≥0.3) = a latency-jitter COIN FLIP (write lands 0.5-1.5 frames after the onset sample vs the CanTurn
  window end ~0.56). Fresh phone = low jitter = coin lands good (v14 50/50 late onsets); degraded = misses
  (v26/v27: 5/9+ late-onset deaths). Wrong models tried + discarded: v29 constant-0.50 gate (never binds
  in-sim), v30 velocity-arrival lag-2 (refuted: v14 turns fine @v207). Hook drop paths (turn byte):
  debounce 0.3s / invalid-dir(CanTurn false) / not-near-junction — pre-onset turn bytes: 0 in all captures.
- **JUNCTION LEVEL DROPOUT found + fixed** (see Sim section) = why obs30 retrains looked worse on device.
  v31 (v27 recipe + dropout + 0.56 gate) trained; in-sim clean turns WITH dropout. NOTE: in-training
  EvalCallback env ≠ training env (no buffer flags) → its curve says "collapse" when device-true eval says
  healthy. Judge checkpoints with `scratchpad/eval_swipe_deaths.py` (device-true cfg, deaths by mechanic).
- **NEXT WALL @v200+ (both v14 long runs died on it): chained-jump blindness** — a jump spans ~2.75 cells,
  obs lookahead = 1 cell → lands blind into TreeJump (3738m, 5462m deaths). Lever: next2-obstacle obs
  (hook 2-ahead walk device-verified readable on straights; breaks only through turns = dropout above).
- **USER (live, 2026-07-06): low-speed dumb deaths (missed transition/turn) are the MEDIAN killer** —
  high speed works; prioritize low-speed reliability.

**WHERE WE ARE (2026-07-06):**
- **Deploy floor = v14** (`runs/ppo_best.zip`, fresh-phone **median 2010, best 5462**; width 60 — needs
  v14-era obs.py: use a git worktree at 949a6e2, PYTHONPATH=wt/src, NOT a stash).
- Crevasses: v27 best in-sim (3/20 @v210 vs v14 8/20; device 1/8 on tired phone). Fresh-phone v27 re-eval
  RUNNING (`captures/v27_freshphone.jsonl`) → then v31@200k same-session. Best of the two = new candidate.
- NN decides all actions. Branch `sim-calibration` (safe to merge; `runs/*.zip`+MODELS.md gitignored).
