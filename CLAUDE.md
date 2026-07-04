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
  **16p** (`narrowjump-16p`). Streams raw state + derived obs; executes action bytes. NO action-DECISION
  rules (NN decides); the swipe-queue + must_jump-flag are perception/execution translations, not decisions.

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
- `src/templerun_rl/sim_env.py` = fast headless model (`TempleRunSim-v0`); obs `obs.py` (**OBS_LEN 27**,
  bools + scalars + cell_progress/lane/narrow_target_lane/**predicted_landing**/next_ground_change_higher
  + K=4 action history → 55-wide; power-ups REMOVED 2026-07-04 — always-0 dead inputs); reward `reward.py`;
  PPO `scripts/train_sim.py` (net 128,128, linger 3, ent 0.04). Deploy = same obs on the phone via the hook.
- Sim turn-buffer config: `hook_turn_buffer=true` models the device swipe-queue (turn near a junction lands
  CENTERED, timing-free, `from_buffer` offset 0). `runs/ppo_v14_obs27.zip` = obs-27 warm-start base (EXACT
  surgical equivalent of v14; `scratchpad/surgery_v22.py`).
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

**FAILED — RETRAINING IS EXHAUSTED (2026-07-03/04):**
- **12 straight retrain regressions (v15→v22).** Every warm-start+finetune from v14 drifts BELOW it on
  device, regardless of change (high-speed curriculum, off_center_penalty [device-TOXIC], turn_lane_carry
  "fidelity" [it's a load-bearing REGULARIZER, keep 3.0], tail-risk, action_delay, low-ent, buffer models).
  Offline metrics do NOT predict device turns. **v14 is a fragile optimum; STOP retraining.**
- **What actually works here = HOOK/perception fixes, NOT retraining** (native turn swipe-queue,
  small-narrow must_jump). The turn wall broke via a hook change on v14, no retrain.
- Ruled out (device-measured): over-strafe lateral gap (device caps lane ±7, same as sim); 20Hz sampling
  (already tried, WORSE); crevasse apex miscalibration (sim clears the human's v210 0.58 takeoff).

**OPEN — CREVASSES are the remaining wall** (top death when the phone is healthy, DeathType 4=Ledge). Same
  ~2-frame latency root as turns, but NO clean game-native fix (jumps fire immediately — no "buffer at the
  gap" like turns). Velocity caps 210 (~4000 m). Needs a perception "jump-earlier" signal + a non-regressing
  training method, or the crevasse-latency angle (untried).

**WHERE WE ARE (2026-07-04):**
- **Deploy floor = v14** (`runs/ppo_best.zip`, device **median 1652, best 2908**). Best real run tonight 2430.
- **Deliverable added (no regression):** obs27 (power-ups out) + hook 16p (clean swipe-queue turns +
  small-narrow jump). Warm-start base `runs/ppo_v14_obs27.zip`.
- **⚠️ PHONE DEGRADED after hours of runs** (both v14+16p AND v22 died ~600-800 at crevasses, anomalous vs
  1652). **RE-MEASURE v14 on hook 16p on a FRESH/rebooted phone** for the true median.
- NN decides all actions. Branch `sim-calibration` pushed (safe to merge; `runs/*.zip`+MODELS.md gitignored).
