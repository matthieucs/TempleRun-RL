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
  byte0/1 = hold ±, byte6 = level→game recenters. **Turn = swipe** (bytes 4/5), separate mechanic.
- **Hook** `frida/temple_run_hook.js`, version in `HOOK_VERSION` (`_hv` on the stream). Current
  **16i**. Streams the raw state + derived obs fields; executes the policy's action bytes (NO action
  rules — the NN decides; hook only senses + dispatches).

## Key offsets (this build; from dump.cs)
- **CharacterPlayer**: IsDead 0xD8, IsJumping 0xF8, IsSliding 0x100, Score 0x138, Velocity(Vec3) 0x34,
  Facing 0x78, JumpVelocity 0x80, GroundHeight 0x88 (=raised ledge height), IsGroundHeightChangeHigher.
- **PathElement**: Type 0x30, Level 0x34, Obstacles 0x38, neighbors 0x50/58/60/68, Rectangle 0xA0.
- **Enums**: ObstacleType {Hole1, Narrow2, TreeStumble3, TreeJump4, TreeSlide5, FlameTower6,
  LevelChange7}. PathType: 1-2 straight, 3-10 turns, 11-16 holes (12/13/15/16=HoleLong=2-cell),
  17-30 narrows, 31-32 flame, **33-36 = SingleLevelStepDownCap (zone crevasse, both up/down)**, 37 temple.
  DeathType {Fall1, Tree2, Slide3, **Ledge4**, Burnt5, ...}. MaxRunVelocity **210** (start 100).
- RVAs (Start/Update/Turn/Jump/Slide/Strafe/CanTurn/IsTurn/GetPosition2D/RunAgain/OnPlay/…): see
  `docs/CLAUDE_FULL_HISTORY.md` §5 / `docs/OFFSETS.md`.

## Sim + training
- `src/templerun_rl/sim_env.py` = fast headless model (`TempleRunSim-v0`); obs `obs.py` (**OBS_LEN 32**,
  bools + scalars + cell_progress/lane/narrow_target_lane/**predicted_landing**/next_ground_change_higher
  + K=4 action history → 60-wide); reward `reward.py`; PPO `scripts/train_sim.py` (net 128,128, linger 3,
  ent 0.04). Deploy = same obs on the phone via the hook.
- **THE FEEDBACK LOOPS (use before any retrain/deploy):**
  - `scripts/fidelity_scorecard.py --strict` — ~13 phone-free checks vs `captures/device_reference.json`.
    GREEN required before retrain. (Blind for un-checked mechanics — green ≠ complete.)
  - `scripts/replay_diff.py <capture>` — policy-free ONE-STEP sim-vs-device transition diff on a real
    capture. Great for turns/obstacles; CANNOT test multi-frame mechanics (the crevasse jump arc).
  - Device eval: `scripts/capture_calib.py <model> --episodes 8 --turn-cooldown 0`.
- Checkpoints are noisy + late-collapse → **keep the BEST checkpoint, not `_final`**. Registry `runs/MODELS.md`.

## STATUS — what worked / what failed / where we are

**WORKED (device-verified):**
- **Turns FIXED** — the structural wall that beat 40+ policies. Fix = `predicted_landing` obs
  (`cell_progress + Δ·velocity` = "where a swipe-now lands"; the turn executes ~1 frame late, so the
  policy must aim the LANDING at cprog 0.44, not the swipe). v10 device: turn landings |lane| ~4 (was
  10-17 fatal), turn deaths ~0. Pure perception input, no hook rule.
- **Narrows** — side-strip lanes; cracked via a NARROW-ONLY isolation curriculum → warm-start full task.
- **Jump-into-slide death**, **turn-direction** (native CanTurn opposite-origin, hook 16f), **lateral =
  analog tilt** (alpha 0.446/0.1s), **stumble** 2-strike — all modeled + scorecard-guarded.

**FAILED / OPEN — CREVASSES (the current wall):**
- After turns were fixed, crevasses (zone step-ups, pt 33-36) became the top device death (~600 m,
  DeathType 4 = Ledge). replay_diff proved it's the **SIM, not the reward** (reaction_bonus unchanged;
  it rewards jumping, never timing — the sim must enforce timing, and it was too lenient).
- Device facts: velocity capped 210 (human's 14,956 m run confirms; speed plateaus ~4000 m). The
  crevasse is a **2-cell step** — the raised ledge (`ground_height=15`) is on the 2nd cell; the runner
  clips it (Ledge) if not near jump apex there. Takeoff is **velocity-dependent** (human jumps earlier at
  high speed). Many sim models tried (fixed late-gate → velocity-dependent apex → **2-cell apex, current**);
  each improved in-sim but device crevasses still fail. The apex threshold is NOT yet calibrated to the
  ground_height=15 ledge. **v210 crevasse takeoff not cleanly measured** (chicken-and-egg: must survive to
  4000 m to reach v210 on device).

**WHERE WE ARE:**
- **Deploy floor = v1** (`runs/ppo_valid_v1_best.zip`, device median **1173**) — still the best real result.
- Best real single run = **v10's 2608 m** (turns fixed, crevasse-limited).
- **v14 training** = warm-start v13 + the 2-cell crevasse apex model (scorecard GREEN). Pending device eval.
- NEXT: calibrate the crevasse apex to the exact ledge (read `GroundHeight`/`JumpVelocity` live), device
  eval v14, and/or a crevasse-only isolation curriculum swept across velocities (the trick that cracked
  narrows). NN decides all actions — no hook action rules.
