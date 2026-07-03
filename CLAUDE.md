# Temple Run RL — project memory

Goal: turn **Temple Run** (`com.imangi.templerun`, v1.39.0) into a reinforcement-learning
environment by instrumenting the running game with **Frida** — read game state and inject
actions through TCP sockets, no screen-scraping, no synthetic input.

## 17. TURN MECHANIC MEASURED + TRACK-FORCING (2026-07-02) — read this for turns

**THE turn mechanic (device-MEASURED, `scripts/probe_turn_exec.py`, 40 controlled swipes):** a turn
executes INSTANTLY (facing flips next frame; the game's HandleDelayedTurning buffer is NOT used) and
the runner LANDS at lateral offset ≈ **(swipe_cprog − 0.44) × 45 + pre_turn_lane × ~3**. So: swipe at
the `is_turn` flash (cprog ~0.2) → land 8–12 off → **water**; swipe at **cell CENTER (~0.44)** from
**lane 0** → centered → survive. Survivable window ~[0.24, 0.60], |offset| kill ~9. Two ways to die:
(1) swipe too early/late (off the center cprog), (2) **STRAFE INTO the turn** (leaning) — the pre-turn
lane is AMPLIFIED ~3.5× (lane 5 → lands ~18) AND the analog tilt PERSISTS through the turn → off-path.
The policy MUST be centered (NoOp, lat_target 0) AND swipe at cell center.

**Sim model** (`sim_env._execute_turn`): `offset = (cp-turn_center_cprog 0.44)*turn_offset_scale 45 +
lane*turn_lane_carry 3`; Fall if `|offset|>turn_land_kill 9`; is_turn presented across
[turn_exec_early 0.15, turn_exec_late 0.60]; tilt target NOT reset (persists). Oracle + turn_gate.prompt
swipe at cprog≥0.28 from center. `turn_gate.py` gates (all phone-free, must hold): oracle/prompt survive;
PRE-FIRE / OFF-CENTER / SPAM / DOUBLE-TURN(linger) / FLASH-SWIPE(early) / **LEAN-INTO-TURN** all die.
Scorecard `turn.landing` = formula vs the device probe. `--strict` GREEN 10/10.

**Hook `lateral-accel-16e`:** accel tilt (§15) + Region2D narrow band (§15) + **is_turn latch to
cprog 0.60** (so the policy SEES the window to aim center — raw is_turn is a 1-frame flash at 0.2, too
early) + GameInput turn-buffer probe fields + **TRACK FORCING**. Turn double-turn resistance = train
with `--turn-linger 3` (linger 0 → double-turns; §16).

**TRACK FORCING (the cheap sim-vs-real comparison the user wanted) — `TRACK_MODE` in the hook:** sets
the game's `PathElement.s*` STATIC generation flags via the IL2CPP static-field API
(il2cpp_domain_assembly_open→class_from_name(PathElement)→class_get_field_from_name→
field_static_set_value; resolve exports via the module instance's findExportByName — Frida 17 removed
`Module.getExportByName`). Re-applied each Update; streams `track_statics` to verify. Modes: 'off',
'turns' (turn-dense), 'clean' (straight). **Gotchas:** turns mode needs `sAllowSidePaths=1` (turns exit
onto side paths; =0 = every turn a dead-end, even a perfect policy dies) AND still generates CURVE
pieces (pt3/4, path bends) that `is_turn` doesn't flag like T-junctions → an is_turn-based policy runs
off them. TODO: force T-junctions (pt7-10) specifically for a clean turn-reliability test.

**STATUS:** turn mechanic measured + modeled (scorecard green). ppo_tilt_v3 (linger-3, aim-center) hit
99.7% turns in-sim but on device STRAFED INTO turns → died (the controlled track exposed it). Fix in
sim (lean-into-turn punished, `turn_lane_carry`); **ppo_tilt_v4 training** with it + `--strafe-penalty
0.05`. Device eval pending (injector flaky — force-stop+relaunch, app pid can stick on HyperOS). Model
registry: `runs/MODELS.md`. Deploy baseline still ppo_faithful_stumble (device 736) until v4 verified.


This file is the durable record of how the pipeline was built and the non-obvious gotchas.
Re-read it before changing the instrumentation.

---

## 1. The game

| Fact | Value |
|---|---|
| Engine | **Unity 2022.3.62f2**, scripting backend **IL2CPP** |
| Native code | `libil2cpp.so` (42 MB) + `libunity.so`, **arm64-v8a only** (also armeabi-v7a) — **no x86/x86_64** |
| Package / launch activity | `com.imangi.templerun` / `com.imangi.unityactivity.ImangiUnityActivity` |
| Distribution | split-APK bundle: base `com.imangi.templerun.apk` + `config.arm64_v8a.apk`. **No OBB** — data is in-APK (Addressables under `assets/aa`, `assets/bin/Data`). |
| Local player id | `PlayerManager.GetActivePlayer()` is hard-coded **`1`** — every per-player record/upgrade call uses `playerId = 1`. |

### Reverse engineering
- Tool: **Il2CppDumper v6.7.46** (run via `dotnet`/its `.exe`) on `libil2cpp.so` + `global-metadata.dat`.
- Output: `il2cpp_work/output/dump.cs` (17 MB, full C# signatures with field offsets + method RVAs) and
  `il2cpp_work/output/DummyDll/Assembly-CSharp.dll` (browsable in dnSpy/ILSpy).
- IL2CPP means **no readable C# source** — only signatures/offsets. Method bodies are native arm64.
- Instance-method native ABI: first arg = `this`, **trailing arg = `MethodInfo*`** (pass `NULL`/`ptr(0)`).
  Float args follow the arm64 convention (Frida `NativeFunction` handles it).

---

## 2. Why a real phone, not an emulator (important)

- **x86_64 emulator (android-34, google_apis):** runs the game via **ndk-translation** (ARM→x86).
  Frida attaches but **cannot hook** — `Process.findModuleByName("libil2cpp.so")` returns `null`
  because the arm64 lib is mapped by the translation guest linker (`libndk_translation*.so`), invisible
  to Frida's host (x86_64) module API. Good for *playing/inspection*, useless for Frida.
- **arm64 emulator on this x86_64 Windows host:** **impossible** — QEMU2 aborts:
  `FATAL | Avd's CPU Architecture 'arm64' is not supported by the QEMU2 emulator on x86_64 host`.
  (Modern Android Emulator only runs guest-arch == host-arch.)
- **Conclusion:** Frida instrumentation requires **native arm64** — a real phone (used here) or an
  arm64 cloud host (Ampere/Graviton) for parallel instances. Docker does **not** help
  (`--platform linux/arm64` = qemu-user, no Android + Frida ptrace broken).

The android-34 x86_64 AVD `TempleRunAVD` and an unusable `TempleRunARM0` AVD still exist in the SDK.

---

## 3. Target device & Frida transport

- Device: **Redmi (model 23021RAA2Y), arm64-v8a, Android 15 (SDK 35)**, serial `e4de25a1`. **NOT rooted.**
- No root ⇒ **`frida-server` can't attach** to another app. Use **Frida Gadget** (loads inside the app).
- Frida version pinned everywhere: **17.14.0** (host `frida-tools`; gadget; server binaries).

### Gadget injection (no smali editing)
Built by `build_gadget_apk.py`:
1. **Merge** the splits → one universal APK with `APKEditor.jar m` (sets `extractNativeLibs=true`).
2. **LIEF**: add `DT_NEEDED "libgadget.so"` to `lib/arm64-v8a/libmain.so` (Unity loads `libmain` early,
   so the linker auto-loads the gadget). Add `libgadget.so` + `libgadget.config.so`
   (`{"interaction":{"type":"listen","address":"127.0.0.1","port":27042,"on_load":"resume"}}`).
3. Drop the old signature, **zipalign + apksigner** (debug key `tr-debug.keystore`, alias `trkey`, pass `android`).
4. Output: **`TempleRun-gadget.apk`**.

Result: gadget listens on **127.0.0.1:27042** inside the phone; logcat shows `Frida: Listening on ... 27042`.

---

## 4. Running it

```bash
# env for adb (Git Bash): device paths get mangled unless MSYS_NO_PATHCONV=1
ADB="D:/Users/AppData/Local/platform-tools/adb.exe"; S="-s e4de25a1"
# forward gadget + state + action ports to host
"$ADB" $S forward tcp:27042 tcp:27042
"$ADB" $S forward tcp:9999 tcp:9999      # state stream
"$ADB" $S forward tcp:9998 tcp:9998      # action sink
# launch game on the phone, then inject:
python frida/gadget_inject.py frida/temple_run_hook.js 127.0.0.1:27042
```
(All of the above is automated by `python scripts/setup_device.py` — forwards the 3 ports + injects.)

`gadget_inject.py` connects to the gadget (remote device) and loads `temple_run_hook.js`.
The script: resolves `libil2cpp.so` base, installs hooks, maxes upgrades once, serves state on 9999,
listens for actions on 9998.

### Protocol
- **State (port 9999):** newline-delimited JSON, broadcast every `POLL_MS=100` ms (**measured ~101 ms,
  9.9 Hz**), **22 fields**: the 17 original — `is_dead, is_jumping, is_sliding, is_on_ground, is_stumbling,
  has_invincibility, has_angel_wings, has_vacuum, has_boost, is_mega_boost, score, velocity, path_type,
  path_level, must_jump, must_slide, is_game_over` — plus `is_strafing`, `facing` (PathDirection N=1..W=4),
  and the turn cues `can_turn_left`, `can_turn_right`, `is_turn`.
  `velocity` = `CharacterPlayer.GetRunVelocity()` sampled on the main (Update) thread.
  `must_jump/must_slide` are derived from `PathElement.Obstacles`. `can_turn_left/right` call the game's
  own `PathElement.CanTurnLeft/CanTurnRight(facing)` on the nearest element, `is_turn` = `PathElement.IsTurn()`
  — all cached on the main thread in the Update hook (verified flipping correctly at corners/T-junctions).
- **Actions (port 9998):** one byte per action, **over a single PERSISTENT connection**.
  `0`=StrafeLeft `1`=StrafeRight `2`=Jump `3`=Slide `4`=TurnLeft `5`=TurnRight `6`=NoOp
  `7`=RestartGame (logic-only, leaves run in limbo) `8`=RunAgainClicked (start from the **game-over** screen)
  `9`=OnPlayClicked (start from the **main menu**, "JOUER"). `reset()` sends **8 then 9** — each is gated
  gadget-side to a no-op on the wrong screen, so a run starts from whichever screen the phone is on.
- **Hooks:** `GameController.Start` (force IsTutorialMode=false @0x104, TutorialID=999 @0x48);
  `PlayerPrefs.GetInt` (return 999 for key `"TR Tutorial"`); `GameController.Update` (capture instance,
  sample run velocity, drain action queue on main thread, one-shot max-upgrades);
  `DeserializeRecordManager` onLeave (max upgrades at load). All 6 actions + the full
  read→act→respond→RestartGame→new-run loop are verified on the device.

---

## 5. Key offsets / RVAs (this build only — re-dump if APK changes)

All from `dump.cs`. Field = instance byte offset; RVA = offset added to `libil2cpp.so` base.

**CharacterPlayer (instance):** IsDead `0xD8`, IsJumping `0xF8`, IsSliding `0x100`, IsOnGround `0x130`,
IsStumbling `0x10C`, IsStrafing `0x125`, HasInvincibility `0x154`, HasAngelWings `0x161`, HasVacuum `0x16C`,
HasBoost `0x174`, IsMegaBoost `0x175`, Score `0x138`, Velocity(Vec3) `0x34`, Facing(PathDirection, from
CommonPlayer) `0x78`, GameOverInterface(EndGameUI*) `0x1A0`.

**GameController (instance):** TutorialID `0x48`, GamePlayer `0x70`, IsGameOver `0xDE`,
IsTutorialMode `0x104`, NearestElement(PathElement*) `0x88`, RecordManager `0xB0`.

**PathElement (instance):** Type `0x30`, Level `0x34`, Obstacles(enum) `0x38`; directional neighbors
PathNorth `0x50` / PathEast `0x58` / PathWest `0x60` / PathSouth `0x68`
(must_jump = Obstacles∈{1 hole,4 treeJump}; must_slide = {5 treeSlide,6 flameTower}).
**PathDirection enum:** None 0, North 1, East 2, South 3, West 4 (= `CharacterPlayer.Facing`).

**RVAs:** Start `0x1269CE8`, Update `0x126C60C`, RestartGame `0x126B5B8`,
EndGameUI.RunAgainClicked `0x12BCCF0`, MainMenu.Update `0x12C94C8`, MainMenu.OnPlayClicked `0x12CA204`
(MainMenu is a singleton: static `Instance` @ field 0x0; we capture the instance in MainMenu.Update),
GetRunVelocity `0x12982D8`, Jump `0x1297B80`, Slide `0x1298768`,
StrafeLeft `0x1298874`, StrafeRight `0x1298908`, TurnLeft `0x129899C`, TurnRight `0x1298A14`,
DeserializeRecordManager `0x128AF58`, SerializeRecordManager `0x128BDC8`,
SetPlayerLevelForUpgradeType `0x128D7BC`, GetPlayerLevelForUpgradeType `0x128D568`,
GetStoreItemMaxLevel `0x128E5D8`, PlayerPrefs.GetInt(string,int) `0x24B37B8`
(Il2CppString: len@0x10, UTF-16 chars@0x14),
PathElement.CanTurnLeft(dir) `0x12E2E88`, CanTurnRight(dir) `0x12E2F44`, IsTurn() `0x12E28B0`.

**StoreItemType (upgrade indices):** AngelWings 0, HeadStart 1, HeadStartMega 2, Vacuum 5,
Invincibility 6, CoinBonus 7, CoinValue 8, Boost 9. Leveled ones (have a max>0): 5,6,7,8,9.
Maxed to: Vacuum/Invincibility/CoinBonus/Boost = 8, CoinValue = 5. (0/1/2 are consumable counts, max 0.)

---

## 6. Gotchas (cost real debugging — keep these in mind)

- **MIUI/HyperOS blocks `adb install`** (`INSTALL_FAILED_USER_RESTRICTED`) and **`adb input`**
  (`SecurityException: INJECT_EVENTS`) unless "Install via USB" / "USB debugging (Security settings)"
  are enabled (needs Mi-account sign-in). **Workarounds:** install the merged APK by tapping it in the
  Files app; **never needs `adb input`** because actions are in-process Frida calls (this is the whole point).
- **Screen sleep freezes everything → crashes training.** When the phone's display sleeps, Unity
  pauses (`onApplicationPause`) → `GameController.Update` stops ticking → the state stream stops
  broadcasting. The env's next `step()` times out on its ~3 s frame-wait and the `SubprocVecEnv` worker
  dies, taking the whole run down (`EOFError` in the parent). Worse, HyperOS may then **kill+relaunch**
  the app, which drops the injected JS agent (the embedded gadget reloads on 27042 but our
  `temple_run_hook.js` is gone) — recover by re-running `setup_device.py` for that device.
  **Fix (now automatic in `setup_device.py`, `keep_awake()`):** `svc power stayon true` works on
  permissive ROMs (older MIUI/Android 9) but **silently no-ops on strict HyperOS**, which also denies the
  secure `stay_on_while_plugged_in` write (`WRITE_SECURE_SETTINGS` revoked from shell). The path that
  actually sticks there: `appops set com.android.shell android:write_settings allow` then
  `settings put system screen_off_timeout 2147483647` (~25 days). `adb input keyevent WAKEUP` is blocked
  (INJECT_EVENTS), so an already-asleep screen needs a physical tap; the timeout fix prevents recurrence.
  **Resilience layer (so one phone's hiccup doesn't kill a multi-hour run):** `StateClient` auto-reconnects
  + `ActionClient` retries through a gap + the env waits up to `freeze_timeout` (90 s) → a transient
  freeze *pauses* the worker instead of crashing `SubprocVecEnv`. `scripts/watchdog.py` runs alongside
  training and self-heals silent devices (keep-awake → `am start` → re-inject on app restart). A phone
  parked on the **main menu** is now recovered by `reset()` itself — byte 9 (`MainMenu.OnPlayClicked`,
  "JOUER") starts a run from the menu (the menu's background runner keeps `GameController.Update` ticking,
  so the byte-9 action still dispatches; verified live). Still **not** auto-healable: a fully hung Unity
  (no ticking `GameController` at all) — needs a physical tap / app relaunch.
- **Actions need ONE persistent socket to 9998.** Per-action connect/close races the reader on close —
  only the first byte arrives. The server accept loop must be `async while(true){ await accept() }`
  (the recursive-accept pattern wedged after the first connection).
- **Restart:** `RestartGame()` (byte 7) only resets game logic and leaves the run in **game-over LIMBO**
  — the player is never placed on the track, so it never scores or dies (verified: a NoOp rollout stays
  at score 0, `ground=0`, `path_type=-1`). The real restart is **`EndGameUI.RunAgainClicked()`** (byte 8),
  reached via `CharacterPlayer.GameOverInterface` @ `0x1A0` — it starts a real, progressing run (~7-frame
  UI-transition first). **`env.reset()` sends byte 8 then byte 9** (`MainMenu.OnPlayClicked`) so it starts
  a run from the game-over screen OR the main menu — the latter recovers from the **interstitial ad** that
  occasionally drops the game back to the menu after a run (this used to stall the reset; now self-heals,
  verified live). A genuinely blocking full-screen ad with no menu underneath still can't be tapped via adb
  on MIUI — future: suppress it at the source (`EndGameUI`/ads hook).
- An **un-steered run dies in ~2 s** (hits the first obstacle). To demo/inject an action, keep the
  character alive and inject reactively.
- **Gadget `console.log` is flaky** over the remote channel — use Frida `send()` for diagnostics
  (`gadget_inject.py` prints `type:'send'` as `DBG:`).
- The upgrade-max only auto-ran via the `DeserializeRecordManager` startup hook; since we inject after
  startup, it's now also triggered **once from the `Update` hook** (main thread) via `GameController.RecordManager`.
- **Git Bash:** Windows Python doesn't share bash `/tmp` → use `cygpath -w`; quote device paths with
  `MSYS_NO_PATHCONV=1`.

---

## 7. Files & tool paths

After the repo reorg, source lives in subfolders (the big artifacts below stay at root, git-ignored):

| File | Purpose |
|---|---|
| `frida/temple_run_hook.js` | Frida agent: hooks, state server (9999), action server (9998), upgrade-max |
| `frida/gadget_inject.py` | connect to gadget @27042 and load the agent |
| `frida/inject.py` | same, but attach by pid (for a rooted/frida-server target) |
| `frida/diag.js` | one-shot module/arch diagnostic |
| `tools/build_gadget_apk.py` | merge splits + LIEF DT_NEEDED + add gadget → unsigned patched APK |
| `src/templerun_rl/` | the Gymnasium env (`env.py`, `client.py`, `obs.py`, `constants.py`) |
| `scripts/` | `setup_device.py`, `train.py`, `evaluate.py`, `smoke_test.py` |
| `docs/` | `SETUP.md`, `GADGET.md`, `OFFSETS.md` |
| `TempleRun-gadget.apk` | the installable gadget build (git-ignored) |
| `il2cpp_work/output/dump.cs` | full IL2CPP signature dump, source of all offsets (git-ignored) |

- **Android SDK:** `D:\Users\AppData\Local` (cmdline-tools installed manually under `cmdline-tools\latest`).
- **JDK:** `D:\Logiciel\Android Studio\jbr` (no system Java). **Python:** `G:\python312\python.exe`.
- **Frida:** 17.14.0. Build-tools: `D:\Users\AppData\Local\build-tools\37.0.0` (apksigner/zipalign).
- arm64 `frida-server` and `frida-gadget` are pulled from `github.com/frida/frida/releases/<ver>`.

---

## 8. Status

**Fully validated end-to-end on the Redmi (zero `adb input` — all in-process Frida calls):**
- 19-field state stream on 9999.
- All 6 actions confirmed: Jump/Slide/StrafeLeft/StrafeRight via their state flags, TurnLeft/TurnRight
  via `facing` change (note: forcing a turn mid-path kills the run — turns only make sense at junctions).
- Full loop: read state → Jump (response) → reset → fresh run. NB: in the env, reset uses **byte 8
  (RunAgainClicked)** — byte 7 (RestartGame) only resets logic and leaves the run in limbo (see §6/§9).
- All leveled upgrades maxed (8/8/8/5/8) and persisted to disk.
- Tutorial neutralized (Start patch + PlayerPrefs `"TR Tutorial"`→999).

**Built (this repo):** the Gymnasium env `TempleRun-v0` (`src/templerun_rl/`) wrapping the two sockets
(Box(18) obs, Discrete(7) actions, score-delta reward, **RunAgainClicked/byte-8 reset** — RestartGame
alone leaves the run in limbo), plus SB3 PPO `scripts/train.py` / `scripts/evaluate.py` /
`scripts/smoke_test.py`, packaging, and docs. Smoke-tested live (50 random steps → reward, deaths,
fresh resets). See `README.md`.

Possible next steps: robust reset that suppresses the occasional interstitial ad in `EndGameUI`;
richer observation (one-hot enums, a `valid` flag for the -1 sentinels); reward shaping.

---

## 9. The RL environment (`TempleRun-v0`)

`src/templerun_rl/` wraps the two sockets as a Gymnasium env (`gym.make("TempleRun-v0")`). **One physical
phone = one env**, real-time ~10 Hz, no simulator, no parallelism. Defaults live in `constants.py`;
encoding in `obs.py`; the env in `env.py`.

### Observation — `Box(0.0, 1.0, (21,), float32)`
21 features, all normalized to [0,1] by `state_to_obs()`. The **raw 22-field state dict is always in
`info["raw_state"]`**. Raw `score` is **excluded** from the obs (unbounded/accumulating → it drives the
reward instead).

| idx | feature | transform |
|---|---|---|
| 0–16 | the 17 bool flags | `float(bool)` → 0/1 — is_dead, is_jumping, is_sliding, is_on_ground, is_stumbling, has_invincibility, has_angel_wings, has_vacuum, has_boost, is_mega_boost, is_strafing, must_jump, must_slide, is_game_over, **can_turn_left, can_turn_right, is_turn** |
| 17 | velocity | `clip((v-100)/120, 0, 1)`  (VEL_MIN 100, VEL_MAX 220) |
| 18 | path_type | `0 if <0 else clip(pt,0,37)/37` |
| 19 | path_level | `0 if <0 else clip(pl,0,4)/4` |
| 20 | facing | `0 if <0 else clip(fc,0,4)/4`  (PathDirection N=1..W=4) |

The `-1` "none" sentinel on the three enums maps to `0.0` (aliases real value 0 at junction edges —
acceptable for v0; a `valid` companion flag is the documented future tweak).

### Action — `Discrete(7)`
Policy emits **0–6**: `0` StrafeLeft · `1` StrafeRight · `2` Jump · `3` Slide · `4` TurnLeft · `5` TurnRight ·
`6` NoOp. Bytes `7`/`8`/`9` are **reset-only** (never sampled by the policy): `7` RestartGame (limbo),
`8` RunAgainClicked (game-over restart), `9` OnPlayClicked (main-menu start). `reset()` sends 8 then 9.
The env raises if asked to `step()` an action ≥ 7.

### Reward
```
reward = (score - prev_score) * reward_scale          # per step
if terminated (is_dead or is_game_over): reward -= death_penalty
```
Defaults `reward_scale = 0.01`, `death_penalty = 1.0`. Empirically: a typical alive step ≈ **+0.27**
(~27 score/step at run start), coin/boost pickups give larger jumps (e.g. +0.5–1.6), and the terminal
step is ≈ **−1**. `prev_score` is set from the reset frame (~162), so the first delta is well-behaved.

### Episode / step mechanics
- **`reset()`** — send byte 8 (RunAgainClicked, from game-over) **then** byte 9 (OnPlayClicked, from the
  menu); only the one matching the current screen fires (each is gated gadget-side). Poll until the player
  is actually on the track (`is_on_ground or path_type ≥ 0`) for `reset_settle_steps` (3) consecutive
  frames within `reset_timeout` (5 s), with one retry; returns a *started* run (skips the UI-transition
  limbo). This makes reset robust to the game landing on the main menu (e.g. after an interstitial ad).
- **`step(a)`** — send the action byte on the persistent socket; wait one control tick (`step_dt` 0.1 s,
  reading the newest frame *after* the action — never stale); `terminated = is_dead or is_game_over`;
  `truncated` at `max_episode_steps` (1000).
- **Adaptive episode cap (opt-in, default off):** `max_episode_steps` is a *ceiling*, not the usual run
  end — almost every episode `terminated`s by death first, so the cap is unreached early on. With
  `grow_episode_steps=True` the env **ratchets** the cap up — only on a *pure truncation* (alive at the
  limit, `truncated and not terminated`), multiplying by `max_episode_steps_growth` (1.5) up to
  `max_episode_steps_ceiling` (6000 ≈ 10 min). Never shrinks (incl. on death). Rationale: surviving the
  cap proves the agent can play that long, so unlock more of the *faster* late game (the runner keeps
  accelerating) instead of clipping the best runs. Growth is surfaced in `info["episode_cap_grew_to"]`;
  `info["max_episode_steps"]` carries the current cap every step. `_grow_episode_cap()` does the bump.
- **Tunable ctor args:** `step_dt, max_episode_steps, reward_scale, death_penalty, reset_settle_steps,
  reset_timeout, connect_timeout, grow_episode_steps, max_episode_steps_growth,
  max_episode_steps_ceiling, auto_connect` (+ `host`/ports). `train.py` exposes
  `--grow-episode-steps`/`--max-episode-steps-growth`/`--max-episode-steps-ceiling`.
- **Connection layer (`client.py`):** `StateClient` reader thread keeps only the latest frame (`seq`
  counter); `ActionClient` holds one persistent `TCP_NODELAY` socket.

---

## 10. Headless sim + policy training — sim-to-real findings (2026-06-18)

**Pipeline:** `TempleRunSim-v0` (`src/templerun_rl/sim_env.py`) is a fast headless model of the game
(track gen, 3 lanes, obstacles, junctions, 3 altitude **zones**, velocity curve, domain randomization).
Train PPO in the sim (`scripts/train_sim.py`, CPU, minutes), deploy the `.zip` on the phone via the same
`TempleRunEnv` + Frida hook. **Reward = distance-delta** (DistanceTraveled@0x124), not score (score is
multiplier-inflated). The sim is essentially SOLVED (a good policy hits 100% turns / 10,000m on a
normal-window config) — so the on-device shortfall is **sim-to-real fidelity**, not net capacity.

**Deployment model: `runs/ppo_best.zip` (= lingerclean)** — MLP [128,128], action-history K=4, trained
with turn-linger + DR + zones + strafe-dock. Each model has a sidecar `<name>.json` (frame_stack, K,
arch) so `scripts/eval_real.py <model>` auto-configures the obs. **Eval with `--turn-cooldown 3`**.
On device nzqwrkbyfeey89fa (2026-06-19): **median 607, best 1610m, 8/8 stable, NO strafe / double-turn /
double-jump**. Remaining deaths = late-first-turns at the window edge.
NOTE: turn-linger (harder task) + ent_coef 0.04 let the **strafe-spam back through** (linger2 strafed
SL:17-43/run, which breaks turns off-center). FIX: dock STRAFE via `idle_action_penalty` (strafe is never
needed) in `reward.py`/`env.py`; retrain `--idle-action-penalty 0.08` → strafe 0%, per-turn held ~99.3%.

### What MOVED the on-device number (best real run 393 → 1258+ native, 1643 with cooldown)
1. **The double-turn was THE dominant on-device turn-death.** On the phone `is_turn` stays True ~2
   frames at a junction (facing-change lags the input); with the 1-frame obs lag the policy fires the
   turn on BOTH frames — the 1st lands on the corner (good), the 2nd OVER-ROTATES the character off the
   path → Fall (facing goes e.g. N→E→S). The plain sim consumed `is_turn` instantly, so the policy was
   never tempted → never learned to resist. **Verified live** (capture) + reproduced in-sim.
2. **TURN COOLDOWN (host-side, no retrain): `env.py` `turn_cooldown`** — after a turn is sent, drop
   further turn actions for N steps (NoOp). `--turn-cooldown 3` is best (2/4/5 much worse). Instant
   fix: median 281→~590, best 393→1643.
3. **TURN LINGER (principled, sim-to-real): `sim_env` `turn_linger_frames`** — force `is_turn` True for
   N frames after a turn (a 2nd turn during them = over-rotation death, death_type 8). The policy then
   MUST learn (via action-history memory) to turn ONCE. In-sim proof: a non-linger model dies 40/40 to
   over-rotation on a linger env; the linger-trained model survives 40/40. On device: best 1258 at
   **cooldown=0** (native single-turns, no crutch). This is the real fix; cooldown is the patch.

### What did NOT help (ran the experiments, drew the conclusion)
- **Bigger / deeper nets HURT.** Sweep (per-turn on tight window): [128,128]=98.95% > [256,256]=94.77%
  > [128,128,128]=no gain. Most-params config was worst. Sim turn-reliability is timing-limited, not
  capacity-limited. **Keep 128×128.**
- **Frame-stacking HURTS** (fs=1 98.95% > fs=2 97.46% > fs=4 97.52%): the base obs already has the
  `next_is_turn` lookahead, so stacking past frames adds dilution not anticipation. `--frame-stack 1`.
- **Tightening the training turn window** (lingertight, dr_turn_window_frac[0.28,0.45]) did NOT beat the
  default-window linger2 (99.03% vs 99.33% per-turn on the linger+tight metric).
- **Modeling action lag (`action_delay=1`) is UNFAIR** even with linger — a memory+lag-aware oracle dies
  30/30 (the 1-step lag overruns the tight DR window). Don't re-add without re-widening the window.

### Key reward/obs knobs (all in `constants.py` defaults; train_sim flags)
- `action_history_len` (K): one-hot of last K actions appended to obs (memory → curbs double-turn; needs
  ent_coef≈0.04 to avoid a NOOP-collapse). `action_cost`=0.06 kills the strafe-spam high-ent_coef causes.
- `redundant_action_penalty`=0.12 (kills the double-JUMP: reward anticipatory jump, dock the trailing one).
- `frame_stack` (N), `turn_linger_frames`, `--sim-config '{json}'` (arbitrary sim overrides).

### Metrics / scripts
- `scripts/score_turns.py` — per-TURN reliability + 10k-reached on a tight-window config (phone proxy).
- `scripts/score_hard.py` — survival on the hard held-out config (arch discrimination).
- `scripts/exp_phone.py <model> --cooldowns a,b` — on-device dist + death-cause classification (turn/
  obstacle/zone). `scripts/eval_real.py` — plain on-device eval (auto-reads sidecar).
- Current state: linger2 ~99.3% per-turn on the hardest sim (8460 junctions), max sim run 11,500m. The
  last mile to 99.99% (needed for a reliable 10,000m) is the late-first-turn at the window edge — likely
  needs a hook-side fix (latch is_turn=0 on turn-byte RECEIPT, or a distance-to-junction obs).

### Device note
New phone **nzqwrkbyfeey89fa** (replaces topaz e4de25a1). Same crash-loop after ~30-60 min of eval;
recover with clean_inject + force-stop + relaunch + setup_device (see [[launch-recipe-multi-phone]]).

## 11. Host-side action-timing assists — the big on-device wins (2026-06-19)

Re-diagnosed the on-device deaths from scratch with a NEW hook capture: `lane` (signed lateral
offset from path center, + = right of travel) + `cell_progress` (longitudinal position within the
nearest cell, 0..1), both computed from `CharacterPlayer.GetPosition2D()` (Vector2, RVA `0x12E0E9C`,
struct-by-value via libffi: retType `['float','float']`) and `PathElement.Rectangle` (@`0xA0`, a
`Rect{xmin,zmin,w,h}`; grid cell = 60×60, `GridX/GridZ=60`). These are now broadcast as state fields
`lane`/`cell_progress` (forward-compatible: existing 26-d obs ignores them). The deaths were NOT what
§10 assumed — and two **host-side, no-retrain** fixes in the hook removed the two dominant causes:

**1. Turn-buffering (commit 47cf1ab, the dominant death — 9/12 runs).** The off-center-after-turn
fall was a TIMING bug, not over-rotation: the policy fires the turn on the FIRST `is_turn` frame
(cell_progress ~0.1) and the turn arc from there swings the runner **~12 units off-center → off the
exit path → fall**. Measured `progAtTurn → postLatMax`: 0.05–0.22 → 6–15 (falls); 0.42–0.56 → 1–3
(centered); 0.90+ → 0. Fix: don't execute a TurnLeft/Right byte immediately — LATCH it (`pendingTurn`)
and fire the native turn from the Update hook once `cell_progress >= TURN_FIRE_PROG=0.40` (cell
center). A turn issued on a pure straight (neither `is_turn` nor `next_is_turn`) is dropped; buffered
turns expire after 20 frames. Validated: 37 turns fired at progress 0.40–0.59, post-turn offset **0.0
across 35 turns** (was 6–15), turn-deaths **→ 0**. The host-side `turn_cooldown` is now redundant but
kept (composes fine).

**2. Crevasse jump-assist (commits 72396a3 / 8b, the #2 death after turns).** The zone/altitude
transition (`ObstacleType.LevelChange=7`, path_type 33–36 = `SingleLevelStepDownCap A/B Horz/Vert`) is
a **~2-cell GAP**, not a single must-jump cell as the sim models it. The policy reacts to the 1-cell
lookahead and jumps too EARLY → airtime ends mid-gap → falls in (or never jumps). Deterministic
response, so handle it host-side: near a LevelChange (nearest or forward-neighbor obstacle == 7),
SUPPRESS the policy's jump and fire ONE native Jump when the pre-gap cell reaches
`cell_progress >= JUMP_FIRE_PROG` so airtime spans the gap. **0.70 fired too early and missed the
wider Horz (33/34) gaps; 0.90 (fire near the gap entry, at the hook's 60 Hz Update precision) clears
them.** One jump per crevasse + a safety late-jump. Validated: zones that were the ~580 m death wall
now clear; runs reach 1600+.

**3. Narrow strafe-assist (open-loop, facing-aware).** NarrowLong pieces (path_type 17–30) have a
one-lane strip on an ABSOLUTE compass side; the runner (sits at its home/center lane, never strafes)
falls off the edge. The strip side per subtype: NarrowLong{Left=West / Right=East} (Vert/NS pieces),
{Up=**South** / Down=**North**} (Horz/EW pieces — the Up/Down→S/N mapping is MEASURED, counter to the
name). The strafe (L/R) to reach a compass side depends on heading (`narrowStrafeDirFor`). Open-loop:
strafe ONCE to the side on approach, ONCE back to center on exit — a CLOSED-LOOP lane controller
OSCILLATED catastrophically (noisy mid-strafe `lane` reads + recenter feedback → runner swung −6..+6,
151 strafes; never do closed-loop). NarrowSmall 17/18 = center (survives). Lane geometry: home/center
~ −1.3, one strafe ≈ one lane ≈ 3.7 units.

**4. Obstacle reactor (safety net).** The policy clears plain holes/trees ~95% but misses some — esp.
a hole right AFTER a turn (busy turning → reacts a frame late; the dominant residual death). If the
policy hasn't jumped/slid by `cell_progress >= 0.72` of the pre-obstacle cell (or it's already the
nearest must-jump/slide cell) AND the runner isn't already jumping/sliding, the hook fires it.
Excludes wide gaps. Fires rarely (≈2–5/run) — never doubles a good policy play.

**Net on device (ppo_best, unchanged weights, full assist stack, turn-cooldown 3):** median ~280→~1650,
**best 2253** (was ~600/1610 at session start). All four obstacle families now handled host-side.
Remaining deaths are rare edge cases (a specific narrow subtype mis-mapped, an after-turn hazard the
reactor missed, an occasional fall). HOOK_VERSION `reactor-9f`. **Path to 10,000 m:** the limiter is now
the COMPOUNDING per-event death rate (≈99% per event still caps a 100-event run); options are a fuller
deterministic obstacle autopilot (suppress policy jumps too — but the user wants a *policy*, weigh it)
or a retrain with the new `lane`/`cell_progress` obs + a faithful 2-cell-gap / narrow-fall sim.

## 12. NN-DECIDES retrain — no hook rules (2026-06-19, user-mandated)

**User: the neural network must take the decisions; NO hardcoded hook rules ("a hook playing would
kill the project").** So all §11 host-side ACTION assists are DISABLED (the hook only reads state +
executes the policy's raw action bytes). The §11 diagnosis became the SIM SPEC so the policy learns it.

New OBS (perception, not rules; `obs.py` OBS_LEN 26 -> 29; hook streams them): `cell_progress` (time
turns), `lane` (3-lane bucket via `lane_to_obs`, for narrows), `next_wide_gap` (2-cell gap ahead).
New SIM death models (`sim_env.py`):
- **turn timing**: turn before `turn_early_frac` (0.30) OR after `turn_late_frac` (0.62) cell_progress
  -> off-center Fall (death_type 9). The policy must aim the window CENTER (cprog ~0.45) via the
  cell_progress obs — aiming the late edge + the real ~half-frame obs lag is what made turns fall.
- **double-turn**: `turn_linger_frames` (use **3**) keeps is_turn True post-turn; a 2nd turn during it
  = over-rotation Fall. The policy learns (via action-history K=4) to NOT re-turn. **linger 2 still
  double-turned on device; linger 3 fixed it (0 double-turns / 19 turns).**
- **wide gaps** (long holes 12/13/15/16, zone caps 33-36): a single must-jump cell needing a FRESH
  jump on entry (`jump_t >= wide_gap_min_airtime` 0.30) so airtime spans it; jumping a cell early
  falls in. `next_wide_gap` obs signals it.

**Training recipe that works (`ppo_nn_best.zip` = turn4b @1.5M):** `--net-arch 128,128
--action-history-len 4 --frame-stack 1 --turn-linger 3 --ent-coef 0.04 --action-cost 0.06
--idle-action-penalty 0.12 --turn-spam-penalty 0.14 --redundant-action-penalty 0.12` (DR on).
Pitfalls: ent-coef 0.03 -> NOOP-collapse (greedy policy stops turning, dies at the 1st junction at a
fixed step count); keep **0.04**. PPO can COLLAPSE late (turn4b 910@1.5M -> 376@1.8M) — KEEP THE BEST
CHECKPOINT, not the highest. Oracle (`train_sim._oracle`) must turn at cprog>=0.32 (aim center) +
late-jump wide gaps, else its baseline reads as un-survivable.

**On device, NN-only (eval_real --turn-cooldown 0, no assists):** median 235 (old policy) -> **median
1210, best 2182** (turn4b). Turns land at cprog 0.31-0.59 (aim-center), 0 double-turns, wide-gaps
cleared. Nearly matches the §11 rule-based assists (1605/2253) but the POLICY decides everything.
**Remaining deaths = NARROWS** (path_type 17-30; ~half the deaths — the policy doesn't know the safe
lane) + the odd missed slide/zone. NEXT: empirically map each narrow subtype+facing -> safe lane (host
guesses were wrong), add a `narrow_safe_lane` obs + sim narrow-fall model, retrain. HOOK_VERSION
`nn-only-obs-10b`.

NOTE: HOOK_VERSION is now `crevasse-assist-8b`. The capture/analysis tooling lives in `/tmp`
(`cap_turn.py` logs the stream incl. `lane`/`cell_progress`; `classify_deaths.py` buckets deaths
turn/obstacle/zone/after-turn-hazard). Calibration method that works: capture at 10 Hz, correlate the
action's `cell_progress` with the outcome (off-center swing / cleared), pick the fire-progress.

## 13. STEP-BACK: turns are a STRUCTURAL failure — the high-fidelity-sim roadmap (2026-07-01)

Goal reframed by user: **a high-fidelity sim to train different policies** (distance first, coins modeled).
40+ policies trained project-wide, ALL fail turns → NOT a training/hyperparameter problem, it's
**structural** (obs / sim / reward). We kept tuning the sim turn window + retraining inside a loop we
never validated. See memory [[turn-investigation-roadmap]], [[device-calibration-2026-07-01]],
[[lateral-control-recenter-mechanic]].

**What we got RIGHT this session (real progress):**
- **Controls understood + device-CONFIRMED:** lateral = CONTINUOUS TILT (recenters to center 0 when not
  steered; edges ±7), NOT lock-lanes. A mistimed swipe / over-tilt = a survivable **STUMBLE** (velocity
  ×0.85 + recenter; 2nd stumble within `stumble_kill_window` → death). `stumble_kill_timer` counts UP =
  time-since-stumble. The OLD sim instant-killed on mistimed turns — the core fidelity bug.
- **Sim reworked** (sim_env.py): continuous tilt+recenter, wall-stumble + 2-strike death, step-up jump
  must-fire-late. obs.py lane range symmetric ±8 center 0. **Result: mid→high crevasses now RELIABLE on
  device (user-confirmed, first time).** Deploy candidate `runs/ppo_faithful_stumble_1300000_steps` (device median 736).
- **Instrumentation:** hook `obs-probe-13a` streams the game's own metric signals — `over_turn_area`
  (IsPointOverTurnArea, the REAL turn-window test), `dist_to_turn`, `dist_to_change_*` (crevasse),
  `stumble_kill_timer`, real `death_type` enum, + COINS (`has_coins`/`coin_loc_start/end`/`coin_count`/
  `next_*`). Coins matter: **a tilt = collect coins OR ride a narrow** (player lane follows the coin lane
  L→−4/C→0/R→+5) — so "tilt→narrow" is false-positive-prone; coin placement disambiguates + lets a
  coin-collecting policy be trained. New scripts: probe_fields.py, probe_turn_mistake.py, capture_calib.py
  (+coins), log_stream.py (+flush). Human calib capture: `captures/human_calib_v2.jsonl` (a 14,956 m run).

**THE TWO ROOT GAPS on turns (why they still fail):**
1. **The policy times turns off a PROXY** (`is_turn` bool + `cell_progress`), NOT the game's real
   `over_turn_area` (instrumented but never fed to the obs or analyzed).
2. **No rigorous sim-vs-real turn comparison** (fidelity_audit spot-checked, never for turns); never read
   the game's turn CODE.
LESSON: tightening the sim turn window BACKFIRED (ppo_stumble2 regressed 736→193 — over-fit a tight
window that doesn't match device). The answer is the REAL SIGNAL + fidelity validation, NOT window tuning.

**ROADMAP (in order, stop blind retraining):** (1) read the game turn mechanic in dump.cs (TurnLeft/
IsPointOverTurnArea/TurnRect geometry/GetPathElementIfPointOverTurn2D) → (2) capture real turns on device
logging over_turn_area/is_turn/cell_progress/dist_to_turn/facing → (3) compare sim-vs-real turns
frame-by-frame → (4) fix the obs to feed the REAL turn signal + re-model the sim turn → (5) retrain ONCE.

**STEPS 0 + 1 DONE (2026-07-01, 2nd session — all MEASURED from the capture, NO phone):**
- **Step 0 (harness) BUILT:** `scripts/fidelity_audit.py` gained **section E = TURN WINDOW** — measures
  is_turn cprog distribution + run length + facing-change cprog + next_is_turn window for BOTH the human
  capture AND an oracle sim rollout, side by side. This is the phone-free gate we never had.
  Run: `.venv/bin/python scripts/fidelity_audit.py captures/human_calib_v2.jsonl`.
- **`over_turn_area` IS DEAD (roadmap premise CORRECTED — do NOT feed it to the obs):** measured on
  `human_calib_v2` (365 executed turns) it is True in only **3/7441** frames; `dist_to_turn`∈{750,850,1000}
  + `dist_to_turn_end`≡100 are generation-pacing CONFIG, NOT live countdowns. The reliable turn signal is
  the EXISTING obs — **`is_turn` + `next_is_turn` + `can_turn_left/right`** (already in the obs; no obs
  change needed).
- **REAL turn window MEASURED:** is_turn is a **~1-FRAME flash at cell START** (cprog med 0.23 / p90 0.38,
  91% 1-frame). Human swipes the instant it appears (cprog med 0.25); facing flips ~1 step later (0.68);
  next_is_turn ≈ 3-frame warning. Old sim taught "aim centre [0.30,0.62]" → on device is_turn is GONE by
  0.40 → policy waited, missed the flash, ran off a MANDATORY turn → Fall. That is the structural death.
- **RE-MODEL DONE in `sim_env.py` (validated on the harness):** (1) DELETED the off-center
  `turn_early_frac`/`turn_late_frac` death (early turns are how humans play). (2) `turn_window_frac`
  0.70→**0.45** so the sim reproduces the 1-frame flash. (3) **BUFFERED early swipe** (`turn_buffer_cells`,
  `_attempt_turn` rewrite + `_execute_turn` + `_pending_turn` fired in `_cross_into_next_cell`): a swipe on
  the approach cell (next_is_turn) is latched and fires on entering the turn area (the game's
  HandleDelayedTurning). (4) ONLY turn death = miss a mandatory turn → Fall; swipe on a straight →
  survivable stumble. Oracle turns on the FIRST is_turn frame + pre-fires on next_is_turn.
- **BEHAVIORAL GATE PASSED:** re-modeled sim, 120 eps each — prompt/oracle survive 120/120; a **late-aim
  policy (the OLD center-aim behavior) dies 120/120 at the 1st turn (Fall)**. The sim now rewards prompt
  turns + punishes late aim = the device reality.

**RESUME — NEXT: RETRAIN ONCE, then device-eval (step 1 tail).**
- Recipe (CLAUDE.md §12) on the re-modeled sim, but **use `--turn-linger 0`** (device is_turn = a
  measured 1-frame flash; and BUFFERING already prevents double-turns — the pre-fired turn is consumed on
  cell entry so is_turn never re-shows. linger 3 is now INCOMPATIBLE: it kills the prompt+buffer oracle
  80/80 by re-firing during the linger → repeated stumbles. Verified: linger 0 oracle survives 80/80).
  Keep the BEST checkpoint (PPO collapses late). Then `scripts/eval_real.py <model> --turn-cooldown 0`.
  Success = turn-Falls drop on device (classify via real `death_type`); target past the ~1000 m turn wall.
- THEN step 2 (crevasse jump-clear window: the sim `stepup_min_takeoff_prog` gate vs measured takeoff)
  and step 3 (narrows: audit section D shows sim under-represents narrows 1500-3000 m, real 11.5% vs sim
  4.6% — read the narrow mechanic in dump.cs + measure, then model). Narrows are ~half the residual deaths.

Rule going forward: **measure/read the real mechanic + validate on the offline harness BEFORE retraining.**
No more blind train→eval→tweak loops. Deploy model so far: `runs/ppo_faithful_stumble_1300000_steps`
(device median 736, crevasses reliable).

## 14. SESSION 2 (2026-07-01) — the turn "problem" was really LATERAL. RESUME HERE (2026-07-02)

**The whole arc this session (turns → really lateral):** re-modelled turns (window flash, buffering) and
retrained TWICE — both REGRESSED on device (226, 201 < the 736 baseline). Root-caused each with a device
capture (not guesses):
1. Buffering (pre-fire on next_is_turn) was a SIM EXPLOIT — safe in-sim, on device it turned off-path into
   the water. Removed it. Built `scripts/turn_gate.py` (adversarial gate: PRE-FIRE/SPAM/OFF-CENTER must DIE).
2. Un-buffered still regressed → the policy turned OFF-CENTER (lane ~4.7) and fell off the path. Added a
   lateral-HOLD model + off-center-turn Fall + oracle centering → device 295 (better, still < 736).
3. **THE REAL ROOT (device probe `scripts/probe_strafe.py` + dump.cs):** our discrete `StrafeLeft/Right`
   call `CharacterPlayer.StrafeLeft()`/`Right()` which SET a PERSISTENT `PlayerXOffset` (±7, slam-to-edge
   + HOLD). px is only 0 at run start → **the policy can NEVER return to lane 0 after strafing → off-center
   at the next turn → Fall.** The human uses ANALOG tilt (`GetAccelerometerForceX`; level phone → recenter),
   which we never drove. THIS is why 40+ policies + this session failed turns. See [[lateral-control-recenter-mechanic]].

**USER DECISIONS:** (a) the character MUST be able to strafe (don't train it away); (b) add FINER lateral
control — **3 levels: Left / Center / Right**.

**DONE (code, NOT device-verified — phone unplugged mid-verify):**
- Hook `frida/temple_run_hook.js` = **`lateral-tilt-14b`**, `LATERAL_TILT=true` (line ~179). Actions now:
  byte0=hold LEFT, byte1=hold RIGHT, byte6/NoOp=LEVEL (recenter). The hook OWNS the lateral: sets a target
  px (LEFT→+reach, RIGHT→−reach; measured `lane = −px`) and GLIDES `PlayerXOffset` toward it in the Update
  **onLeave** (must be onLeave — an onEnter write is overwritten by the game's char-update; the first attempt
  14a failed for this reason). `LAT_REACH=7`, `LAT_GLIDE=0.30`/Update-frame. All other assists still false.
- Sim/harness from step 2 in place (lateral hold `lane_recenter_time=100`, `turn_center_tol=4.0`, un-buffered
  turns, `turn_gate.py`). Model `runs/ppo_lateral_best` (device 295). Calibrated strafe reach (`strafe_reach
  7`, `strafe_glide_time 0.6`) but that models the OLD coarse strafe — will change once tilt is verified.

**RESUME STEP 1 — VERIFY THE TILT CONTROL ON DEVICE (make-or-break, do FIRST):**
- Recover injector: `adb -s 3f80a3a shell am force-stop com.imangi.templerun` → `am start -n
  com.imangi.templerun/com.imangi.unityactivity.ImangiUnityActivity` → wait ~12 s → forward ports 27042/
  9999/9998 → `setsid nohup .venv/bin/python frida/gadget_inject.py frida/temple_run_hook.js
  127.0.0.1:27042 &` (setsid so it survives; injector is FLAKY after kill cycles — force-stop+relaunch is
  the fix). Verify `_hv=lateral-tilt-14b` on :9999.
- Run `python scripts/probe_strafe.py --runs 6 --out captures/strafe_probe_tilt.jsonl` → CONFIRM: StrafeLeft
  glides lane toward −7, StrafeRight toward +7, **NoOp glides lane back to 0 (the new centering)**. If the px
  write does NOT move the runner (lane unchanged), fall back to hooking `GetAccelerometerForceX` (TouchInput
  RVA 0x12AD500) to return a commanded tilt force instead of writing px.

**RESUME STEP 2 (after verify): re-model the sim lateral as ANALOG-TILT + retrain.**
- Sim `sim_env.py`: a strafe glides `_lane` toward ±reach WHILE the tilt is held; NoOp glides back toward 0
  (recenter) — this REVERTS toward the original recenter model (which was right for analog tilt, wrong for
  the discrete-persistent strafes). Calibrate the glide rate to the device probe. Keep off-center-turn Fall.
- Add LATERAL fidelity to the harness (we only ever validated TURNS): replay the strafe script in the sim,
  diff the lane trajectory vs `captures/strafe_probe_tilt.jsonl`. Then `turn_gate.py` must still PASS.
- Retrain (`train_sim.py`, recipe §12, linger 0). Device-eval `eval_real.py --turn-cooldown 0`. Target > 736.

Hook re-inject one-liner (verify `_hv` on :9999 after): `setsid nohup .venv/bin/python
frida/gadget_inject.py frida/temple_run_hook.js 127.0.0.1:27042 &`. Device serial **3f80a3a**.

## 15. THE FIDELITY SCORECARD — the cheap feedback loop (2026-07-02). RULE: scorecard before retrain.

**WORKFLOW RULE (user's #1 gap, now closed): no retrain, no deploy, no sim-knob commit without
`fidelity_scorecard.py` green. The retrain gate is `--strict` green (0 FAIL, 0 SKIP).**

- `scripts/extract_device_reference.py` — freezes device truth from captures into
  **`captures/device_reference.json`** (committed): turn window stats, velocity curve, event rates
  per band, stumble mechanics, narrow safe-lane map (prefilled from human play), crevasse takeoff
  bands, lateral tilt dynamics (`--lateral <probe>`). Re-run only when new captures land.
- `scripts/fidelity_scorecard.py [--json out] [--fast] [--strict]` — one command, phone-free, ~4 min:
  9 checks vs the reference (velocity MAE, turn.window, turn.gates via turn_gate.run_gates(),
  lateral.step = REPLAY of the probe_strafe TILT_SCRIPT in-sim, narrow.rates / .semantics
  (per-strip aim-survives/wrong-dies) / .map (hook NARROW_SIDE vs measured), crevasse.curve
  (jump-at-cprog sweep vs device band), stumble.window). PASS/FAIL/SKIP; SKIP ≠ PASS.
- First run already caught: stumble slowdown overwritten next tick by the velocity curve (fixed:
  `_stumble_vel` + `stumble_recover_rate` 1.0 u/s measured), narrow under-density (onset 1600,
  prob 0.7 calibrated), narrow_lane_tol 2.2→3.0 (device band ±3), and reproduced the §11 crevasse
  too-early-jump death offline (sim window [0.5,0.6] vs device fatal~0.82/clear~0.92 — OPEN, Phase 4).

## 16. DOUBLE-TURN REGRESSION + FIX (2026-07-02, user-caught)

**Symptom (user, device eval of ppo_tilt_v1):** turns unreliable — the policy swipes MULTIPLE times
in one turn → over-rotation Fall. Device capture confirmed: 12 double-turn bursts, **8/8 deaths at
turns**, all at velocity ~150-163 (mid-speed early junctions), dying at dist 180-540.

**Root cause = trained with `--turn-linger 0`.** In the sim a turn instantly consumes is_turn (goes
False same frame), so the policy NEVER saw is_turn still True post-turn → never learned to resist a
2nd swipe. Device is_turn is True for **~2 frames** at a junction (facing-change lags the swipe;
measured 91% 1-frame / 8% 2-frame, but at mid-speed more often 2) → the policy fires on BOTH → double.
This is the §10 #1 mechanism. The model that NAILED turns (`ppo_nn_best`, median 1210, "0 double-turns
/19 turns") trained with **`turn-linger 3`** — the value §12 found necessary ("linger 2 still
double-turned on device; linger 3 fixed it"). The §13/§14 "use linger 0" note was for a buffering
variant that no longer exists; following it was the regression.

**FIX (retrain `ppo_tilt_v2`, `--turn-linger 3`):**
- The sim forces is_turn True 3 frames post-turn; K=4 action-history + `turn_spam_penalty 0.14` +
  `redundant_action_penalty 0.12` (both ALREADY in the recipe) now get exercised → the policy learns
  to fire ONCE. Yes, the obs HAS action history (K=4) and the recipe HAS a turn-spam penalty — they
  just never trained against the double-turn because linger was 0.
- Oracle + turn_gate `prompt` made linger-safe via RISING-EDGE turn detection (turn only on
  is_turn False→True; the held linger frames are not a fresh junction). `train_sim._oracle_mem` /
  `turn_gate._prompt_mem`.
- New adversarial gate `turn_gate.double_turn` (turn on EVERY is_turn frame) MUST die — verified:
  dies 0/80 under linger 3 while oracle/prompt survive 80/80. This guards the exact regression.
- Hook is_turn LATCH (16a, held is_turn to cprog 0.45) DISABLED → `lateral-accel-16b` uses RAW is_turn
  (natural 1-2 frame). The latch ADDED is_turn frames = more double-turn temptation; ppo_nn_best nailed
  turns on raw is_turn. Double-turn is a TRAINING fix (linger 3), not a perception one.

**LATERAL = ANALOG ACCEL (hook `lateral-accel-15a`, VERIFIED on device 2026-07-02):** the 14b px-write
tilt FAILED (char-update owns PlayerXOffset outside GameController.Update — write war, px pinned).
15a replaces `TouchInput.GetAccelerometerForceX` (RVA 0x12AD500, Interceptor.replace + NativeCallback
— float return rides s0, attach/retval can't) with commanded `tiltCmd`: byte0=hold LEFT (−0.6·sign),
byte1=hold RIGHT, byte6=LEVEL→game recenters natively. MEASURED (captures/strafe_probe_tilt2.jsonl,
`probe_strafe.py --script tilt --start-dist 50`): lane = first-order lag toward ±7/0, **alpha 0.446
per 0.1 s frame both directions (lat_tau 0.169 s)**, tilt persists mid-jump, **early-run input lock**
(lateral ignored below ~dist 25–45), holding INTO the wall can scrape-bounce (snap −6.6→+1..3, then
re-glide). Sim `sim_env.py` remodeled to exactly this (`_lat_target` + `lat_tau`; nudge-timer/near-hold
model DEAD; NoOp recenters). Oracle: center = NoOp; narrow lane ±3.7 = bang-bang dither (only stable
commanded points are −7/0/+7). Old policies' strafe outputs now mean HOLDS — expect degraded old-model
captures.

## 17. VALIDATED-SIM policies + the TURN-DIRECTION perception fix (2026-07-02)

**The validation loop that finally worked - `scripts/replay_diff.py` (policy-FREE).** Every prior sim
change was checked by train->device-eval (the expensive loop the user flagged). replay_diff does a
**one-step transition diff**: for each real `(state, action)` in a device capture, `env.set_state(cur)`,
override `_dist_into_cell = cell_progress*cell_len`, `env.step(action)`, and compare the sim's next
frame to the device's next frame - per mechanic, in seconds, no policy overhead. This localizes fidelity
gaps directly. RULE (still): validate/read the real mechanic THEN change the sim; scorecard/replay_diff
green before any retrain (section 15).

**Mechanics device-anchored + fixed this way (all in `sim_env.py` `_execute_turn`/`_attempt_turn` +
`reward.py`):**
- **Turn LANDING (device-measured, `probe_turn_exec`):** a swipe executes instantly; the runner lands at
  lateral offset ~ `(swipe_cprog - 0.44)*45` (aim CELL-CENTER 0.44). `turn_offset_scale 45`,
  `turn_center_cprog 0.44`, `turn_lane_carry 3` (pre-turn tilt carries in), kill if `|offset| >
  turn_land_kill 9`. **SIGN: TurnLeft -> +offset, TurnRight -> -offset** (was backwards, cost a session).
  `_turn_landed` flag skips the recenter on the landing frame (turn_land_err 4.6 -> 1.4).
- **Turn-on-a-STRAIGHT is FATAL** (device 8/8), not a survivable stumble - `_attempt_turn` `_kill(1)`
  when not near a junction; mistimed AT a junction = `_stumble()`.
- **Narrows = SIDE strips only** (never center): generate `narrow_lane in {-1,+1}`, ptype {-1:19,+1:22};
  Region2D bands center +-4.5 half-width 2.5. Dither fix (user: "should strafe and HOLD"): `reward.py`
  EXEMPTS narrow strafes from `action_cost` + `--narrow-hold-bonus 0.05` rewards being on the strip.

**TURN-DIRECTION perception fix - hook `lateral-accel-16f` (the real win this session).** `can_turn_left/
right` were unreliable: the old `neighborExists` check degraded across the cell -> the policy had to GUESS
turn direction. Root: the game's native `PathElement.CanTurnLeft/Right(PathDirection origin)` takes the
**INCOMING direction = OPPOSITE of facing**, not facing. Fix: `OPP_DIR = {1:3,2:4,3:1,4:2}`, call
`fn.CanTurnLeft(pe, OPP_DIR[facing], NULL)` and CACHE it at the is_turn FLASH (latchCanL/R), hold through
the latch. **Device-verified: can_turn now 100% at junctions (6/6) - was ~0% mid-junction.** Pure
perception (can_turn already in the obs); helps every policy with no retrain.

**DEVICE EVAL under 16f (8 eps each, `--turn-cooldown 0`):**
- `ppo_valid_v1_best` (900k, recipe section 12 linger 3 + `--narrow-hold-bonus 0.05`): **median 1173,
  best 1956** (16e was 1144/1752). Deaths: **5/8 LATERAL-OFF-EDGE** (lane -9..-23, past +-7 edge),
  1 crevasse, 2 narrow.
- `ppo_valid_v2_best` (3M, + `--aim-center-weight` + `--velocity-drop-penalty` + `ent 0.035`):
  **REGRESSED - median 469, best 1365**, 6/8 lateral-off-edge (lane +-11..+-16). The aim-center/velocity
  reward tuning + lower ent made over-strafe WORSE. **KEEP v1; do NOT deploy v2.**

**NEW FRONTIER (turns are no longer the top killer - 16f fixed the direction guess): LATERAL
OVER-STRAFE.** Both v1/v2 now die mostly by driving `lane` to +-9..+-23 - far past the device +-7 edge
and the obs clip +-8 (so the policy is OOD out there) - the policy holds a strafe and never recenters.
This is the recurring "strafe-off-edge" device failure (MODELS.md faithful* rows) resurfacing as the
dominant death once turns are handled. NEXT: make the sim lateral over-strafe COSTLY/fatal at the edge
(a `|lane|>edge` Fall + a real recenter incentive), re-validate via replay_diff on the v1_16f
lateral-death frames, then ONE retrain from the v1 recipe (NOT v2's). Deploy model:
**`runs/ppo_valid_v1_best.zip`** (median 1173, best 1956). Hook: `lateral-accel-16f`. Device serial
**3f80a3a**; injector must run as a PERSISTENT background task (it exits -> in-phone agent unloads ->
stream dies; re-launches race each other, so run exactly ONE).

## 18. THE LATERAL "OVER-STRAFE" WAS A MISDIAGNOSIS - it's off-center TURN LANDING (2026-07-02)

**Context:** under hook 16f, v1's dominant death looked like LATERAL over-strafe (5/8 deaths, lane driven
to -9..-23 past the +-7 edge). Hypothesis: the policy strafes randomly on straights, drifts off-center,
and turn_lane_carry (~3x) amplifies it into a >kill turn landing. Fix tried: a RECENTER incentive.

**FIX BUILT (`off_center_penalty`, `reward.py` + sim_env + train_sim `--off-center-penalty`):** dock
`off_center_penalty*(|lane|-1.5)` on non-narrow cells (narrows + narrow_target_lane approach EXEMPT, so
strip-holding stays free). Reward-only -> sim dynamics untouched -> `fidelity_scorecard.py --strict`
stayed GREEN 10/10. Retrained `ppo_valid_v3` = v1 recipe + `--off-center-penalty 0.03` (linger 3, ent
0.04, narrow-hold 0.05). NO `_best` auto-saved (train_sim only checkpoints + `_final`; `_final` is
late-COLLAPSED) -> picked the best periodic checkpoint by an OFFLINE lateral rollout (scratchpad
`eval_lateral.py`: median + mean|lane|-on-straight + off-center-turn%).

**IN-SIM: the fix worked PERFECTLY** - |lane|-on-straight 0.41->**0.07**, off-center-turn 31%->**0%**,
median 1700. **ON DEVICE (v3 @1100k, 8 eps): HARD REGRESSION - median ~382 (v1 was 1173), best 968.**
The penalty made the policy PASSIVE/turn-shy (86% NoOp, StrafeRight 0) -> dies EARLY at the first turns.

**THE REAL DIAGNOSIS (v3 exposed it - trace `captures/valid_v3_16f.jsonl`):** the killer is the turn
LANDING itself, and it happens **EVEN FROM A CENTERED APPROACH.** ep5: policy arrives **lane 0.0**, fires
TurnRight (correct direction), and **lands at -14.6 -> off path.** So off-center-at-the-turn is NOT caused
by strafing-on-straights (that's a real but SECONDARY contributor) - the device **executes turns LATE**
(landing -14.6 from center implies exec ~cprog 0.76 via `offset=-(cp-0.44)*45`, vs the safe aim 0.44).
This is a TURN-TIMING problem. `off_center_penalty` addressed the wrong cause and suppressed the lateral
motion a turn NEEDS -> net worse. **KEEP v1 (median 1173). DO NOT deploy v3.** The knob is kept opt-in
(default 0, harmless).

**BUG FOUND + FIXED en route: the "0% can_turn" scare was a LOGGING omission.** Analyzing the captures,
`can_turn_left/right` read as 0 across ALL is_turn frames (0/124 v1, 0/90 v3) - looked like the 16f fix
was dead. It was NOT: the **raw :9999 stream carries can_turn at 100%** (re-verified live, correct per
facing) and the policy's obs is built from that full state. The captures just never LOGGED the two fields
- `capture_calib.py LOG_FIELDS` omitted `can_turn_left/right` (now added). 16f works; the policy was
never blind. Death classification (lane/path_type based) is unaffected.

**REGRESSION LEDGER - all three retrains since v1 trace to TURN LANDING/TIMING, not the thing tuned:**
v2 (aim-center + velocity-drop + ent 0.035) -> 469; the turnfix BUFFERING variant -> 226; v3
(off_center_penalty) -> 382. Lesson: stop tuning lateral/reward around the turn; the turn EXECUTION
timing is the structural gap. **RULE reaffirmed: measure the real mechanic + validate offline BEFORE
retraining.**

**NEXT (task #21, NOT started):** measure the turn command->execution latency on device with the
now-logged `can_turn` + facing-flip cprog (does the policy command turns late, or does cmd->native-exec
latency push 0.54->0.76? does the 16f is_turn LATCH to cprog 0.60 pull the aim to the LATE edge?). Then
model that latency in the sim so the policy learns to PRE-AIM toward exec-at-0.44, validate via
replay_diff on the turn frames, retrain ONCE from the v1 recipe (NOT off_center, NOT v2's knobs). Keep v1
deployed until beaten.

## 19. NARROWS CRACKED via a NARROW-ONLY CURRICULUM + the two-phase warm-start (2026-07-02)

**The narrow death was NOT reward/network/warm-start — it was a HARD-EXPLORATION wall.** v1 (and every
retrain) NEVER strafed onto narrow strips (device+sim |lane|-on-narrow ~0, drifts to center, Fall). A
narrow kills INSTANTLY on entry if not already on the +-4.5 strip (`|lane-strip|>tol 2.5 -> Fall`), so
the policy must strafe onto the strip DURING the 1-cell approach (a precise ~2-frame action, correct
direction, before any survival feedback). PPO's random exploration almost never produces that sequence,
and there's no room for trial-and-error ON the narrow (first wrong frame = death). Eliminated 3
hypotheses with experiments: NOT warm-start prior (from-scratch v6 also |lane| ~0), NOT reward magnitude
(0.05->0.18 barely moved it), NOT reward shape alone. The `_oracle` (hardcoded reference policy, never
runs on device) strafes narrows fine (0/40, |lane| 2.84) -> the sim IS survivable; RL just can't DISCOVER
the skill on a mixed track (it dies on TURNS at ~1500m before ever practicing narrows).

**THE FIX (user's idea): train on a NARROW-ONLY track until the skill is learned, then warm-start.**
- **Narrow-only sim config** (`--sim-config`, `--no-domain-rand` so DR doesn't re-add turns): `turn_prob
  0, turn_prob_late 0, obstacle_prob 0.9, narrow_prob 1.0, narrows_after_distance 0,
  zone_change_every_cells 1e9`. **GOTCHA: narrows are a SUBTYPE of obstacle placement** (the narrow
  branch is nested inside `if place:` gated by `obstacle_prob`) -> need `obstacle_prob>0` AND
  `narrow_prob=1.0` (every placed obstacle becomes a narrow, none become jumps/holes), AND
  `narrows_after_distance=0` (an early-dying policy never reaches a late onset). Verified: track is
  narrows-only (0 turns, 0 jumps, 483 approach-warning frames / 3000).
- **`runs/ppo_narrowonly` (from scratch, narrow-only): SOLVED narrows in 700k steps** — |lane|-on-narrow
  **3.81** (beats the oracle 2.84), **0/40 narrow deaths**, median 1823 (sim cap). Proof the NN CAN do
  narrows; the mixed track was the only blocker.
- **Phase 2 `runs/ppo_valid_v7` (warm-start full task from ppo_narrowonly_700000):** `--init-from`
  (new flag: `model.set_parameters(zip)` before training) + narrows kept DENSE-early
  (`narrows_after_distance 200, narrow_prob 0.70`) so the narrow skill is RETAINED (not catastrophically
  forgotten) while turns/obstacles/crevasse are relearned. DR on. Full recipe + `--aim-center-weight 1.0`.

**RESULT `ppo_valid_v7_best` (=3M): the FIRST policy with ALL THREE mechanics at once (in-sim, 40 eps):**
narrow deaths **0** (|lane|-on-narrow **3.86**, was 0.42 on v1), turn deaths **5** (off-center-turn 20%,
was 31% on v1), crevasse **0**, **31/40 reached the ~1823 sim cap**, median **1777**. NO forgetting.

**REWARD CHANGES that support it (`reward.py`, all reward-only -> scorecard --strict stayed GREEN):**
- **Graded+reshaped narrow strip reward** (`narrow_hold_bonus`): reward `lane*sign(ntl)` (progress onto
  the correct side), FULL once |lane|>=3 -> covers the oracle's ~2.84 dither AND the EASY +-7 full-tilt
  hold (a stable commanded point). The earlier BINARY reward (on a narrow cell, any lane) gave NO
  gradient; a tent PEAKED at 4.5 under-rewarded the +-7 hold. This graded version is the pull off center.
- **Turn AIM-CENTER** (`--aim-center-weight 1.0`): scale the turn reaction_bonus by closeness of the
  swipe cprog to 0.44 (0 by 0.60) -> pulls the swipe off the late edge. Works (offctr 31%->20%, warm-
  started even). NOTE: on device a swipe still executes ~1 frame late (the §18 lag), so this REDUCES but
  may not zero turn-landing deaths; the full fix is modeling that lag (§18 / task #21).

**NEXT SESSION (device eval, first thing):** re-inject hook 16f, `capture_calib.py runs/ppo_valid_v7_best
--episodes 8 --turn-cooldown 0`, classify deaths (turn/narrow/crevasse). Compare vs v1 median 1173.
Expect narrow deaths gone; watch turn-landing (the §18 lag) + whether all-3-mechanics holds on device.
If narrows regress on device (sim-only skill), the lever is more dense-narrow phase-2 steps.

**DEPLOYED (user decision 2026-07-02): `runs/ppo_best.zip` = `ppo_valid_v7_best` (v7 @3M).** The canonical
deploy slot now holds v7 (all-3-mechanics GREEN in-sim: 0 narrow / 5 turn / 0 crevasse, |lane| 3.86,
median 1777). **NOT yet device-verified** — deployed on the sim result; device eval is the first task next
session (`capture_calib.py runs/ppo_best.zip --episodes 8 --turn-cooldown 0` on hook 16f, classify vs the
v1 baseline device median 1173). Prior deploy `ppo_valid_v1_best` (device 1173) kept as the rollback.
LESSON: rare critical skills need ISOLATION training (narrow-only), not just reward tuning on the full
task.

## 20. SPEED CURRICULUM (v8) — turns UP, crevasse/slide REGRESSED; two fidelity gaps fixed (2026-07-03)

**User idea (fixes training-data diversity): randomize sim START SPEED.** The policy always resampled the
slow early track → overfit it, rarely reaching new fast situations before dying. Fix = `curriculum_max_
start_dist` (was BUILT, never enabled): a fraction of episodes jump-start at a random distance in [0,cmax],
velocity set from the curve there, usual accel continues. Turns get harder at speed FAITHFULLY (cprog
advances ~0.35/tick @vel210 vs 0.17 @vel100 — the `velocity*dt/cell_len` coupling) → trains the fast
sections. Added `--curriculum-max-start-dist` flag (train_sim); `--sim-config` still the escape hatch.

**v8 = warm-start v7 + `--curriculum-max-start-dist 8000` (recipe §12 + narrow-hold 0.18 + aim-center 1.0).**
Best checkpoint **1.1M** (later COLLAPSED — 1.4M+ regressed; KEEP BEST not _final). Offline (curriculum
config, bucketed by START velocity, `scratchpad/eval_v8.py`): high-speed turn survival **33%→58%** (v7→v8),
narrows retained (|lane| 4.2), off-center-turn 27→24%. **The curriculum WORKED for turns.**

**DEVICE (hook 16f, 8 eps): REGRESSED — median 830 (< v1 1173).** Clean 4/4 death split (capture
`captures/speed_v8_16f.jsonl`, `scratchpad/eval_v8.py` classify):
1. **CREVASSE falls (4/8, death_type 4, @v165-180 mid-speed):** the policy UNDER-JUMPS on approach (NoOp'd
   71% of next_must_jump frames — the user's "hardly jumping") then jumps too late on the gap cell
   (is_jumping never engages), or too EARLY (ep0: airtime ends mid-gap). The curriculum's high-speed weight
   PULLED the single jump-timing policy toward the fast regime → degraded the MID-speed crevasses the
   device reaches FIRST. NEGATIVE TRANSFER — the sim high-speed crevasse is self-consistent (new
   `crevasse.speed` check: win_lo≡win_hi≡[0.1..0.8]), so NOT a broken sim window, pure interference.
2. **SLIDE-then-JUMP death (4/8, death_type 3) — NEW bug (user-caught):** slides correctly UNDER the
   obstacle, then fires a spurious JUMP mid-slide → pops UP into the low bar → death (3 explicit
   JUMP-while-sliding events confirmed). New because (a) the high-speed policy is more erratic (stray
   jumps), (b) **the SIM never modeled jump-into-slide = death** → the policy was never punished. A sim
   FIDELITY GAP.

**COMPUTE IS NOT THE FIX (user asked "train longer?"): the BEST checkpoint was 1.1M; 1.4M+ got WORSE.**
More steps on this curriculum = more collapse. Failures are DETERMINISTIC mis-timings + a spurious action
the sim can't teach against — not undertraining. Fix = DESIGN + fidelity, validated offline.

**TWO FIXES BUILT (scorecard GREEN 12/12, +2 new permanent guard checks):**
- **Jump-into-slide = death** (`sim_env` `_resolve_obstacle_on_enter` slide branch + per-tick `_advance`:
  airborne `_jump_t>0` on a `_MUST_SLIDE` cell → `_kill`). Reward `slide_jump_penalty` docks a JUMP when
  must_slide/next_must_slide/is_sliding (`--slide-jump-penalty`, try 0.15). New scorecard `slide.jump`
  (slider survives 0.92 / jumper dies 0.00; both clear jumps identically → isolates the slide death).
- **Curriculum TAMING** (`curriculum_deep_start_frac` 0.35 — was a hardcoded 0.5; `curriculum_dist_power`
  1.5 → sample d=cmax·u^power, biases start-distance toward the mid-speed ramp so high-speed doesn't drown
  the mid-speed crevasses). New scorecard `crevasse.speed` (high-speed crevasse window must be non-empty
  AND overlap low-speed — the across-speed check the old slow-only `crevasse.curve` lacked; would have
  caught this regression before device).

**"the sim is closed" was an OVERCLAIM (user rightly pushed):** scorecard-GREEN = the ~10 mechanics WITH a
device reference agree; it is structurally BLIND to any mechanic/interaction never checked (jump-into-slide
was one — an unknown-unknown). Open gaps: turn cmd→exec LATENCY (§18, unmodeled, the #1 turn killer);
SUSPECTED-unknown (asked user, pending) = mid-run speed SPIKES/boosts (would break velocity=f(distance)),
smooth path CURVES (non-junction bends the policy can't swipe), hazards right AFTER a turn, 2-action
obstacle combos. Deploy floor stays v1 (device 1173); `ppo_best` still v7. v8 shelved (`ppo_speed_v8_best`
= 1.1M).

**NEXT: v9 = warm-start v7 + `--slide-jump-penalty 0.15` + TAMED curriculum (`--curriculum-max-start-dist
5000`, frac 0.35, power 1.5 defaults).** Device-eval vs v1 (1173) + classify crevasse/slide deaths.

**GAP-QUESTION ANSWERS (user, 2026-07-03) — 2 suspected gaps CLOSED, 3 real gaps CONFIRMED:**
- **Speed spikes: NONE on this build.** Original TR has short invincibility-boost powerups; the user's
  modified phone build has NO boost. Speed rises ONLY gradually → **velocity=f(distance) model is VALID**
  (no boost state needed). NOT a gap.
- **Path curves: NONE.** The path is STRAIGHT + narrows (lean) + swipe-junctions; no smooth non-junction
  bends. The §17 "curve piece pt3/4" worry was about the TRACK-FORCING experiment, not normal play. NOT a
  gap; the sim's straight+junction geometry is right.
- **TURN mechanic (THE fix, task #21):** a swipe JUST BEFORE the junction is HELD and fires AT the
  junction → CENTERED. TOO EARLY → a hit + slowdown (stumble). AT the junction → immediate (lands per
  cprog). So the SAFE play is PRE-AIM on the late-approach cell; the policy currently waits for is_turn
  (junction cell) + swipes late (16f latch holds is_turn to 0.60) → immediate exec at high cprog →
  off-center landing = the §18 death. FIX: model a TIGHT approach-buffer (late-approach swipe → held →
  fires at junction center; far-early → stumble). CAUTION (§14): the OLD buffer fired on next_is_turn ANY
  time = SIM EXPLOIT (off-path on device) → keep it TIGHT + replay_diff-validate on real turn frames +
  turn_gate PRE-FIRE(far) must still stumble/die. This is **v10** (after v9).
- **After-turn hazards + two-action combos: BOTH confirmed.** (1) a hazard often appears right AFTER a
  junction (the policy is busy turning → reacts late; §11 residual death) — sim should place obstacles
  right after junctions. (2) some obstacles need TWO quick actions (jump-then-slide / lean+jump) — the sim
  models ONE obstacle per cell; a real gap. Model both (secondary to the turn buffer). UPDATE: user says
  after-turn holes are handled "quite well" already → DEPRIORITIZE (1); combos stay low priority.

## 21. v9 DEVICE eval + the TURN fix REFRAMED by replay_diff (2026-07-03) — RESUME HERE

**v9 device (hook 16f, 8 eps, `captures/speed_v9_16f.jsonl`): median 830→ REGRESSED (v1=1173).** BUT the
v9 FIXES WORKED: slide-jump deaths 0 (was v7=22), crevasse mostly gone. The deaths MOVED: **7/8 are
now the off-center TURN LANDING / lateral** (death-frame |lane| 10–17 on straights, death_type 1 Fall).
So the crevasse/slide fixes exposed the turn+lateral death underneath. Offline v9 crushed v7 (slide-jump
22→0, high-speed survival 33→68%, narrows held) — but the sim couldn't SEE the turn/lateral death (below).

**MEASURED turn latency (device, no guess):** a swipe ON the junction cell (is_turn True) flips facing
**1 frame (100ms) later** (21/21, constant in time). A swipe EARLY on the APPROACH (probe_turn_window,
can_turn set) is BUFFERED → fires ~4 frames later AT the junction → CENTERED (= user's Q2). Late
junction-cell swipe → off-center: swipe cprog>0.60 → |lane| 8.9med/17max (dies), r=0.68. Per-frame cprog
advance ≈ 0.0016·vel.

**`replay_diff.py captures/speed_v9_16f.jsonl` REFRAMED the fix — the sim turn model is wrong 3 ways
(not just latency):**
1. **Over-kills turns from off-center lane** — `turn_lane_carry=3` kills turns the device SURVIVES
   (junction is_dead agreement only **37–39%**; turn_land_err 3.7–4.3).
2. **Under-kills off-edge straights** — `sim_env.py:1002` HARD-CLIPS `_lane` to ±7, so the real death
   (lane→11 off the path edge on a straight) is UNMODELED → the policy strafes/lands high for FREE in sim.
3. **No approach-BUFFER** — the safe real play (early swipe → held → centered) isn't modeled; nor is the
   1-frame late-swipe penalty.
Death chain: late junction-cell swipe → land off-center ~lane 10 → can't recover → fall off the edge on
the next straight. **Off-edge death is CONDITIONAL, NOT a lane threshold** — device SURVIVING frames
reach |lane| 17 transiently (post-turn recenter); deaths are high-lane AT a cell crossing/no-recover. So
DON'T hard-threshold |lane| — must measure the trigger.

**DONE this session (safe):** `predicted_landing` obs (obs.py, OBS_LEN 30→**31**, index 30 =
`cell_progress + DCPROG_PER_VEL·velocity`, DCPROG_PER_VEL=0.1/60; host-side, no hook change; VERIFIED).
Plan file `read-the-claude-md-before-abundant-gray.md` (v10). Memory [[turn-latency-sampling-ideas]].
**GOTCHA: OBS_LEN=31 breaks the K-adaptive loaders for OLD obs-30 models** ((dim−31)//7≠4) — revert
OBS_LEN or add an adapter to eval v1/v9 again.

**RESUME — the turn+lateral fidelity package (replay_diff-gated, each threshold MEASURED, user to steer
which first):** (a) off-edge death (CONDITIONAL — measure the trigger: cell-crossing at high lane? N
sustained frames? un-clip `_lane`), (b) `turn_lane_carry` recalibration (sim over-kills; maybe junction
lane reads are unreliable → clamp), (c) approach-buffer + 1-frame latency (model the user's Q2 so the
policy pre-swipes on approach → centered). Then retrain v10 (warm-start-surgery from v9_best for obs 31).
Deploy floor stays v1 (1173); ppo_best still v7. Injector was alive on 3f80a3a this session (may drop).

## 22. v10 = predicted_landing obs + approach-buffer — TURNS FIXED ON DEVICE (2026-07-03)

**Built:** `turn_approach_buffer` flag (HandleDelayedTurning — approach-swipe latched → fires centered;
validated 20→1823, spam/wrong die, scorecard GREEN flag-off) + `predicted_landing` obs (OBS_LEN 30→**31**,
`cell_progress + DCPROG_PER_VEL·velocity`, DCPROG_PER_VEL=0.1/60, host-side no hook change) + aim-center
reward re-keyed to the LANDING + `_peek_next()` bugfix (was `_next_cell()` = a GENERATOR advancing RNG).
Commit cbfdb09. v10 = warm-start-surgery from v9 (zero-col insert @ obs idx 30,
`scratchpad/warmstart_surgery.py`) + v9 recipe + `--turn-approach-buffer`. Best turn ckpt = **2.8M**.

**OFFLINE (buffer ON): ≈ v9, approach-swipes = 0** — the warm-started junction-swiper NEVER adopted the
buffer (no reward incentive; buffer AVAILABLE, unused). The sim has NO device latency → it CANNOT show the
predicted_landing benefit; only the device can.

**DEVICE (hook 16f, 8 eps, `captures/v10_16f.jsonl`): median 615, best 2608. THE PREDICTED_LANDING INPUT
WORKS — TURNS FIXED.** Turn landings |lane| **median 4.0 / max 8.1** (v9 was 10–17 = fatal); swipe cprog
0.42 (centered); turns landing off (|lane|>8) **1/6**; **turn deaths ~0**. Dominant death SHIFTED
turns→**CREVASSE (5/8 at pt 33/35/36 ~600m)**. The ~1000m turn wall that blocked 40+ policies is DOWN on
device via a pure PERCEPTION input (predicted_landing) — NO hook action rule. Median flat only because a
CREVASSE regression (v8-style §20 negative-transfer under-jump — NOT shown offline @2.8M → a sim↔device
crevasse fidelity gap) is newly EXPOSED now that runs survive turns and reach the crevasses.

**DEPLOY:** v1 still the median floor (1173); v10 best-run 2608 but median 615 (crevasse-limited). Keep v1
until the crevasse fix lands. **NEXT: crevasse takeoff fidelity** (measure device takeoff cprog vs sim at
the mid/high transitions, §11/§20; the TURN frontier is CLOSED). Unused buffer + off-edge/carry replay_diff
fixes remain optional. Registry: v10 = `runs/ppo_v10_best.zip` (=2.8M). See [[turn-latency-sampling-ideas]].
