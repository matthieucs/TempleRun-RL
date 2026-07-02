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
