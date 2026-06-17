# Temple Run RL — project memory

Goal: turn **Temple Run** (`com.imangi.templerun`, v1.39.0) into a reinforcement-learning
environment by instrumenting the running game with **Frida** — read game state and inject
actions through TCP sockets, no screen-scraping, no synthetic input.

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
