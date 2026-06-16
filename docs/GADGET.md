# Gadget injection — how & why

How we get a Frida agent running inside Temple Run on a **non-rooted** phone, and the design choices
that cost real debugging. Source of truth for the instrumentation is `frida/temple_run_hook.js`.

## Why Frida Gadget (not frida-server)
`frida-server` needs **root** to `ptrace`-attach to another app. The phone here is a stock, non-rooted
device, so instead we embed the **Frida Gadget** library *inside the app*: it loads in-process when the
app starts (no root, no ptrace) and listens on a TCP control port. The trade-off is we must repackage and
re-sign the APK once.

## Getting the gadget to load — DT_NEEDED on `libmain.so`
The usual gadget approach edits the app's smali to add a `System.loadLibrary("gadget")`. We avoid smali
entirely with an **ELF trick** (`tools/build_gadget_apk.py`):

1. Merge the split APKs into one universal APK (APKEditor) so all native libs live in one package.
2. Use **LIEF** to add a `DT_NEEDED("libgadget.so")` entry to **`lib/arm64-v8a/libmain.so`**. `libmain.so`
   is the small Unity bootstrap the app loads first; the dynamic linker then auto-loads any `DT_NEEDED`
   dependency — so the gadget comes up before the game does, with zero code edits.
3. Add `libgadget.so` and a config file `libgadget.config.so` next to it:
   ```json
   {"interaction": {"type": "listen", "address": "127.0.0.1", "port": 27042, "on_load": "resume"}}
   ```
   `on_load: resume` means the app runs normally and the gadget just listens (vs. blocking at startup).
4. Drop the old signature, then **zipalign + apksigner** with any debug key.

At runtime, logcat shows `Frida: Listening on 127.0.0.1 ... 27042`. `frida/gadget_inject.py` connects to
that port (`adb forward tcp:27042`), attaches to the gadget's pseudo-process named **`Gadget`**, and loads
`temple_run_hook.js`. The agent then resolves `libil2cpp.so`, installs hooks, and opens the two TCP servers.

## Why a native arm64 target is mandatory
The game is **arm64-only IL2CPP**. On an x86_64 emulator it runs under ARM→x86 translation, and
`Process.findModuleByName("libil2cpp.so")` returns **null** (the lib is mapped by the translation guest
linker, invisible to Frida's host module API) — so no hooks resolve. A modern Android emulator also won't
boot an arm64 image on an x86_64 host. Net: use a real arm64 phone (or an arm64 cloud host).

## The agent (`temple_run_hook.js`) at a glance
- **Hooks** (Interceptor): `GameController.Start` (neutralize tutorial), `PlayerPrefs.GetInt` (force
  `"TR Tutorial"`→999), `GameController.Update` (capture the live instance, sample run velocity, and
  drain the action queue **on the main thread** — essential for thread-safe IL2CPP calls),
  `RecordManager.DeserializeRecordManager` (max upgrades).
- **State server (9999):** every 100 ms it reads ~19 fields directly from `CharacterPlayer` /
  `GameController` / `PathElement` (pure memory reads) and broadcasts newline-delimited JSON.
- **Action server (9998):** reads one action byte and queues it; the `Update` hook dispatches it by
  calling the real game function (`Jump`, `Slide`, `StrafeLeft/Right`, `TurnLeft/Right`, `RestartGame`).

## Gotchas worth remembering
- **Actions need ONE persistent socket.** Per-action connect/close races the server's reader on close, so
  only the first byte survives — the host `ActionClient` keeps a single socket open (`src/templerun_rl/client.py`).
- **Reset = `RunAgainClicked()` (byte 8), not `RestartGame()` (byte 7).** Verified empirically:
  `RestartGame()` resets game *logic* (score→0, alive) but leaves the run in **game-over limbo** — the
  player is never placed on the track, so it never scores or dies (no learning signal). The UI "Rejouer"
  handler **`EndGameUI.RunAgainClicked()`** (reached via `CharacterPlayer.GameOverInterface` @ `0x1A0`)
  actually starts a fresh, *progressing* run, so `env.reset()` uses byte 8. After it there's a ~7-frame
  transition (UI sliding out) before the run starts; `reset()` waits until the player is on a track tile.
  Caveat: `RunAgainClicked` occasionally routes through an **interstitial ad**, which would stall a reset
  until dismissed (can't be tapped via adb on MIUI) — a known limitation; future work is to hook and
  suppress the ad. Byte 7 (`RestartGame`) is kept exposed per the original action spec.
- **Actions are in-process calls — no `adb input`.** MIUI blocks synthetic input (`INJECT_EVENTS`); we never
  use it. This is the whole point of the approach.
- **Diagnostics use `send()`, not `console.log`** (the gadget's console is flaky over the remote channel);
  `gadget_inject.py` prints them as `DBG:`.
- **Offsets are version-specific** (v1.39.0) — see [OFFSETS.md](OFFSETS.md).
