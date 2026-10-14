# Setup — reproduce from scratch

This builds the gadget-injected APK from **your own** copy of Temple Run, installs it on a non-rooted
arm64 phone, and brings the device up for training. The instrumentation offsets here target
**Temple Run v1.39.0** (`com.imangi.templerun`); for another version, re-dump (step 2) and update
`frida/temple_run_hook.js` per [OFFSETS.md](OFFSETS.md).

> You must supply your own legally-obtained game files. Nothing copyrighted is included in this repo.

## 0. Prerequisites
- Non-rooted **arm64-v8a** Android phone, USB debugging on.
- **Android platform-tools** (`adb`) and **build-tools** (`zipalign`, `apksigner`).
- A **JDK** (for `apksigner` and APKEditor).
- **Python 3.12+**, then `pip install -r requirements.txt`.
- **Frida 17.14.0** matching binaries (host `frida-tools` and the arm64 gadget; see below).

## 1. Obtain the game APK(s)
Get the split APKs for the version you own — `com.imangi.templerun.apk` (base) and
`config.arm64_v8a.apk` (native libs). These commonly come as an APKPure/XAPK bundle or a backup of an
app you installed. **Keep them out of the repo** (they're git-ignored).

## 2. (Only for a new version) Dump the offsets
Run **Il2CppDumper** (v6.7.46) on the game's `libil2cpp.so` + `assets/bin/Data/Managed/Metadata/global-metadata.dat`:
```
Il2CppDumper.exe libil2cpp.so global-metadata.dat out/
```
The field offsets and method RVAs in `frida/temple_run_hook.js` come from `out/dump.cs`.
For v1.39.0 they are already filled in (see [OFFSETS.md](OFFSETS.md)) — skip this step.

## 3. Build the gadget-injected APK
1. **Merge the splits** into one universal APK with [APKEditor](https://github.com/REAndroid/APKEditor):
   ```
   java -jar APKEditor.jar m -i <dir-with-the-two-apks> -o universal.apk
   ```
   (This sets `extractNativeLibs=true`, which the next step relies on.)
2. **Get the Frida gadget** (must match host frida — 17.14.0): download
   `frida-gadget-17.14.0-android-arm64.so.xz` from `github.com/frida/frida/releases`, decompress to `libgadget.so`.
3. **Inject** the gadget (adds `DT_NEEDED libgadget.so` to `libmain.so`, drops the old signature):
   ```
   python tools/build_gadget_apk.py --universal universal.apk --gadget libgadget.so --out patched-unsigned.apk
   ```
4. **Align + sign** (any debug keystore works; create one once with `keytool`):
   ```
   zipalign -f 4 patched-unsigned.apk patched-aligned.apk
   apksigner sign --ks tr-debug.keystore --ks-pass pass:android --ks-key-alias trkey \
                  --out TempleRun-gadget.apk patched-aligned.apk
   ```
See [GADGET.md](GADGET.md) for why this works.

## 4. Install on the phone
`adb install TempleRun-gadget.apk` — but **MIUI/HyperOS commonly blocks adb install**
(`INSTALL_FAILED_USER_RESTRICTED`). Workaround that needs no Mi-account:
```
adb push TempleRun-gadget.apk /sdcard/Download/
```
then open **Files → Downloads** on the phone and tap the APK to install (allow the file manager to
install unknown apps; choose "install anyway" past any scan warning).

## 5. Bring the device up
```
python scripts/device/setup_device.py            # add --serial <id> if more than one device
```
This forwards `27042/9999/9998` and injects `frida/temple_run_hook.js`. Leave it running; wait for:
```
ready: state->9999 action<-9998
```
Open the game and **start a run** (the env's `reset()` restarts from a run or the game-over screen).

## 6. Validate (no training)
```
pip install -e .
python scripts/device/smoke_test.py
```
A pass means the env is usable to train. Then see the README for `train.py` / `evaluate.py`.

## Troubleshooting
- **`Cannot reach state/action ... 127.0.0.1:99xx`** — `setup_device.py` isn't running, or the game/gadget
  isn't loaded. Check the injector printed `ready: ...` and the game is in the foreground.
- **Reset fails / no fresh run** — make sure the game is actually in a run or on the game-over screen, not
  in a menu/cutscene, and that the **phone screen is on** (a locked screen pauses Unity so resets can't
  start). `reset()` sends `RunAgainClicked` (byte 8) and waits up to `reset_timeout`. If an interstitial
  ad pops up on restart it will stall (can't be dismissed via adb on MIUI) — just tap it on the phone.
- **adb input/tap "needs INJECT_EVENTS"** — expected on MIUI and irrelevant: actions are in-process Frida
  calls, never synthetic input.
