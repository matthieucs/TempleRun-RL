![Banner](assets/banner.png)

# TempleRun-RL 🏃

Train reinforcement-learning agents to play **Temple Run** on a **physical Android phone** — **no root**,
**no screen-scraping**. A [Frida](https://frida.re) gadget injected into the game reads the live game
state straight from memory and executes actions, both over TCP. A
[Gymnasium](https://gymnasium.farama.org) environment wraps those two sockets, and a
[Stable-Baselines3](https://stable-baselines3.readthedocs.io) PPO agent trains against it.

```
  Android phone (arm64, no root)                Host (this repo)
 ┌───────────────────────────────┐            ┌──────────────────────────────┐
 │ Temple Run + Frida gadget     │  tcp:9999  │ StateClient ─► TempleRunEnv  │
 │  • reads CharacterPlayer/...  │ ─────────► │  (gymnasium.Env)  ─► SB3 PPO │
 │  • calls Jump/Slide/Turn/...  │ ◄───────── │ ActionClient ◄─ env.step()   │
 │  (state @9999, actions @9998) │  tcp:9998  │                              │
 └───────────────────────────────┘            └──────────────────────────────┘
        (adb forward maps the phone's 9999/9998/27042 to 127.0.0.1)
```

> ⚠️ **The game is not included.** This repo ships only our own code — the env, scripts, the Frida
> agent, and the APK-build tooling. Supply your own legally-obtained copy of Temple Run and build the
> instrumented APK yourself ([docs/SETUP.md](docs/SETUP.md)). See [LICENSE](LICENSE).

---

## Why a physical phone (not an emulator)

The game ships **arm64-only** (Unity IL2CPP). That rules out emulators:

- On an **x86_64 emulator**, the game runs through ARM→x86 translation, which **hides `libil2cpp.so`
  from Frida** — hooks can't resolve.
- A modern Android emulator **won't boot an arm64 image on an x86_64 host** at all.

So instrumentation needs native arm64: a real phone (used here) or an arm64 cloud host. Evidence in
[CLAUDE.md](CLAUDE.md) / [docs/GADGET.md](docs/GADGET.md).

## Prerequisites

- A **non-rooted arm64 Android phone** with USB debugging, plus `adb` (Android platform-tools).
- **Python 3.12+**.
- Your own **Temple Run** APK, and a one-time build of the gadget-injected APK ([docs/SETUP.md](docs/SETUP.md)).
- *For the APK build only:* a JDK (`zipalign`/`apksigner` from Android build-tools), `frida`, `lief`.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quickstart

1. **Install the gadget APK** on the phone (one-time build — see [docs/SETUP.md](docs/SETUP.md)).
   On MIUI/HyperOS, `adb install` is often blocked; push the APK and tap it in the Files app.
2. **Bring the device up** — forwards the ports and injects the agent. Leave it running:
   ```bash
   python scripts/setup_device.py            # add --serial <id> if multiple devices
   ```
   Wait for `ready: state->9999 action<-9998`.
3. **Open the game** and start a run (or sit on the game-over screen). The env's `reset()` restarts from there.
4. **Validate the env** (no training):
   ```bash
   python scripts/smoke_test.py
   ```
5. **Train / evaluate:**
   ```bash
   python scripts/train.py --timesteps 100000
   python scripts/evaluate.py --model runs/ppo_templerun_final.zip --episodes 10
   python scripts/evaluate.py --random --episodes 5     # baseline
   ```

Or use the env directly:

```python
import gymnasium as gym
import templerun_rl                       # registers "TempleRun-v0"

env = gym.make("TempleRun-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

---

## Environment spec

### Observation — `Box(0, 1, (32,))`

A normalized vector of **32 features** (all scaled to `[0, 1]`), grouped by what they describe:

| Group | Fields |
|---|---|
| **Player state** (bool) | `is_dead`, `is_jumping`, `is_sliding`, `is_on_ground`, `is_stumbling`, `is_strafing`, `is_game_over` |
| **Power-ups** (bool) | `has_invincibility`, `has_angel_wings`, `has_vacuum`, `has_boost`, `is_mega_boost` |
| **Obstacle cues** (bool) | `must_jump`, `must_slide` + 1-cell lookahead `next_must_jump`, `next_must_slide`, `next_wide_gap`, `next_ground_change_higher` |
| **Turn cues** (bool) | `can_turn_left`, `can_turn_right`, `is_turn` + lookahead `next_is_turn`, `next_can_turn_left`, `next_can_turn_right` |
| **Scalars** | `velocity`, `path_type`, `path_level`, `facing` |
| **Lane & timing** | `cell_progress` (position within the current cell), `lane` (L/C/R), `narrow_target_lane`, `predicted_landing` |

- The raw, unbounded **`score` is deliberately excluded** — it drives the reward, not the policy. The
  full raw state dict is always available in `info["raw_state"]`.
- **Width scales with the wrappers:** `+7` per action-history frame (`action_history_len`) and `×frame_stack`.
  The authoritative field list lives in [`src/templerun_rl/obs.py`](src/templerun_rl/obs.py).

### Action — `Discrete(7)`

| Byte | Action | | Byte | Action |
|:---:|---|---|:---:|---|
| `0` | Strafe left  | | `4` | Turn left  |
| `1` | Strafe right | | `5` | Turn right |
| `2` | Jump         | | `6` | No-op      |
| `3` | Slide        | | | |

**Reset-only bytes** (never emitted by the policy):

- `8` **RunAgainClicked** — restart from the *game-over* screen.
- `9` **OnPlayClicked** — start from the *main menu* ("JOUER").
- `7` **RestartGame** — resets game logic but leaves the run in limbo; unused.

`reset()` sends `8` **then** `9`. Each is a gadget-side no-op on the wrong screen, so a run starts
whether the phone is on game-over **or** the menu — which keeps `reset()` robust even when the game
drops to the menu after an interstitial ad.

### Reward

- **Per step:** `(score − prev_score) × reward_scale` (falls back to distance-delta if score is unavailable).
- **On death:** minus `death_penalty`.
- **Defaults:** `reward_scale = 0.01`, `death_penalty = 1.0` (both set in the constructor).

### Episode

- **`terminated`** when `is_dead` / `is_game_over`.
- **`truncated`** at `max_episode_steps` (default **1000**) — a *ceiling*, since runs almost always end
  by death first.
- **Adaptive cap** *(optional, off by default)* — with `grow_episode_steps=True` (or `--grow-episode-steps`),
  the cap ratchets up each time the agent survives a full episode (`×max_episode_steps_growth`, bounded by
  `max_episode_steps_ceiling`), so it keeps unlocking the faster late game instead of being clipped.

---

## Layout

```
src/templerun_rl/   the package: env.py, client.py, obs.py, constants.py
scripts/            setup_device.py, train.py, evaluate.py, smoke_test.py
frida/              temple_run_hook.js (the agent), gadget_inject.py, inject.py, diag.js
tools/              build_gadget_apk.py (APK patcher)
docs/               SETUP.md, GADGET.md, OFFSETS.md
CLAUDE.md           full project notes (offsets, gotchas, history)
```

## Scaling to multiple phones

**One phone = one env**, real-time at ~10 Hz, with short episodes. There's no simulator here, so the only
way to collect faster is to run **more phones in parallel** — and the plumbing supports it:

- **Ports.** In-phone ports are always `27042/9999/9998`; on the host each device gets its own triple at a
  stride of 100 (`device_ports(i)` → `9999+i*100` / `9998+i*100`).
- **Bring each device up** with a distinct slot:
  ```bash
  python scripts/setup_device.py --serial PHONE_A --device-index 0
  python scripts/setup_device.py --serial PHONE_B --device-index 1
  ```
- **Train across all of them** — PPO collects `n_steps × num_envs` transitions per rollout:
  ```bash
  python scripts/train.py --timesteps 100000 --num-envs 2
  ```
  `--num-envs > 1` switches from `DummyVecEnv` to `SubprocVecEnv`, so each phone's real-time wait
  **overlaps** (≈100 ms/vec-step regardless of N, up to USB/host limits) instead of serializing.

Each phone needs the gadget APK installed, the game open, and `svc power stayon true`. A powered USB
hub helps once you have several devices.

## Staying up over long runs (resilience)

Real hardware drops out — screen sleep, app relaunch, the occasional interstitial ad — and one phone
freezing must not take the whole run down. Two layers guard against this:

- **Resilient clients + freeze tolerance.** `StateClient` auto-reconnects, `ActionClient` retries
  through a gap, and the env waits up to `freeze_timeout` (default 90 s) for the stream to resume — so a
  transient freeze *pauses* the worker instead of crashing the `SubprocVecEnv` run.
- **Watchdog.** Runs alongside training to self-heal silent devices (re-apply keep-awake, wake &
  foreground, re-inject the agent if the app restarted):
  ```bash
  python scripts/watchdog.py --serials SERIAL_A,SERIAL_B     # position = device index
  ```
  > A phone stuck on a *non-gameplay* screen (main menu) still needs a manual tap — `adb` input is
  > blocked on MIUI, and `reset()` can only restart from the game-over screen.

## Caveats

- The instrumentation offsets are specific to **Temple Run v1.39.0**; a different version needs a re-dump
  (see [docs/OFFSETS.md](docs/OFFSETS.md)). The Python layer keys only on JSON field names, so as long as
  the state contract holds, no Python changes are needed.

## License

[MIT](LICENSE) for the code in this repository. Not affiliated with Imangi Studios; the game and its
assets are not included or licensed here.
