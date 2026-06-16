# templerun-rl

Train reinforcement-learning agents to play **Temple Run** on a **physical Android phone**, with
**no root** and **no screen-scraping**. A [Frida](https://frida.re) gadget injected into the game
reads 19 game-state fields straight from memory and accepts action commands, both over TCP. A
[Gymnasium](https://gymnasium.farama.org) environment wraps those two sockets, and a
[Stable-Baselines3](https://stable-baselines3.readthedocs.io) PPO agent trains against it.

```
  Android phone (arm64, no root)                Host (this repo)
 ┌───────────────────────────────┐            ┌──────────────────────────────┐
 │ Temple Run + Frida gadget     │  tcp:9999  │ StateClient ─► TempleRunEnv   │
 │  • reads CharacterPlayer/...   │ ─────────► │  (gymnasium.Env)  ─► SB3 PPO  │
 │  • calls Jump/Slide/Turn/...   │ ◄───────── │ ActionClient ◄─ env.step()    │
 │  (state @9999, actions @9998)  │  tcp:9998  │                               │
 └───────────────────────────────┘            └──────────────────────────────┘
        (adb forward maps the phone's 9999/9998/27042 to 127.0.0.1)
```

> **The game is not included.** This repo contains only our code (env, scripts, the Frida agent,
> the APK-build tooling). You must supply your own legally-obtained copy of Temple Run and build the
> instrumented APK yourself — see [docs/SETUP.md](docs/SETUP.md). See [LICENSE](LICENSE) for details.

## Why a physical phone (not an emulator)
The game ships **arm64 only** (IL2CPP). On an x86_64 emulator it runs via ARM→x86 translation, which
**hides `libil2cpp.so` from Frida** so hooks can't resolve; and a modern Android emulator **won't boot
an arm64 image on an x86_64 host** at all. So instrumentation needs native arm64 — a real phone (used
here) or an arm64 cloud host. Details and evidence in [CLAUDE.md](CLAUDE.md) / [docs/GADGET.md](docs/GADGET.md).

## Prerequisites
- A **non-rooted arm64 Android phone** with USB debugging, and `adb` (Android platform-tools).
- **Python 3.12+**.
- Your own **Temple Run** APK + a one-time build of the gadget-injected APK ([docs/SETUP.md](docs/SETUP.md)).
- For the APK build only: a JDK (`zipalign`/`apksigner` from Android build-tools), `frida`, `lief`.

## Install
```bash
pip install -r requirements.txt
pip install -e .
```

## Quickstart
1. **Install the gadget APK** on the phone (one-time build — see [docs/SETUP.md](docs/SETUP.md)).
   On MIUI/HyperOS, `adb install` is often blocked; push the APK and tap it in the Files app.
2. **Bring the device up** (forwards the ports and injects the agent — leave it running):
   ```bash
   python scripts/setup_device.py            # add --serial <id> if multiple devices
   ```
   Wait for `ready: state->9999 action<-9998`.
3. **Open the game and start a run** (or sit on the game-over screen). The env's `reset()` restarts from there.
4. **Validate the env** (no training):
   ```bash
   python scripts/smoke_test.py
   ```
5. **Train / evaluate**:
   ```bash
   python scripts/train.py --timesteps 100000
   python scripts/evaluate.py --model runs/ppo_templerun_final.zip --episodes 10
   python scripts/evaluate.py --random --episodes 5     # baseline
   ```

Use the env directly:
```python
import gymnasium as gym
import templerun_rl                       # registers "TempleRun-v0"
env = gym.make("TempleRun-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Environment spec
- **Observation** — `Box(0, 1, (21,), float32)`: 17 boolean state flags (player state, power-ups,
  obstacle cues `must_jump`/`must_slide`, and turn cues `can_turn_left`/`can_turn_right`/`is_turn`) +
  normalized `velocity`, `path_type`, `path_level`, `facing`. Raw, unbounded `score` is **not** in the
  observation (it drives the reward); the full raw state dict is always in `info["raw_state"]`.
- **Action** — `Discrete(7)`: `0` StrafeLeft · `1` StrafeRight · `2` Jump · `3` Slide ·
  `4` TurnLeft · `5` TurnRight · `6` NoOp. (Bytes `7`/`8` are reset-only and never emitted by the
  policy: `7`=RestartGame resets game logic but leaves the run in game-over limbo, so `reset()` uses
  `8`=RunAgainClicked — the "Rejouer" flow that actually starts a fresh, progressing run.)
- **Reward** — `(score - prev_score) * reward_scale` each step, minus `death_penalty` on the
  terminal step. Defaults `reward_scale=0.01`, `death_penalty=1.0` (configurable in the constructor).
- **Episode** — `terminated` when `is_dead`/`is_game_over`; `truncated` at `max_episode_steps` (default 1000).

## Layout
```
src/templerun_rl/   the package: env.py, client.py, obs.py, constants.py
scripts/            setup_device.py, train.py, evaluate.py, smoke_test.py
frida/              temple_run_hook.js (the agent), gadget_inject.py, inject.py, diag.js
tools/              build_gadget_apk.py (APK patcher)
docs/               SETUP.md, GADGET.md, OFFSETS.md
CLAUDE.md           full project notes (offsets, gotchas, history)
```

## Scaling & caveats
- **One phone = one env**, real-time at ~10 Hz, with short episodes. There is no simulator and no
  parallelism; to scale, add more phones/arm64 hosts, each with its own forwarded ports and env.
- The instrumentation offsets are specific to **Temple Run v1.39.0**; a different version needs a
  re-dump (see [docs/OFFSETS.md](docs/OFFSETS.md)). The Python layer keys only on JSON field names,
  so as long as the 19-field contract holds, no Python changes are needed.

## License
[MIT](LICENSE) for the code in this repository. Not affiliated with Imangi Studios; the game and its
assets are not included or licensed here.
