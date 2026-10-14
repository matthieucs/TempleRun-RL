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

---

## Pretrained policies

Two finished policies ship in [`models/`](models/MODELS.md) (SB3 zips, `[512,512]` net, obs width 67):

| file | device runs | median | max | >=10 km |
|---|---|---|---|---|
| `ppo_v120_obs67_base.zip` | 14 | **4736 m** | >=15454 m * | 1/14 |
| `ppo_v133_sdonly_10002432_steps.zip` | 16 | 4708 m | 12007 m | **2/16** |

<sub>\* that run never ended — the capture harness stops at 4000 frames, and v120 was still alive
at the velocity cap when it did. The medians are right-censored and so are pessimistic; see
[models/MODELS.md](models/MODELS.md).</sub>

`v120` is the champion on accumulated evidence; `v133` is its exact world plus the step-down fix
(it no longer jumps the mid->low ground transition: 100% -> 0% over 154 step-downs) and ties it on
distance with a better tail. Run either on the phone with:

```bash
python scripts/capture_calib.py models/ppo_v133_sdonly_10002432_steps.zip --episodes 8 --turn-cooldown 0
```

See [models/MODELS.md](models/MODELS.md) for the obs-width caveat and the full numbers.

---

## Current training recipe (2026-07-31)

Training happens in a **headless simulator**, not on the phone — the phone is only used to *measure*
fidelity and to evaluate a finished policy. The simulator has a C++ backend that is step-exact with
the Python reference, which makes a 10M-step run take about **6 minutes**.

```bash
# 0) build the C++ sim once (and after any sim change)
sh cpp_sim/build.sh

# 1) MANDATORY gate — the C++ port must match the Python reference exactly
python scripts/sim_equivalence.py --backend cpp --det-only   # -> 6552/6552 pass
python cpp_sim/test_rng_order.py                             # -> 60/60 lockstep

# 2) train -- this is v133's EXACT recipe, reproducible from the shipped weights:
#    v120's world (_v120_cfg.json) + ONLY the step-down fix.
python scripts/train_fast.py \
  --backend cpp \
  --init-from models/ppo_v120_obs67_base.zip \
  --timesteps 10000000 \
  --net-arch 512,512 --action-history-len 4 --frame-stack 1 \
  --num-envs 256 --n-steps 16 --batch-size 512 --cuda-graph \
  --turn-linger 3 \
  --ent-coef 0.03 --ent-coef-final 0.005 --ent-schedule cosine \
  --learning-rate 1.2e-4 --lr-schedule cosine --lr-final 5e-6 \
  --narrow-hold-bonus 0.18 --aim-center-weight 1.0 \
  --slide-jump-penalty 0.0 --idle-action-penalty 0.0 \
  --airborne-action-penalty 0.08 --strafe-penalty 0.05 \
  --landing-hazard-penalty 0.30 \
  --stepdown-jump-penalty 8.0 --stepdown-explore-prob 0.5 --stepdown-explore-anneal-frac 0.5 \
  --turn-buffer-bonus 0.5 --turn-explore-prob 0.5 --turn-explore-anneal-frac 0.5 \
  --curriculum-max-start-dist 5000 --turn-approach-buffer --stepdown-bonus-gate \
  --sim-config "$(cat scratchpad/_v120_cfg.json)" \
  --ckpt-every 2000000 --out runs/ppo_next

# 3) evaluate on the phone (hook 17j or 17k)
python scripts/capture_calib.py runs/ppo_next_final.zip --episodes 8 --turn-cooldown 0 \
    --out captures/next_device.jsonl
```

### Why these settings

| flag | why |
|---|---|
| `--backend cpp` | 7.07× the torch sim at matched geometry (26–28k vs 3.7k steps/s) |
| `--num-envs 256 --n-steps 16` | **keep `num_envs × n_steps = 4096`.** Widening `num_envs` alone is faster but far worse — the invariant is updates per *data collection*, not per sample |
| `--batch-size 512` | measured; larger was tried and rejected |
| `--landing-hazard-penalty 0.30` | prices a jump whose arc lands on a hazard — the fix that produced v111 |
| `--stepdown-jump-penalty 8.0 --stepdown-explore-prob 0.5` | stops the useless jump at a mid->low step-down (v133). **Both halves are required** — at a 100% jump rate the alternative is never sampled, so a penalty alone is a constant offset that advantage normalisation removes |
| `--timesteps 10000000` | measured ceiling, not a budget: a 25M run scored a device median of 2536 m — below its 10M counterpart — and a separate 20M arm plateaued at parity. Essentially all learning happens by ~3M; the rest is LR annealing |
| `--ent-coef … --ent-schedule cosine` | re-opens exploration when the world changed under a warm-started policy, then anneals |
| `--turn-explore-prob 0.5` | guided exploration for the turn pre-swipe (solved the turn dead-letter) |

### How a candidate is judged

Distances are extremely spread (v133 ran 1,506 → 12,007 m), so an 8-episode **median has ~11% power**
to see a 20% gain. Judge on **per-encounter rates** instead — one batch contains ~400 hazard
encounters, ~60 crevasse crossings and ~900 jumps:

```bash
python scripts/detect_patterns.py captures/next_device.jsonl     # named mistakes, per km
python scratchpad/chain3_forensic.py captures/next_device.jsonl  # 3-obstacle cascades
python scripts/takeoff_timing.py captures/next_device.jsonl      # crevasse takeoff slope ± s.e.
python scratchpad/stepdown_forensic.py captures/next_device.jsonl
```

Before spending phone time, check the world itself is device-true — score the current champion in it
and compare to *its own device median*. That check is what catches an over-hard sim
(`scratchpad/hole_landing_sim.py::rollout`).

---

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

### Observation — `Box(0, 1, (39,))`

A normalized vector of **39 features** (all scaled to `[0, 1]`): 23 booleans read straight off the
game, then 16 derived columns. Deployed width is **67** = `39 + 4x7` action-history columns.

| Group | Fields |
|---|---|
| **Player state** (bool) | `is_dead`, `is_jumping`, `is_sliding`, `is_on_ground`, `is_stumbling`, `is_strafing`, `is_game_over` |
| **Obstacle cues** (bool) | `must_jump`, `must_slide`, and the 1/2/3-cell lookahead `next_must_jump`, `next_must_slide`, `next_wide_gap`, `next_ground_change_higher`, `next2_must_jump`, `next2_must_slide`, `next3_must_jump`, `next3_must_slide` |
| **Turn cues** (bool) | `can_turn_left`, `can_turn_right`, `is_turn` + lookahead `next_is_turn`, `next_can_turn_left`, `next_can_turn_right` |
| **Scalars** | `velocity`, `path_type` (coarse category), `path_level`, `facing` |
| **Lane & timing** | `cell_progress`, `lane`, `narrow_target_lane`, `next2_narrow_lane`, `predicted_landing` |
| **Altitude** | `next_path_level`, `next2_path_level` |
| **Derived cues** (host-computed, no hook change) | `predicted_jump` (step-up takeoff), `predicted_turn` (approach-cell deadline margin), `landing_hazard` (my jump arc ends on a must-jump hazard), `slide_landing_hazard`, `landing_margin` |

The derived cues are where most of the device gains came from: `landing_hazard` + an explicit price
produced v111, and each one is computed from already-streamed fields, so adding one needs **no hook
change**. Two zero-init columns for a 6-cell hazard horizon exist but are **off**
(`obs.HORIZON_OBS = False`) — three arms left them unlearned.

- **Power-ups were removed from the observation** (2026-07-04) — they were never load-bearing.
- The raw, unbounded **`score` is deliberately excluded** — it drives the reward, not the policy. The
  full raw state dict is always available in `info["raw_state"]`.
- **Width scales with the wrappers:** `+7` per action-history frame (`action_history_len`) and `xframe_stack`.
  The authoritative field list lives in [`src/templerun_rl/obs.py`](src/templerun_rl/obs.py), and
  `scripts/sim_equivalence.py` fails loudly if the three sims disagree on it.

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
src/templerun_rl/   the package: env.py, client.py, obs.py, reward.py, constants.py
                    sim_env.py      <- the CPU reference simulator (the SPECIFICATION)
                    sim_env_torch.py   RETIRED from the loop; not gated, will drift
cpp_sim/            sim.cpp/.h      <- the TRAINER (~27k steps/s), cpp_vec_env.py, build.sh
scripts/            setup_device.py, train_fast.py, eval_cpp.py, eval_paired.py,
                    sim_equivalence.py (the gate), gate_battery.py, capture_calib.py,
                    detect_patterns.py, cue_response.py, takeoff_timing.py
scratchpad/         per-arm training recipes (train_v*.sh), sim configs (_v*_cfg.json)
                    and one-off forensics — how every result in CLAUDE.md is reproduced
models/             the two shipped policies + MODELS.md
frida/              temple_run_hook.js (the agent), gadget_inject.py, inject.py, diag.js
tools/              build_gadget_apk.py (APK patcher)
docs/               SETUP.md, GADGET.md, OFFSETS.md, SIM_CONTRACT.md, SIM_MODEL_SPEC.md,
                    CLAUDE_FULL_HISTORY.md
CLAUDE.md           full project notes (offsets, gotchas, every arm and why it failed)
```

Three simulators exist on purpose: **`sim_env.py` is the specification**, `cpp_sim/` is the trainer,
and `scripts/sim_equivalence.py` proves they are step-exact before any arm is trusted.

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
