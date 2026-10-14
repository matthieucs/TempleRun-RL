<p align="center">
  <img src="assets/banner.png" alt="TempleRun-RL" width="100%">
</p>

<h1 align="center">🏃 TempleRun-RL: Reinforcement Learning on the Real Game, on a Real Phone</h1>

<p align="center">
  <b>A PPO agent trained in a headless simulator, deployed on an unrooted Android phone,<br>
  running past 16 km in the real <i>Temple Run</i>, with no screen scraping and no synthetic touches.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/record-16%2C782%20m-success?style=plastic" alt="Record 16,782 m">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=plastic&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=plastic&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Stable--Baselines3-PPO-8A2BE2?style=plastic" alt="SB3 PPO">
  <img src="https://img.shields.io/badge/Frida-gadget-EF6456?style=plastic" alt="Frida">
  <img src="https://img.shields.io/badge/Android-arm64-3DDC84?style=plastic&logo=android&logoColor=white" alt="Android">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=plastic" alt="MIT"></a>
</p>

<p align="center">
  <img src="assets/demo_10k.gif" alt="The agent crossing 10,000 m" width="300">
  <br>
  <i>The agent crossing the <b>10,000 m</b> mark at top speed, mid-way through a 16,782 m run.<br>
  The bar at the bottom lights up each command the policy sends: turn ↰, tilt, jump ↑, slide ↓, turn ↱.</i>
</p>

---

## 📣 News

- **[2026-10-14]** 🏆 **New record: 16,782 m** on the device (policy `v120`), passing 10 km after about 5 minutes.
  Recorded with `scripts/device/record_overlay.py`; the GIF above is taken from that run.
- **[2026-07-31]** 📦 Pretrained policies `v120` and `v133` released in [`models/`](models/MODELS.md), trained
  with a C++ simulator that runs about 27k steps/s.
- **[2026-07-06]** 🎯 **10,000 m reached for the first time** on the real game.

## ✨ Overview

*Temple Run* is a closed, arm64-only Unity game. This project plays it with a neural network:

- 🔌 **No root, no pixels.** A [Frida](https://frida.re) gadget injected into the APK reads the game state
  directly from memory (player, track cells, obstacles, turns) and executes actions by calling the game's
  own methods. Both go over TCP at 30 Hz; the policy acts at 10 Hz.
- 🧪 **Sim-to-real.** Training happens in a headless simulator of the track generator and runner physics,
  **calibrated against device captures** (jump arcs, turn windows, crevasse ledges, input latency).
  A C++ port, checked step for step against the Python reference, trains 10M steps in about 6 minutes.
- 🤖 **The network decides everything.** The phone-side hook only translates actions, for example routing a
  turn through the game's own swipe queue. It contains no hand-written playing rules.

```
  Android phone (arm64, no root)                Host (this repo)
 ┌───────────────────────────────┐            ┌──────────────────────────────┐
 │ Temple Run + Frida gadget     │  tcp:9999  │ StateClient ─► TempleRunEnv  │
 │  • reads CharacterPlayer/...  │ ─────────► │  (gymnasium.Env)  ─► PPO     │
 │  • calls Jump/Slide/Turn/...  │ ◄───────── │ ActionClient ◄─ env.step()   │
 └───────────────────────────────┘  tcp:9998  └──────────────────────────────┘
        adb forward maps the phone's 9999 / 9998 / 27042 to 127.0.0.1
```

## 🏆 Pretrained Policies

Stable-Baselines3 PPO, MLP `[512, 512]`, observation width 67. Distances are **measured on the device**:

| Policy | Device runs | Median | Best | ≥ 10 km |
|---|:---:|:---:|:---:|:---:|
| [`ppo_v120_obs67_base.zip`](models/) **(champion)** | 14 | **4,736 m** | **16,782 m** † | 1 / 14 |
| [`ppo_v133_sdonly_10002432_steps.zip`](models/) | 16 | 4,708 m | 12,007 m | **2 / 16** |

<sub>† From the 2026-10-14 record session (20 attempts, separate from the 14-run benchmark). The older
benchmark medians are pessimistic: the capture harness stopped episodes at 4,000 frames, which cut off a
v120 run that was still alive at 15,454 m. Details in [`models/MODELS.md`](models/MODELS.md).</sub>

`v133` is `v120`'s training world plus one fix: it no longer jumps pointlessly at mid-to-low step-downs
(100% → 0% over 154 step-downs). Distance is a statistical tie.

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/matthieucs/TempleRun-RL.git
cd TempleRun-RL
pip install -r requirements.txt
pip install -e .
```

You also need `adb`, a **non-rooted arm64 Android phone** with USB debugging enabled, and your own copy of
Temple Run v1.39.0, patched once with the Frida gadget ([`docs/SETUP.md`](docs/SETUP.md)).

> [!IMPORTANT]
> **The game is not included.** This repo ships only our own code. Supply a legally obtained APK and build
> the instrumented copy yourself with `tools/build_gadget_apk.py`.

### 2. Bring the phone up

Open the game, then forward the ports and inject the agent. **Leave this process running**: if it exits,
the in-game agent unloads.

```bash
python scripts/device/setup_device.py --serial <your-serial>
```

### 3. Watch a pretrained policy play

```bash
python scripts/device/capture_calib.py models/ppo_v120_obs67_base.zip --episodes 5 --turn-cooldown 0 --max-steps 8000
```

### 4. Record a video with the action overlay

```bash
python scripts/device/record_overlay.py models/ppo_v120_obs67_base.zip \
    --serial <your-serial> --episodes 1 --out-dir videos/ --max-steps 8000
```

This mirrors and records the phone with [scrcpy](https://github.com/Genymobile/scrcpy), logs every
action, then draws the command bar onto the video (`ffmpeg` required).

## 🔬 Training

Training runs in the simulator, not on the phone. The phone is only used to **measure** fidelity and to
evaluate finished policies.

```bash
# build the C++ simulator (once, and after any sim change)
CXX=g++ sh cpp_sim/build.sh

# gate: the C++ port must match the Python reference step for step
python scripts/validate/sim_equivalence.py --backend cpp --det-only
python cpp_sim/test_rng_order.py

# train v133's exact recipe, starting from the shipped v120 weights
bash configs/train_v133.sh

# evaluate on the phone
python scripts/device/capture_calib.py runs/ppo_next_final.zip --episodes 8 --turn-cooldown 0 \
    --out captures/next_device.jsonl
```

<details>
<summary><b>Full training command and why each setting matters</b></summary>

```bash
python scripts/train/train_fast.py \
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
  --sim-config "$(cat configs/v120_world.json)" \
  --ckpt-every 2000000 --out runs/ppo_next
```

| Flag | Why |
|---|---|
| `--backend cpp` | 7× the PyTorch sim (26–28k vs 3.7k steps/s) |
| `--num-envs 256 --n-steps 16` | keep `num_envs × n_steps = 4096`; widening `num_envs` alone is faster but much worse |
| `--landing-hazard-penalty 0.30` | penalizes a jump whose arc lands on a hazard (the fix behind v111) |
| `--stepdown-jump-penalty 8.0 --stepdown-explore-prob 0.5` | removes the pointless step-down jump (v133). Both halves are needed: at a 100% jump rate the alternative is never sampled |
| `--turn-explore-prob 0.5` | guided exploration for the early turn swipe, which fixed turns being missed under latency |
| `--timesteps 10000000` | a measured ceiling: 20–25M runs plateaued or did worse on the device |

</details>

<details>
<summary><b>Judging a candidate</b></summary>

Device distances vary a lot (v133 ranged from 1,506 to 12,007 m), so an 8-run median detects a real
20% gain only about 11% of the time. Judge **per-encounter rates** instead: one batch has ~400 hazards,
~60 crevasses and ~900 jumps.

```bash
python scripts/analysis/detect_patterns.py   captures/next_device.jsonl  # named mistakes per km
python scripts/analysis/chain3_forensic.py   captures/next_device.jsonl  # 3-obstacle cascades
python scripts/analysis/takeoff_timing.py    captures/next_device.jsonl  # crevasse takeoff vs speed
python scripts/analysis/stepdown_forensic.py captures/next_device.jsonl
```

Before using phone time, check that the training world matches the device: score the current champion in
the sim and compare it to its own device median (`scripts/analysis/hole_landing_sim.py`).

</details>

## 📁 Repository Layout

```
src/templerun_rl/     the package: env, TCP clients, observation, reward, Python sim (the reference spec)
cpp_sim/              C++ simulator used for training (~27k steps/s) + its Python vec-env wrapper
frida/                temple_run_hook.js (in-game agent) and gadget_inject.py
models/               shipped policies + MODELS.md
configs/              the v120 sim world and the v120 / v133 training recipes
scripts/
  device/             phone bring-up, capture, evaluation, screen recording + overlay
  train/              trainers (train_fast.py = C++ / GPU path)
  validate/           sim-vs-device fidelity checks, sim equivalence, checkpoint evaluation
  analysis/           forensics on device captures (deaths, jumps, crevasses, chains)
  probes/             one-off device measurements (turn window, jump latency, …)
experiments/          archived research: per-version recipes, sim configs and one-off studies
docs/                 setup, gadget, offsets, sim contract; docs/archive/ = full project history
tools/                build_gadget_apk.py (APK patcher)
```

[`CLAUDE.md`](CLAUDE.md) is the working lab notebook: offsets, device-measured mechanics, and every
experiment along with why it worked or failed.

<details>
<summary><b>Environment spec (observation, actions, reward)</b></summary>

**Observation:** 39 features in `[0, 1]`, plus 4 × 7 action-history columns = **67**.

| Group | Fields |
|---|---|
| Player state | `is_dead`, `is_jumping`, `is_sliding`, `is_on_ground`, `is_stumbling`, `is_strafing`, `is_game_over` |
| Obstacle cues | `must_jump`, `must_slide`, 1/2/3-cell look-ahead `next*_must_jump`, `next*_must_slide`, `next_wide_gap`, `next_ground_change_higher` |
| Turn cues | `can_turn_left/right`, `is_turn`, `next_is_turn`, `next_can_turn_left/right` |
| Scalars | `velocity`, `path_type`, `path_level`, `facing`, `next_path_level`, `next2_path_level` |
| Lane & timing | `cell_progress`, `lane`, `narrow_target_lane`, `next2_narrow_lane`, `predicted_landing` |
| Derived cues | `predicted_jump`, `predicted_turn`, `landing_hazard`, `slide_landing_hazard`, `landing_margin` |

The authoritative list is [`src/templerun_rl/obs.py`](src/templerun_rl/obs.py).

**Actions:** `Discrete(7)`: `0` strafe-left, `1` strafe-right, `2` jump, `3` slide, `4` turn-left,
`5` turn-right, `6` no-op. Lateral moves are analog tilt (the hook replaces the accelerometer read);
turns go through the game's own swipe queue, so they fire at the corner with native timing.
`reset()` sends `8` (run again) then `9` (play), which restarts from either the game-over screen or the menu.

**Reward:** score delta × `reward_scale`, minus `death_penalty` on death, plus the shaping terms listed
in the training table.

```python
import gymnasium as gym
import templerun_rl                      # registers "TempleRun-v0" (device) and "TempleRunSim-v0"

env = gym.make("TempleRun-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

</details>

<details>
<summary><b>Why a physical phone, multiple phones, and staying up</b></summary>

- **No emulator:** the game is arm64-only (Unity IL2CPP). On x86 emulators, ARM translation hides
  `libil2cpp.so` from Frida, so a real arm64 device is required.
- **Several phones:** run `setup_device.py --device-index i` once per phone; host ports are offset by
  `100 × i`.
- **Resilience:** the clients reconnect automatically, and `scripts/device/watchdog.py` re-applies keep-awake
  and re-injects the agent if the app restarts.
- **Version:** offsets are specific to Temple Run **v1.39.0**; other versions need a re-dump
  ([`docs/OFFSETS.md`](docs/OFFSETS.md)).

</details>

## 📄 License

[MIT](LICENSE) for the code in this repository. Not affiliated with Imangi Studios; the game and its assets
are not included or licensed here.

## 🙏 Acknowledgements

Built on [Frida](https://frida.re), [Stable-Baselines3](https://stable-baselines3.readthedocs.io),
[Gymnasium](https://gymnasium.farama.org), [Il2CppDumper](https://github.com/Perfare/Il2CppDumper),
[LIEF](https://lief.re) and [scrcpy](https://github.com/Genymobile/scrcpy).
