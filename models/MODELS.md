# Shipped policies

All three are stable-baselines3 PPO zips, net `[512, 512]`, trained in the C++ sim
(`cpp_sim/`, `--backend cpp`) and evaluated on a physical phone (Redmi, Android 15,
`com.imangi.templerun` v1.39.0) through the Frida hook `frida/temple_run_hook.js`
(`HOOK_VERSION` 17j/17k).

The metric that matters is the **device** distance distribution, not the sim median —
the sim mispredicted the device four times in the last session (see `CLAUDE.md`).

| file | obs width | device n | median | max | >=8 km | >=10 km |
|---|---|---|---|---|---|---|
| `ppo_v120_obs67_base.zip` | 67 | 14 | **4736 m** | >=15454 m * | 5/14 | 1/14 |
| `ppo_v133_sdonly_10002432_steps.zip` | 67 | 16 | 4708 m | **12007 m** | 3/16 | **2/16** |
| `ppo_v120_chaincoast_10002432_steps.zip` | 66 | 14 | 4736 m | >=15454 m * | 5/14 | 1/14 |

**\* that run did not end — it was cut off.** `capture_calib` stops an episode at 4000 logged
frames; at that frame v120 was at **15,454 m, still alive, still at the velocity cap (210)**, death
type 0. Re-verified from `captures/w18_v120_clean.jsonl` ep5: 4000 frames, strictly monotonic
distance, no duplicate values. So the true distance is **unknown and >= 15,454 m**, and this is the
longest run the project has recorded.

**Consequence for every number above: the medians are right-censored and therefore pessimistic.**
A sweep of the device captures finds several episodes truncated at the cap while the runner was
still at full speed (v120 twice, v126 three times). Any future batch should raise the frame cap
before the median is quoted.

## Which one to run

**The current code is obs67** (`src/templerun_rl/obs.py`: `OBS_LEN = 39`, full width
`39 + 28` action-history columns `= 67`). Only the two obs67 artifacts load against it.

- **`ppo_v120_obs67_base.zip` — the champion.** v120 widened to obs67 by a
  behaviour-preserving zero-column surgery (0/400 action mismatches vs the obs66
  original), so it is byte-for-byte the same policy on the shared columns.
- **`ppo_v133_sdonly_10002432_steps.zip` — the equal-distance alternative, and the
  one to prefer if you care about clean play.** v120's exact world plus only the
  step-down fix: it does not jump the mid->low ground transition any more
  (step-down jump rate 100% -> 0%, measured over 154 step-downs), which is a wasted
  and occasionally fatal action. Distance is a statistical tie with v120
  (median 4708 vs 4736, Fisher p~0.57) with a better tail.
- **`ppo_v120_chaincoast_10002432_steps.zip` — obs66, for the record.** The original
  training output. To load it, set `OBS_LEN = 38` (drop `landing_margin`) in
  `obs.py`; otherwise use the obs67 artifact above.

## How they were trained

- v120 = `configs/train_v120.sh` + `configs/v120_world.json`
  (= the v111 recipe + `chain_coast_kill_prob 0.5`).
- v133 = v120's exact world + `--stepdown-jump-penalty 8.0 --stepdown-explore-prob 0.5`.

Both are 10M steps, `--num-envs 256 --n-steps 16`, ~6 min on a GTX 1080.
**Longer is not better:** the same recipe at 25M scored a device median of 2536 (n=8).

## Deploying

```
python scripts/device/setup_device.py                      # forwards 27042 / 9999 / 9998, injects the gadget
python scripts/device/capture_calib.py models/ppo_v133_sdonly_10002432_steps.zip --episodes 8 --turn-cooldown 0
```

Run exactly one injector and keep its host process alive — if it exits, the in-phone
agent unloads and the state stream goes quiet. Never leave two `capture_calib`
processes on one stream: they double-log the same physical runs.
