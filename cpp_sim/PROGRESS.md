# cpp_sim — full C++ port of the Temple Run RL simulator

## Status: GREEN. Ready to switch the trainer over (not switched).

| deliverable | file |
|---|---|
| the port | `cpp_sim/sim.cpp` (~1750 lines) + `cpp_sim/sim.h` |
| build | `cpp_sim/build.sh` |
| ctypes front-end | `cpp_sim/cpp_vec_env.py` — `CppTempleRunVecEnv`, `CppSimGymAdapter` |
| third backend in the gate | `scripts/validate/sim_equivalence.py --backend cpp` (additive; default stays `torch`) |
| draw-order test | `cpp_sim/test_rng_order.py` |
| benchmark | `cpp_sim/bench.py` |

## Results

    python scripts/validate/sim_equivalence.py --backend cpp --det-only
    -> DETERMINISTIC: 5616/5616 pass      (24 tracks x 18 variants x 13 policies)

    python scripts/validate/sim_equivalence.py --backend cpp --stats-only --stats-eps 120
    -> all 8 bands PASS (median 1.021, mean 0.991, event rates 0.890-1.070)

    python cpp_sim/test_rng_order.py            # 60 eps, shared random stream
    -> 60/60 episodes lockstep, 3950 steps, per-step draw counts equal

    python cpp_sim/bench.py
    -> 1.17-2.40 M env-steps/s (full step incl. obs + reward), 166-339x the
       Python reference (7.1 k/s); projected 7.4x end-to-end vs the torch GPU sim

NOTE: the brief quoted "the torch backend scores 1344/1344". The matrix in the
current `sim_equivalence.py` is 24 x 18 x 13 = **5616**; 1344 is a stale figure
from an older matrix. Both backends run the same 5616 today.

## Gate-sensitivity controls (each: patch, rebuild, run, revert)

| injected fault | result |
|---|---|
| lane-lag `alpha` + 1e-9 | **RED** — catches sub-1e-9 numeric drift |
| `crevasse_jump_peak * height * 0.5` | **RED** — world-Y graze branch is reached |
| double-obstacle draw moved BEFORE its `prob > 0` gate | **RED** in `test_rng_order.py` at the first reset (draw count 26 vs 29) |
| `gap_entry_grace * 1.02` | GREEN — no fixed track straddles that boundary |
| `crevasse_jump_peak * height * 0.9` | GREEN — arc_y ~19.2 vs cap 15, 10% does not cross |
| cap-height level gate `>=3` -> `>=2` | GREEN — no level-2 LevelChange apex cell exists in the fixed tracks |

i.e. the tier is extremely sensitive where a track exercises a boundary and blind
where none straddles one. That is a property of the harness, not of the port.

## Design notes

- **The deterministic tier injects fixed tracks, so the procedural generator and
  the whole RNG are invisible to it.** Generation is covered by (a) the
  statistical tier and (b) `test_rng_order.py`, which drives the Python reference
  and the C++ off ONE shared uniform sequence (`sim_rng_replay_set`) and asserts
  equal consumed-draw counts after every step. That is the per-step draw-count
  assertion, and it is what catches a `prob > 0` gating slip.
- Config is ONE global flat `double G[K_COUNT]` addressed by name (`sim_cfg_set`).
  Per-env knobs that domain-rand / `episode_profiles` touch live in `State` as
  `e_*` fields, sampled at reset. Each wrapper re-installs its own config when it
  becomes the active env, so differently-configured envs can coexist.
- `sim_set_track()` installs a global injected track; cell 0 is always the plain
  straight reset cell, matching both `make_cpu` and `make_torch` in the harness.
- `numpy.interp` arithmetic is reproduced exactly
  (`slope=(dy1-dy0)/(dx1-dx0); slope*(x-dx0)+dy0`), NOT the algebraically-equal
  `y0 + t*(y1-y0)` — the latter drifts ~1e-14 and compounds.
- Draw order mirrors the reference exactly **including where the reference draws
  even though the feature is off**: `jump_refuse_strafing_prob` is NOT
  `prob>0`-gated in `sim_env.py`, so the port draws there too.
- Track ring is 64 cells; the port generates `lookahead+1 = 13` ahead, the same
  count as the reference, so procedural cells are generated at the same distances.
- Single-threaded. At 1.7 M steps/s the rollout is 2.7 ms of a 43 ms PPO
  iteration, so threading would buy nothing and GCC 4.6.3 has no `std::thread`.

## Not ported (stated plainly)

- **Replay mode** — `track_source` as a recorded list *driven by* `ext_distance`,
  and `set_state()`. `sim_set_track()` covers the fixed-track injection the
  equivalence harness needs, but the `cell_length_real` / `replay_start_into_cell`
  path used by `replay_diff.py` / `trajectory_diff.py` is absent (same as the
  torch port). Those tools keep using `sim_env.py`.
- `action_delay` (the step-level action buffer). Always 0 in procedural tooling;
  the torch port drops it too.
- `accel_per_sec` linear fallback exists but is only reachable with an empty
  `velocity_curve`.

## Known asymmetries (report, do not "fix")

- `truncated`: `sim_env.py` sets it purely on `elapsed >= cap` (can be True while
  terminated); `sim_env_torch.py` masks it with `& ~terminated`. The C++ exports
  both bits; `CppSimGymAdapter` follows `sim_env.py`, `CppTempleRunVecEnv` follows
  the torch semantics `train_fast.py` relies on for value bootstrapping.
- Domain-randomisation draw count: the CPU reference draws 9 uniforms per reset
  (3 feed knobs nothing reads any more); torch draws 6. The port follows the CPU
  reference, so it is behaviourally identical to it and stream-wise different from
  torch. Irrelevant — DR is only ever compared statistically.
- `episode_profiles` consumes ONE extra draw per reset when profiles are
  configured (`sim_env.py` has no profiles at all, so there is nothing to match).

## Switching `train_fast.py` over (NOT done — the user's call)

Two lines in `scripts/train/train_fast.py`:

```python
# line 31
from templerun_rl.sim_env_torch import TorchTempleRunVecEnv
# ->
sys.path.insert(0, str(Path(__file__).parent.parent))
from cpp_sim.cpp_vec_env import CppTempleRunVecEnv as TorchTempleRunVecEnv
```

Nothing else changes: the constructor kwargs, `reset_tensor()`, `step_tensor()`
(8-tuple, matching shapes AND dtypes), `approach_turn_hint()`,
`observation_space` / `action_space` / `num_envs` and the SB3 `VecEnv` methods are
all identical, and SB3's `PPO(...)` accepts it.

Caveats when you do switch:
- `--device cuda` still works: the sim runs on the CPU and the tensors are moved
  to the requested device. `--cuda-graph` is unaffected (it captures the PPO
  minibatch, not the env).
- Rebuild the library first: `sh cpp_sim/build.sh`.
- The rollout stops being the bottleneck, so `--num-envs` / `--n-steps` become
  nearly free — bigger batches now cost only PPO time. (Per CLAUDE.md, do NOT
  raise `--batch-size` above 512: that trade was measured and rejected.)
