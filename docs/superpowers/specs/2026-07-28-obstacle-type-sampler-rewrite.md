# Obstacle type sampler — rewrite to draw-then-substitute

**Date:** 2026-07-28
**Baseline frozen at:** `backups/sim_baseline_2026-07-28_v111/`
**Status:** spec, ready to implement

---

## 1. Why

Read from the game (`il2cpp_work/output/dump.cs`), obstacle generation is **constraint-filtered
rejection sampling**, not independent sampling from a weight table:

| Evidence (all SEEN in dump.cs) | |
|---|---|
| `PathElement.ACertainTypeCheck(PathType currentType, PathType parentType)` | @283485, RVA `0x12E3DF4`, 84 B — an explicit **pairwise** admissibility rule |
| `PathType OriginalPick` 0x98, `PathType FinalPick` 0x9C | the generator **draws a type then substitutes it** |
| `int SpacesSinceLastObstacle` 0x7C, `int BackToBackObstacleCount` 0x80 | per-node placement memory |
| `GetRandomPathElementForSide` @283488, RVA `0x12E3E48`, 3500 B (~875 instr) | a hardcoded `Random.value < literal` chain, far too large for a table lookup |

Confirmed **absent**, file-wide: any authored pattern/template/combo collection, any
`ObstacleType[]`/`List<PathType>`, any string literal containing "obstacle", and **any per-type
probability field**. The game has named probability statics only for bonus items
(`sProbabilityCoinBonus`, …); obstacle numbers are compiled-in literals. Per-type control is only
the boolean gates `sAllowPathHoles` / `sAllowTreeSlide` / ….

**The structural key.** Holes, narrows, flames and level-changes are `PathType` — the *cell
geometry* — and must survive the pairwise + fit + spacing filters. Trees (Stumble / Jump / Slide)
are `ObstacleType`, placed onto an already-chosen straight by a **separate `AddObstacle` pass that
those filters never touch.** One mechanism therefore predicts both device deviations: a filtered
type is pushed **below** chance, an unfiltered one sits **above** it.

## 2. What is wrong today, measured

Joint pair basis, structure units (adjacent same-type cells collapsed), late ramp (`_distance_real`
3000), config verified applied:

| statistic | sim now | sim at `pair_same_type_prob` = 1.0 | **device** |
|---|---|---|---|
| slide-repeat share of all pairs | 7.9% | **10.4% (saturated)** | **17.9%** |
| slide-led share of all pairs | 30.8% | 33.3% | **50.7%** |
| Hole→Hole share of all pairs | 7.9% | — | **6.1%** (independent 10.3%) |
| TreeSlide→TreeSlide | — | — | 9.8% (independent 1.5%) |
| Flame→Flame | — | — | 8.1% |
| same-type total | — | — | 24.6% |

`pair_same_type_prob` **cannot** close this: it only fires when the previous obstacle was a slide,
and slides lead too few pairs. It is a symptom-level patch on the wrong shape.

⚠️ **Flame is a `PathType` yet `Flame→Flame` is *elevated* (8.1%).** So the pairwise rule is **not**
a blanket anti-repeat — it is type-specific, consistent with `ACertainTypeCheck` being only ~21
instructions (one special case, not a matrix). Do not assume "geometry ⇒ suppressed".

## 3. What to build

Shape from the game; numbers calibrated to the device, since the literals live in native code.

**Stage A — geometry types** (`OBST_HOLE`, narrow, `FlameTower`, level-change): draw from the
existing weights, then apply a **pairwise rejection test** against the previously emitted structure
type. On rejection, **re-draw once** (the substitute). One re-draw only — mirrors `OriginalPick` →
`FinalPick`, and it is what makes the floor reachable.

**Stage B — trees** (`TreeStumble`, `TreeJump`, `TreeSlide`): placed onto straights, **not** subject
to the pairwise test. This is the half that produces the elevated slide clustering for free.

### New config keys — every one default-off / no-op

```
"type_reject_model": False,        # master switch; when False draw NO rng (bit-identical stream)
"type_reject": {},                 # {"prev,cur": p_reject}, e.g. {"1,1": 0.45}
"type_reject_redraws": 1,
```

`pair_same_type_prob` and `pair_anti_repeat_prob` stay for backward compatibility and must remain
usable; the new model is selected only by `type_reject_model`.

### Calibration targets

Fit `type_reject` so the emitted **joint** statistics land on the device column above, iteratively —
the rejection probability is *not* the emitted share (same lesson as `calib_mix.py`). Priorities in
order: slide-led 50.7% → slide-repeat 17.9% → Hole→Hole 6.1% → same-type total 24.6%.

## 4. Hard constraints

**RNG DRAW ORDER IS THE WHOLE RISK.** The CPU reference (`src/templerun_rl/sim_env.py`) and the C++
trainer (`cpp_sim/sim.cpp`) must draw the *same number of values in the same order*, or the tracks
diverge silently and the trainer optimises a different world. Therefore:

* Gate on `type_reject_model` **first**, before any `random()`, so the OFF path draws nothing and
  every existing result stays reproducible.
* Draw the rejection roll in exactly one place, in the same position in both implementations.
* This is the one change where the equivalence gate genuinely earns its keep — it exists precisely
  to catch draw-order drift. Both must pass:
  * `python scripts/sim_equivalence.py --backend cpp --det-only` → currently 6864/6864
  * `python cpp_sim/test_rng_order.py` → 60/60 lockstep
* Add an equivalence variant `type_reject` exercising the new path with a non-trivial matrix.

**Files to change, in lockstep:** `src/templerun_rl/sim_env.py` (the specification),
`cpp_sim/sim.cpp` (the trainer), `cpp_sim/cpp_vec_env.py` (config plumbing; note `_TABLE_KEYS` for
dict-valued config), `scripts/sim_equivalence.py` (new variant). Rebuild with `sh cpp_sim/build.sh`.

## 5. Measurement harness — read this before writing any probe

Four checks in one session silently passed while testing nothing. Do not repeat them:

* `gym.make("TempleRunSim-v0", config=...)` — **`config=`, not `sim_config=`.** An unknown kwarg is
  swallowed and the env silently runs defaults, so every dose looks identical.
  `CppTempleRunVecEnv` also takes `config=`.
* **Every probe must `assert env.cfg[k] == v` after construction, before measuring.**
* Cell dicts key the obstacle as **`obstacles`** (plural), not `obstacle`.
* Drive generation with `env._next_cell()` — the single point every generated cell passes through.
  Calling `_next_cell_procedural()` directly bypasses the trackers and makes knobs look inert.
* Count in **structure units** (collapse adjacent same-type cells): a 2-cell `HoleLong` otherwise
  reads as a Hole→Hole pair and inflates the rate to ~30%.
* Compare on the **joint** basis (share of *all* consecutive pairs), which is how the device numbers
  above are defined. The conditional rate is a different quantity (≈ the type weight).
* Set `env._distance_real = 3000.0` to reach the late ramp regime where cascades actually occur.
* A NoOp rollout dies immediately and generates almost no track — drive the generator directly.

## 6. Done when

1. Both gates green (6864/6864 and 60/60).
2. A calibration table showing emitted joint statistics vs the device targets in §2.
3. `type_reject_model: False` reproduces the frozen baseline **exactly** (same medians, same stream).
4. No change to any default — the champion's world is untouched until a policy is trained on the new
   one and beats it.
