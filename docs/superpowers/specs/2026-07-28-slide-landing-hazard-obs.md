# `slide_landing_hazard` — obs 65 → 66

**Date:** 2026-07-28
**Status:** IMPLEMENTED (obs.py + cpp_sim + torch mirror), surgery 0/400, v117 training

---

## 0. CORRECTIONS after implementing — read before §1

Three things measured after this spec was written change what to expect from it. All were found by
running the check against the **shipped encoder** rather than against the derivation.

**(a) THE CUE IS REAL AND SHARP.** `scratchpad/slh_device_check.py` runs `obs.state_to_obs` over
every takeoff in 176 device captures and cross-tabs the emitted column against the outcome:

| `slide_landing_hazard` at takeoff | takeoffs | slide-type deaths | rate |
|---|---|---|---|
| 0.0 | 26,803 | 80 | **0.30 %** |
| 0.5 | 16 | 0 | 0 % |
| 1.0 | 51 | 8 | **15.69 %** |

**52.6× lift**, and takeoff cp is **0.50** when the cue fires vs **0.28** when it does not — the
policy takes off LATE exactly when its arc is about to end on a slide hazard. This survives the real
span law, the level scaling and the integer rounding, which is what the derivation could not test.

**(b) ⚠️ THE CEILING IN §1 IS WRONG — it is ~3 %, not ~33 %.** Over 1077 pooled device deaths the cue
flags **32 (3.0 %)** one jump in advance. The 33 % came from a 24-episode **sim** rollout whose
flame-death share (33 %) is nothing like the device's (Burnt 2.0 % + Slide 5.1 % = 7.1 %). On v111's
own captures it is 3-of-8 slide-type deaths and 1-of-8 flagged — consistent with the mode mattering
more as a policy survives into flame territory, but n = 8 has ~11 % power and is a hint, not a
number. **Expect a small effect and judge it with an instrument that can see one.**

**(c) ⚠️ EXPOSURE IS NOT THE GAP, so the drill was dropped.** The sim already fires this cue on
**1.12 %** of takeoffs vs the device's **0.19 %** — it over-serves the situation **6×**. A jumpslide
drill profile (`scratchpad/_v117_cfg.json`, `jumpslide_prob` 0.55 in 9.2 % of episodes) moved the
rate **1.12 % → 1.06 %** = inert on its own precondition, and was cancelled before it cost any steps.
This also explains v116's null: its jumpslide drill added practice for a situation already
over-served, while carrying the `airborne_slide_kill_prob` 0.24 softening that cost v115 −9.7 %.
⇒ **v117 = v111's exact recipe and exact world + the column only.** One change, clean attribution.

**New tool:** `scripts/cue_response.py` — flips one obs column on states the policy actually visits
and reports dP(action). Validated against a known value: v111's `landing_hazard` reads **−0.01335**
here vs the **−0.01163** on record. This is the 2M kill criterion, and it is what the horizon obs
never had.

---
**Champion at time of writing:** v111 (`runs/ppo_v111_lh030_final.zip`, obs65, 512-net, hook 17j/17k)

---

## 1. Why — derived from device outcomes, not from intuition

The champion's deaths, measured on a 24-episode sim rollout: **Hole 41.7%, Flame 33.3%**, and
**58% of deaths sit inside a 3-hazard cascade**. Split by whether the runner was airborne:

| killing hazard | n | airborne at death | had jumped < 1.2 s before |
|---|---|---|---|
| Flame | 8 | **88%** | 88% |
| TreeSlide | 2 | 50% | 50% |
| Hole | 10 | 40% | 100% |

So **~33% of all deaths are "airborne when a slide hazard arrived."** You cannot slide while
airborne, and those takeoffs are **100% forced** — a `must_jump` demanded them (0 discretionary,
measured in both sim and device). The policy is not choosing badly; it is landing badly.

### The discriminator, from 100 real device events

Pooled over 45 device captures, every takeoff whose predicted arc spans a must-slide hazard:

| | n | takeoff cp | span | hazard at | **arc slack** | pressed slide after |
|---|---|---|---|---|---|---|
| **survived** | 73 | 0.53 | 2.61 | +2.0 | **+1.54** | 34% |
| **died** | 27 | 0.62 | 2.69 | +3.0 | **+0.21** | 89% |

`arc_slack = (cell_progress + span) − hazard_offset` — how far past the hazard cell the arc reaches.

* Survivors **fly past** the hazard and land beyond it.
* Victims' arcs **end on it** — they come down exactly where the flame is.
* **89% of victims pressed SLIDE and were still airborne.** They reacted correctly and died anyway.

The jump is forced, but **the timing is not**: 0.53 vs 0.62 takeoff cp separates living from dying.
That is the learnable counterfactual.

## 2. What to build

`slide_landing_hazard`, mirroring the existing `landing_hazard` exactly (obs.py:86):

```
span    = LAND_SPAN_A * v + LAND_SPAN_B          (height-scaled: LAND_SPAN_LEVEL_SCALED)
landing = cell_progress + span
offset  = int(landing)
value   = 1.0  if the must_SLIDE ladder is True at `offset`      (landing on a slide hazard)
          0.0  if clear
          0.5  if offset is beyond the 3-cell lookahead          (unknown)
```

Ladder = `must_slide`, `next_must_slide`, `next2_must_slide`, `next3_must_slide` — **all already
streamed**. Host-derived, so **NO HOOK CHANGE**. Hook 17k is unaffected.

## 3. Where

| file | change |
|---|---|
| `src/templerun_rl/obs.py` | add `"slide_landing_hazard"` to `OBS_FIELDS` right after `"landing_hazard"`; add the derivation next to `landing_hazard`'s |
| `cpp_sim/sim.cpp` | add the matching `R_*` obs slot and fill it in the obs builder; bump `OBS_BASE` 37 → 38 |
| `scratchpad/surgery_v111_obs66.py` | v111 (65) → 66 by inserting ONE zero column at the new index; must print **0/400 action mismatches** |
| `scripts/train_fast.py` | nothing — no new flag |

⚠️ `obs.py` and the C++ `OBS_BASE` must be changed together; a mismatch shows instantly as a shape
error (37 vs 38).

## 4. How to judge it

1. **Cue counterfactual first, before crowning anything.** Flip the column on REAL rollout states and
   measure dP(JUMP). Reference scale from this project: `landing_hazard` in the mature v111 responds
   **−0.01163**; an eroded cue reads ~−0.0005; the horizon obs never left **−0.00005** against a
   +0.0030 bar. Random probe vectors read ~0 and are meaningless — use on-distribution states.
2. **`python scripts/eval_paired.py v111.zip v117.zip --seeds 8 --episodes 128
   --curriculum-max-start-dist 5000 --sim-config "$(cat scratchpad/_v114_cfg.json)"`**
   Paired s.d. is ~108 m, so 8 seeds resolve ~2.5%.
3. Targeted metric: the airborne-into-slide death share (was ~33% of deaths) and arc slack at
   takeoff — the policy should shift toward the survivors' +1.5, i.e. take off EARLIER.
4. Only then device, and only as an **alternating same-session A/B** on a cooled phone.

## 5. The precedent, and the honest risk

**For:** this is the fourth use of "collapse velocity-dependent arithmetic the policy demonstrably
does not do into one scalar." `predicted_landing` broke the turn wall; `predicted_jump` fixed
crevasse takeoff; `landing_hazard` produced v111. It is one column, host-derived, cheap to falsify.

**Against:** the horizon obs (`dist_next_must_jump/slide`) was also "obviously needed" and died
across three arms including a 40M consolidation — the policy learned "be twitchier in the dark",
never "the hazard is N cells away". The difference here is that this cue **encodes the counterfactual
directly** (will I land on it?) rather than a raw distance requiring conversion, and it is measured
to discriminate real device outcomes 73-vs-27 before being built. If the counterfactual probe in
step 1 reads flat at 2M steps, **kill it early** — that is what the probe is for.

## 6. Traps that already cost time this session

* `gym.make("TempleRunSim-v0", config=...)` — **`config=`, not `sim_config=`**; unknown kwargs are
  swallowed and the env silently runs defaults. `CppTempleRunVecEnv` also takes `config=`.
* Every probe must `assert env.cfg[k] == v` after construction, before measuring.
* Cell dicts key the obstacle as **`obstacles`** (plural).
* **Evaluate with `scripts/eval_cpp.py`, never `eval_ckpt_torch.py`** — the latter runs the retired
  torch sim, which silently lacks anything added to the C++ trainer.
* `eval_cpp` must mirror train_fast's BASE cfg (`turn_approach_buffer`, `turn_linger_frames`,
  `domain_rand`, `curriculum_max_start_dist`) or v111 dies at the first junction (109 m vs 2400).
* Checkpoint rule, fixed in advance: **best training `ep_dist`, ties to the later checkpoint.**

## 7. State at handoff

* **v111 remains champion.** Five fidelity arms today produced no gain: v113 negative, v114 parity,
  v115 −9.7%, v116 −2.1% (t = −2.41, RESOLVED worse).
* Sim worlds: `scratchpad/_v114_cfg.json` (device-calibrated mix), `_v115_cfg.json` (+ airborne
  slide 0.24), `_v116_cfg.json` (+ 3 drills). The drills are **not** the lever.
* Baseline frozen at `backups/sim_baseline_2026-07-28_v111/`.
