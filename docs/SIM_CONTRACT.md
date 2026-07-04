# Sim/Fidelity Contract — shared interface for logger ↔ sim ↔ harness

Three components are built against THIS contract so they integrate cleanly:
A) `frida/trace_logger.js` + `scripts/record_traces.py` — record real phone play.
B) `src/templerun_rl/sim_env.py` (`TempleRunSim-v0`) — the headless sim.
C) `scripts/sim_fidelity.py` — replay traces through the sim, diff, report.

The canonical raw-state field names are in `src/templerun_rl/constants.py` (the 22 `STATE_FIELDS`).
Use those EXACT names everywhere. The sim reuses `obs.py:state_to_obs` and the env reward (score-delta) verbatim.

## 1. TRACE JSONL schema (one JSON object per line)
A trace file = a sequence of episodes. Lines:
```
{"t":"ep_start", "seq":<int>}
{"t":"step",
 "seq":<int>,                      // monotonic frame counter
 "action":<int 0..6 | null>,       // action the host DISPATCHED that produced this frame (null = reset/first)
 "state":{                         // the 22 canonical raw fields (constants.STATE_FIELDS), real values
    "is_dead":bool, ..., "score":int, "velocity":float, "facing":int,
    "path_type":int, "path_level":int, "must_jump":bool, "must_slide":bool,
    "can_turn_left":bool, "can_turn_right":bool, "is_turn":bool, ... },
 "hidden":{                        // extra real state needed to REPLAY deterministically
    "distance":float,              // GameController.DistanceTraveled
    "lane":<int -1|0|1>,           // strafe lane (L/C/R) if derivable, else null
    "ground_height":float,
    "slide_t":float,"strafe_t":float,"stumble_t":float,  // active-timer remainders if readable, else null
    "death_type":<int|null> },
 "track":[                         // the next K path elements AHEAD of the player (K>=8), in travel order
    {"type":int,"level":int,"obstacles":int,
     "is_turn":bool,"can_turn_left":bool,"can_turn_right":bool}, ... ]}
{"t":"ep_end", "reason":"dead"|"reset"|"timeout"}
```
The logger fills `hidden`/`track` best-effort; fields it cannot read → `null` (harness tolerates nulls).
`track` is what makes deterministic replay possible (the sim is FED this instead of generating).

## 2. SIM interface (`src/templerun_rl/sim_env.py`)
Registered `gym.make("TempleRunSim-v0", ...)`. Mirrors the real `TempleRunEnv` so policies/obs are drop-in.
```
TempleRunSimEnv(
    track_source="procedural" | <list[dict] injected track> ,   # injected = replay the recorded `track`
    config=<dict of numeric knobs, defaults from SIM_MODEL_SPEC §KNOWN>,
    seed=int, step_dt=0.1, reward_scale=0.01, death_penalty=1.0,
    # same shaping knobs as the real env so training transfers:
    action_cost=0.0, reaction_bonus=0.0, turn_spam_penalty=0.0, idle_action_penalty=0.0,
)
reset(*, seed=None, options=None) -> (obs, info)     # info["raw_state"] = full 22-field dict
step(action:int 0..6) -> (obs, reward, terminated, truncated, info)  # info["raw_state"] = 22-field dict
raw_state() -> dict                                   # current 22-field dict (for the diff harness)
set_state(raw_state:dict, hidden:dict)                # force sim to a given real state (one-step tests)
```
- obs space/action space IDENTICAL to `TempleRunEnv` (Box(21), Discrete(7)); reuse `obs.state_to_obs`.
- reward IDENTICAL: `(score-prev_score)*reward_scale + shaping; -death_penalty on terminate`.
- **Replay mode** (`track_source` = injected list): the sim consumes the provided elements as the player
  advances instead of RNG-generating — so a recorded action sequence reproduces the recorded trajectory IF the
  physics/collision/scoring are correct. This is what the fidelity harness exercises.
- **Procedural mode**: RNG-generate the track per `config` (for training). Must match the real generator's
  distributions (validated by the statistical test).

## 3. FIDELITY harness (`scripts/sim_fidelity.py`)
`python scripts/sim_fidelity.py <trace.jsonl>` →
- For each episode: build `TempleRunSim-v0(track_source=ep.track)`, `set_state(ep.first.state, ep.first.hidden)`,
  then `step()` with the episode's recorded action sequence; collect sim `raw_state()` each step.
- Diff sim vs real per field, in TIERS:
  - **EXACT** (must match every step): `must_jump, must_slide, is_turn, can_turn_left, can_turn_right, path_type, path_level, facing`.
  - **TOLERANCE**: `velocity` (±2 or ±2%), `score` (per-step delta sign+~magnitude).
  - **±1 STEP**: `is_dead`/`is_game_over` (death fires within 1 step of the real death).
- Output a REPORT (human-readable, the user reads THIS not the code):
  per-field `%match` + first N divergence locations (episode, step, real vs sim) + an overall PASS/FAIL per tier.
- Exit non-zero if any EXACT-tier field < 100% (so it works as a unit/CI test).
Also stub `scripts/sim_gen_fidelity.py` (statistical: compare obstacle-spacing / turn-rate / piece-type
distributions between many real episodes and many procedural sim episodes via histograms + KS-test).
