# Temple Run RL — project memory (minimal)

Goal: an RL policy that reaches **10,000 m** on real **Temple Run** (`com.imangi.templerun` v1.39.0),
trained in a headless sim, deployed on a physical Android phone via **Frida** (read game state + inject
actions over TCP — no screen-scraping, no synthetic input). Full history: `docs/CLAUDE_FULL_HISTORY.md`.

## ⚡ CURRENT STATE (2026-07-07) — read this first
- **DEVICE CHAMPION = v47** (`runs/ppo_v47_land_1703936_steps.zip`, 256-net obs58): fresh-phone median
  **3540 / best 6783** vs v36 3112/4307. First 256-net to beat v36 — the hole landing-model fix
  (`wide_gap_landing_model`) TRANSFERRED, breaking the offline↔device inversion curse. `runs/ppo_best.zip`
  (v36) is the safe fallback. Branch `sim-fidelity-holes-crevasse-turn` (committed 2309c7b, pushed).
- **Fidelity fixes SHIPPED (all config-gated, equivalence-green, in both sims):** `wide_gap_landing_model`
  (holes, device-verified via v47), `crevasse_apex_all` (device-verified via v48 19%→15%), `turn_send_lat_frames`
  (turn dead-letter, parity 16/16, NOT device-proven in isolation). Regressions: v48/v49/v50 all over-
  specialised (stacked too many changes / dense-profile grind). LESSON: change ONE thing per retrain.
- **HOOK 16y (committed): precise hit + arc instrumentation** — the velocity-drop hit heuristic was
  OVER-counting (false positives). Real device mechanic decoded via source-dig: **high-ground (level-3)
  crevasse caps stumble (×0.85) 32%, rule = `world_y < ground_height` at cap entry; mid caps 0%.**
- **IMMEDIATE NEXT ACTION:** (1) recover/cool the phone, re-capture 16y (v47, ~8ep) to lock the 32% rate +
  arc calibration; (2) model the high-ground cap graze in the sim (arc height at cap < cap top → Stumble,
  level-aware, config-gated); (3) retrain ISOLATED (v47 + this one fix, no dense profiles) → device eval.
  See the ★ 2026-07-07 SOURCE-DIG section below for the full mechanic + the exact device numbers.
- **TWO fully-specified fixes ready (both device-grounded this session):**
  - **(A) HIGH-GROUND CAP GRAZE (crevasse fidelity, safe/config-gated).** Device physics NAILED via hook
    16y: **jump peak = 19.2 world-units (velocity-INDEPENDENT constant, p25 19.1/p75 19.2, n=186)**; cap
    ground_height by level: **level-3 = 15, level-2 = 0** (from the capture). Model: arc_y = jump_peak ·
    parabola(frac_at_cap); graze (Stumble ×0.85) iff `arc_y < cap_height`, Ledge-kill iff `arc_y <
    cap_height − band`. Reproduces device 32% graze @lvl3 / 0% @lvl2 because a lvl3 cap (15) sits just
    under the 19.2 peak → the descending arc drops below it ~32%; lvl2 caps (0) always clear. Replace the
    parabola-vs-mh(0.95) proxy in the stepup/`crevasse_apex_all` path with this world-Y check (the sim's
    `_ground_height` is a comment=15, never actually used — wire the cap height by level). Config-gated,
    default off → equivalence stays green. Captures: `captures/v47_16y_arcs.jsonl`, `v47_16y_calib2.jsonl`.
  - **(B) CHAINED-OBSTACLE BLINDNESS (death mode — USER-observed on the 6216m run, THE priority).** Obs has
    1-obstacle lookahead (next_must_jump/slide) + next2_path_LEVEL, but NOT the 2-ahead OBSTACLE. Death
    (v47_16y_calib2 ep1 @6216m): JUMP over hole pt11 → still airborne → FLAME tower pt32 right behind →
    can't slide-under while airborne → Burnt. FIX = add `next2_must_jump` + `next2_must_slide` to the obs
    (OBS_BOOL_FIELDS, after next_ground_change_higher → OBS_LEN 30→32, obs 58→60). BREAKING (obs-dim
    change): both sims' _snapshot add them (nxt2 = _track[cell_index+2], nobst2 ∈ _MUST_JUMP/_MUST_SLIDE,
    with the SAME junction-dropout as next2_path_level: False if cell.is_turn or nxt.is_turn); the torch
    obs builder must add them at the SAME positions (equivalence catches drift); surgery v47 obs58→obs60
    (insert 2 zero cols at index 19); hook + capture_calib add the 2-ahead reads for deploy; retrain from
    the surgery-widened v47. Do this ISOLATED (no dense profiles) as its own line — it's a fork, so the
    obs58 champions stay deployable on the old obs.

## 🎯 GOAL REACHED — 2026-07-06: device run **10,545 m** (v36, hook 16w; `captures/best_run_10545m_v36_hook16u.jsonl`)
Died Burnt on a flame tower at the end after 180 turns / 34 crevasses / 56 narrows / 54 holes / 92 jumps,
117 s at the v210 cap. **NEXT PHASE = RELIABILITY: a 10 k MEDIAN, not just best** (same session runs were
2320/1872/2368/**10545**/589 — high variance). Edge cases still to model in the SIM: (1) FLAME-TOWER
slide-under at speed (the 10545 death), (2) multi-hole jump RHYTHM (4-cell must-jump chains — jump-spam
fails, needs staggered land-then-rejump), (3) narrows-at-speed strict [3,7] band (the 256-net v39 breaks
the narrow↔crevasse interference but drifts — needs a stable from-scratch train, see `scratchpad/
train_gpu_256.sh`), (4) stumble→rejump recovery.

## Setup
- **Engine** Unity 2022.3 IL2CPP, arm64 only → needs a REAL arm64 phone (no emulator: Frida can't hook
  the ARM lib under x86 translation). Device **3f80a3a** (Redmi, Android 15, NOT rooted).
- **Frida Gadget** (not frida-server — no root): `libgadget.so` injected via LIEF `DT_NEEDED` on
  `libmain.so`; listens 127.0.0.1:**27042**. Build: `tools/build_gadget_apk.py` → `TempleRun-gadget.apk`.
- **Reverse eng.**: Il2CppDumper → `il2cpp_work/output/dump.cs` (signatures + field offsets + RVAs;
  bodies are native, not readable). All offsets/RVAs below are from it (re-dump if the APK changes).

## Run it
`python scripts/setup_device.py` forwards 27042 + state **9999** + action **9998**, then injects.
Inject one-liner (run EXACTLY ONE; the host process must STAY alive — if it exits the in-phone agent
unloads. ⚠️ Before re-injecting: `pkill` is forbidden — `kill <PID>` the OLD injector first (`ps -eo
pid,cmd | grep gadget_inject`); stale injectors RE-INJECT after a force-stop+restart → multiple agents,
port contention, corrupted evals [2026-07-06: 3 stale injectors stalled a capture at ep1]):
`setsid nohup .venv/bin/python frida/gadget_inject.py frida/temple_run_hook.js 127.0.0.1:27042 &`
Recover a flaky/crash-looped phone: `adb -s 3f80a3a shell am force-stop com.imangi.templerun` →
`am start -n com.imangi.templerun/com.imangi.unityactivity.ImangiUnityActivity` → wait 12 s → re-forward
→ re-inject. Verify `_hv` on :9999. Keep-awake (§HyperOS): `appops set com.android.shell
android:write_settings allow` + `settings put system screen_off_timeout 2147483647`.

## Protocol
- **State (9999)**: newline JSON @ **30 Hz** (`POLL_MS=33`, hook 16t+ — OBS freshness only; the host env
  drains to the NEWEST line and CONTROLS at 10 Hz. The old "20 Hz was WORSE" verdict was a 20 Hz-CONTROL
  retrain, unrelated). Game Updates @60Hz.
- **Actions (9998)**: 1 byte over ONE persistent socket. `0`=StrafeLeft `1`=StrafeRight `2`=Jump
  `3`=Slide `4`=TurnLeft `5`=TurnRight `6`=NoOp; reset-only `8`=RunAgainClicked `9`=OnPlayClicked
  (`reset()` sends 8 then 9). **Lateral = analog tilt** (hook replaces `GetAccelerometerForceX`):
  byte0/1 = hold ±, byte6 = level→game recenters.
- **Turn = the game's own swipe-queue (2026-07-04, hook 16o+):** the turn byte writes `GI_ShouldTurnLeft/
  Right`+`TimeSinceShouldTurn=0` on GameInput (HandleDelayedTurning) → the GAME fires the turn AT THE
  CORNER with native timing, exactly like a real early swipe. This SIDESTEPS the ~2-frame control latency
  that made plain immediate `fn.TurnLeft` overshoot (see STATUS). NN still decides WHEN to turn; the hook
  only queues it (like the analog-tilt translation). `TURN_NATIVE_BUFFER=true`; anti-double-turn debounce +
  valid-dir guard. (The every-frame flag-HOLD variant looked "exploity" — removed; single write only.)
- **Small narrows (pt17/18) → `must_jump`** (hook 16p): human data = jumps them 82%. The policy's existing
  must_jump→jump generalizes → v14 jumps them, no retrain. Set on both `cachedMustJump` and streamed `s.must_jump`.
- **Hook** `frida/temple_run_hook.js`, version in `HOOK_VERSION` (`_hv` on the stream). Current
  **16u** (`seedprobe-16u`) = 16r (next/next2_path_level, device-verified 10/10 direction) + **16s**
  (one-shot `s.turn_write` verdict: written/debounce/invalid — junction misses self-explain) + **16t**
  (30 Hz broadcast) + 16u seed probe (OFF). Streams raw state + derived obs; executes action bytes. NO
  action-DECISION rules (NN decides); swipe-queue + must_jump-flag = perception/execution translations.
- **TRACK SEEDS (probe 16u, CONCLUDED 2026-07-06): device runs are NOT seedable.** Generation
  (`PathElement.AddRandomPath → GetRandomPathElementForSide`) draws from Unity's GLOBAL RNG;
  `Random.InitState(seed)` (RVA 0x24B3E6C) re-seeded at the first path draw of a run gives only a
  ~6-cell identical prefix — mid-run generation interleaves with other consumers (coins/effects) and
  desyncs. Full determinism would need a private RNG stream (not worth it). **Reproducibility tools:
  SIM `reset(seed=N)` (exact); device runs are fully RECORDED in `captures/*.jsonl` (the 6399 m record
  = `captures/v14_16t.jsonl` ep5) — replayable through replay_diff/analysis, not live.** `SEED_RUNS=-1`
  keeps the capability OFF. Side find: `PathElement` statics `sAllowTurns/sAllowPathHoles/
  sAllowPathNarrows/…` (0x4-0xC) = writable feature toggles (future isolation diagnostics).

## Key offsets (this build; from dump.cs)
- **CharacterPlayer**: IsDead 0xD8, IsJumping 0xF8, IsSliding 0x100, Score 0x138, Velocity(Vec3) 0x34,
  Facing 0x78, JumpVelocity 0x80, GroundHeight 0x88 (=raised ledge height), IsGroundHeightChangeHigher.
- **PathElement**: Type 0x30, Level 0x34, Obstacles 0x38, neighbors 0x50/58/60/68, Rectangle 0xA0.
- **GameController.GameInput @0x78** (HandleDelayedTurning buffer — the turn swipe-queue): GI_JustTurned
  0x20, GI_ShouldTurnLeft **0x21**, GI_ShouldTurnRight **0x22**, GI_TimeSinceShouldTurn 0x24,
  GI_TimeSinceLastTurn 0x28, GI_ShouldJump 0x2C, GI_TimeSinceShouldJump 0x30. (Hook writes ShouldTurnL/R.)
- **Enums**: ObstacleType {Hole1, Narrow2, TreeStumble3, TreeJump4, TreeSlide5, FlameTower6,
  LevelChange7}. PathType: 1-2 straight, 3-10 turns, 11-16 holes (12/13/15/16=HoleLong=2-cell),
  17-30 narrows, 31-32 flame, **33-36 = SingleLevelStepDownCap (zone crevasse, both up/down)**, 37 temple.
  DeathType {Fall1, Tree2, Slide3, **Ledge4**, Burnt5, ...}. MaxRunVelocity **210** (start 100).
- RVAs (Start/Update/Turn/Jump/Slide/Strafe/CanTurn/IsTurn/GetPosition2D/RunAgain/OnPlay/…): see
  `docs/CLAUDE_FULL_HISTORY.md` §5 / `docs/OFFSETS.md`.

## Sim + training
- `src/templerun_rl/sim_env.py` = fast headless model (`TempleRunSim-v0`); obs `obs.py` (**OBS_LEN 30**,
  bools + scalars + cell_progress/lane/narrow_target_lane/**predicted_landing**/next_ground_change_higher
  + **predicted_jump** (velocity-invariant step-up takeoff cue, host-derived) + **next_path_level/
  next2_path_level** (upcoming-altitude direction, hook 16r) + K=4 action history → 58-wide; power-ups
  REMOVED 2026-07-04); reward `reward.py`; PPO `scripts/train_sim.py` (net 128,128, linger 3).
  Deploy = same obs on the phone via the hook.
- Sim turn-buffer config: `hook_turn_buffer=true` models the device swipe-queue (turn near a junction lands
  CENTERED, timing-free, `from_buffer` offset 0) + **late gate `turn_buffer_late_cp=0.56`** (swipe past it =
  silent MISS; device-measured CanTurn window end, velocity-INDEPENDENT — v14 turned 46/46 up to v207).
  Warm-start base **`runs/ppo_v14_obs30_leveldir.zip`** (EXACT surgical equivalent of v14, new obs cols
  zero-init; `scratchpad/surgery_v24_leveldir.py`).
- **JUNCTION LEVEL DROPOUT (2026-07-05 night, device-measured):** the hook's PE neighbor walk BREAKS through
  turns → on-junction frames stream `next_path_level=next2_path_level=-1` (178/202), approach frames
  `(lvl,-1)`. Sim `_snapshot` reproduces it (commit f8d7eff). Obs30 finetunes trained pre-fix saw a
  junction-frame input spike on device they never met in training. v14 immune (zero-init cols).
- **Capture-log alignment (capture_calib):** logged `action(i)` = policy's response to `state(i-1)` (predict
  runs BEFORE env.step; the record holds the POST-step state). And a FIRED turn resets cell_progress → the
  swipe record's cp is POST-turn — don't read "swipe cp" stats off fired turns (artifact; burned us twice).
- **Jump exec latency (sim, 2026-07-05):** `jump_exec_lat_frames`/`jump_exec_lat_jitter` buffer the JUMP
  firing. DEVICE-MEASURED (`scripts/probe_jump_exec.py`, 25/25): **exactly 1 frame @10Hz, zero jitter**
  (uniform with the turn-side 1-frame exec). Constant latency is ABSORBED by policies (window shifts);
  only JITTER creates training pressure (v14 hiMed 2246→130 under 2±1; a predicted_jump pre-aim oracle
  survives). Train at **1±1** (measured mean + TCP obs-staleness variance); harness/replay at 0.
- **Warm-start recipe that does NOT collapse: `--learning-rate 5e-5`** (new knob; default 3e-4 ALWAYS
  collapses a v14 warm-start regardless of ent). Keep the BEST checkpoint — drift resumes past ~500k-1M.
- **THE FEEDBACK LOOPS (use before any retrain/deploy):**
  - `scripts/fidelity_scorecard.py --strict` — ~13 phone-free checks vs `captures/device_reference.json`.
    GREEN required before retrain. (Blind for un-checked mechanics — green ≠ complete.)
  - `scripts/replay_diff.py <capture>` — policy-free ONE-STEP sim-vs-device transition diff on a real
    capture. Great for turns/obstacles; CANNOT test multi-frame mechanics (the crevasse jump arc).
  - Device eval: `scripts/capture_calib.py <model> --episodes 8 --turn-cooldown 0`.
- Checkpoints are noisy + late-collapse → **keep the BEST checkpoint, not `_final`**. Registry `runs/MODELS.md`.

## GPU sim + training (2026-07-06, GPU box — torch port of the sim)
- **`src/templerun_rl/sim_env_torch.py` = batched-tensor port of sim_env.py** (`TorchTempleRunVecEnv`,
  B envs as [B] tensors, cuda/cpu). The CPU sim stays the REFERENCE — if they disagree, fix the port.
  **Equivalence gates (run after ANY change to either sim): `scripts/sim_equivalence.py`** (1344
  deterministic step-exact scenario×config×policy diffs + statistical tier), `scripts/scorecard_torch.py
  --strict` (the unmodified scorecard via a registry alias; 12/13 PASS — turn.gates oracle-survival is
  threshold-MARGINAL on BOTH sims: true rate 73%cpu/72%torch vs the 70% bar, n=80 seed-batch coin flip),
  + v36 plays both sims to median ratio 1.002 (n=160). Sim steps ~40k/s at B=4096 (65x the SubprocVecEnv).
- **Training**: `scripts/train_sim_gpu.py` (SB3 over the batched env; recipe-faithful geometry =
  B=256 n_steps=16 batch=64 → ~900 steps/s end-to-end — the PPO optimizer loop is the wall, NOT the env;
  batch=512 gives 5k steps/s but BROKE a from-scratch run — dynamics change, use only with curve
  validation). `scripts/train_fast.py` = pure-torch GPU-resident PPO (SB3-exact losses, saves normal SB3
  zips) — **VALIDATED 2026-07-06: 1.29x SB3 at recipe geometry + reproduces from-scratch turns-only
  learning exactly (96/96, median 6238 = SB3's). Use it for new runs.** Quick verdict tool:
  `scripts/eval_ckpt_torch.py <ckpts...>` (batched device-true-cfg eval).
- **FROM-SCRATCH WALL (measured 2026-07-06): the full task is NOT from-scratch trainable** — flat ~100m
  (the FIRST junction) for 3.4M steps (reference recipe, CPU trainer) AND 6M (torch). Cause: since the
  2026-07-02 fatal-turn remodel, turn-on-straight=death makes turn exploration a needle; every good
  policy is a warm-start descendant; train_gpu_256.sh was never actually validated. **FIX (user-designed,
  verified): curriculum.** (a) turns-only phase A (obstacle_prob 0, turn dense) → from-scratch 256 masters
  turns in <100k steps (96/96 to cap), then full-task phase B warm-start (climbing, 1173@1.3M); (b) BETTER:
  **per-episode `episode_profiles` mix** (torch sim only: cfg list of {weight, overrides} sampled each
  reset — e.g. 50% standard / 15% turns-only / 12% narrows-from-0 / 12% crevasse-dense / 6% hole-rhythm /
  5% compound) = guaranteed rare-mechanic exposure with zero phase-atrophy. See runs/ppo_mix256 lineage.
- **256-net path that works: v41 = v36 surgery-widened (scratchpad/surgery_v39_256.py, exact-equivalent
  init) + 2M finetune @5e-5 on the v39 recipe → OFFLINE median 2200 vs v36's 1784 (+23%), max 8389, still
  climbing at 2M, NO drift.** v42 = +2M extension. Stop-early workflow: eval checkpoints at 1-2M
  (eval_ckpt_torch) and kill/extend on evidence — training is ~30 min, don't pre-commit to 6M.
- **v41 DEVICE-REGRESSED (2026-07-06 evening, same-session fresh-phone): median 1174 / best 5956 vs v36
  2382 / 4357 — offline +23% did NOT transfer (the v15-v22 lesson again). Death signature: 6/8 on
  low-speed holes/gaps 580-1500m** (v36 same-session: diverse mix, 1-2 holes). Captures:
  `captures/v41_device_16w.jsonl`, `captures/v36_samesession.jsonl`. **ROOT-CAUSE CANDIDATE FOUND
  (USER-observed, device-measured): the NON-FATAL CLIP** — a marginal step-up graze STUMBLES (velocity
  ×0.85 in one frame = StumbleSlowDown, recovery ~+1.4 u/s, 15/15 airborne, 14/15 on pt33-36/long-holes,
  ~0.6/ep) instead of killing; the old binary clear-or-die sim rewarded gap timings that clip-and-slow on
  device, re-timing the following holes → the death cluster. **Modeled: `stepup_clip_band`** (apex height
  in [mh-band, mh) → _stumble; both sims, default 0 = unchanged, equivalence 1638/1638, band 0.25
  calibrated ≈ device rate 0.46 vs 0.6/ep; residual: long-hole lip grazes un-modeled). v43 = clean
  v36_256 + clip-band retrain. **v36 (`runs/ppo_best.zip`) remains the deploy champion.**
- Screen-off play is IMPOSSIBLE (display sleep → onApplicationPause → Update stops, measured); the
  equivalent for long sessions = keep-awake + `settings put system screen_brightness 0`.
- **v43 = v36_256 + clip-band recipe (2M @5e-5): device SAME-SESSION median 1820 / best 3957 vs v36
  1682 / 3420 — parity-leaning-v43, the first 256-net that does NOT device-regress.** n=8 too noisy to
  crown; needs a bigger-n alternating eval. Long holes pt12-16 at 600-1600m = the universal killer now
  (kills v36 too) → next mechanic hunt in captures/{v41,v43}_device_16w.jsonl + v36_session3.jsonl.
  Offline w/ clip-band eval: v43 3889 vs v36 1989 (eval_ckpt_torch, clip band now in device-true cfg).
  **HOLE-DEATH FORENSICS (scan_hole_deaths, 11 deaths / 3 captures): 10/11 long-hole deaths follow a
  CLIP-SLOWDOWN within 3s** — clip → −15% vel → next gap mistimed at the reduced speed → Fall. Levers:
  (1) train post-clip gap timing (clip band already in sim — check the sim reproduces the clip→next-gap
  death chain rate), (2) next2-obstacle lookahead obs (chained-jump blindness: 2 v43 takeoffs weren't at
  a gap; jump spans ~2.75 cells vs 1-cell obs lookahead → blind landings; hook 2-ahead read verified on
  straights; obs 58→60 = retrain), (3) jump-spam persists (8-18 cmds/3s — avoid-state not retry).
- **FLEET RESULTS (2026-07-06 late): OFFLINE gains do NOT transfer while the LONG-HOLE model is
  lenient.** v44 (256+2M) offline 6270 / v45 (512-net) offline 7040 vs v43 3889 — but v45 DEVICE
  median only 1313 (best 5139; 3/8 long-hole deaths; same session v43=1820). USER-OBSERVED live +
  forensics-confirmed: policies jump EARLY at long holes (takeoff cp 0.05-0.4, correct is >=0.5) —
  free in-sim because wide-gap survival = binary airborne-at-entry checks, NO landing-position arc
  model (the step-up apex check has one; plain holes don't). **NEXT SIM FIX (highest value): device
  takeoff-cp sweep via `capture_calib --gap-override '0.2,0.4,0.6,0.8'` on long holes → fit a
  landing-position kill rule → equivalence-gate → retrain (v46).** 512-net verdict: more offline skill
  at equal budget (real capacity signal) but pointless until the hole gap closes. **SWEEP VERDICT
  (holesweep_takeoff.jsonl, 27 controlled jumps): FLAT LONG HOLES ARE FORGIVING (7/7 cleared, takeoff
  cp 0.08-0.99, all speeds — RE-confirms the earlier 10/10 probe). The pt12-16 deaths are NOT takeoff
  timing: killer = CHAINS (multi-hole rhythm) + post-clip reduced velocity + refusal-eaten jumps.
  → v46 levers: hole-rhythm episode profile at post-clip velocities (stumble_vel starts), next2-obstacle
  obs, NOT a landing rule.** Crevasse sweep falls at cp .68-.9 @v150-180 (n=5, apex-window edge, small n). **mixg (from-scratch
  profile-mix, 6M in 17 min): offline median 1712** — from-scratch pipeline WORKS (was impossible);
  hard wall ~1750 (93/96 Falls; profile weights under-serve late-track narrows — tune weights later).
- **VERDICT (2026-07-06 night, decisive): STOP RETRAINING — fix SIM FIDELITY. All 256/512 candidates
  device-REGRESS; v36 champion, v43 best challenger.** Same-session device medians: v45(512-net) 1313,
  v46(hole-rhythm) 1346 vs v36 2504-2805. Offline↔device correlation is INVERTED (v45 offline +250% →
  device −50%): bigger net + more training just EXPLOIT the sim's optimism harder. **CROSS-CHECK (both
  sims score the SAME policies): torch sim is FAITHFUL to the CPU reference (v46 ratio 0.96; v43 0.74 =
  yellow-flag, recheck bigger-n), and BOTH sims sit ~2-3x ABOVE device — the gap is a real sim-to-real
  gap in the REFERENCE sim, not a torch-port defect.** Only v43 held (trained right after the clip gap
  was closed) → closing a real gap converts optimization from poison to fuel; that is the ONLY lever now.
- **★ CURSE BROKEN — WIDE-GAP LANDING MODEL (2026-07-06 late night, THE session that made offline↔device
  correlate again). Forensics (`scratchpad/jump_arc.py`, 1600 jumps / 8 captures): 2-cell HoleLongs
  (pt12/13/15/16) clip 20% on device, concentrated at LOW speed. ROOT CAUSE: the sim only checked
  "airborne at cell entry" — a SHORT arc satisfies that while LANDING inside the far gap cell (device
  clips the far wall → dt4 Ledge). Device jump span = `0.0204*v − 0.94` cells (NEGATIVE intercept; the
  sim's `jump_dur*v/cell_len = v/75` has the wrong slope). FIX = `wide_gap_landing_model` (config knob,
  default OFF): on the LAST gap cell, require `landing (takeoff_pos+span) ≥ far_edge − grace`. HOLES ONLY
  (obst==OBST_HOLE) — must NOT bleed onto LevelChange crevasses. `wide_gap_land_grace=0.35` (calibrated:
  ~20% clip, leaves the good jumper v36 ~untouched — grace-insensitive plateau — while punishing
  jump-spam). Equivalence GREEN 1820/1820 with it ON; scorecard GREEN feature-off. Both sims edited.**
- **v47 = v36_256 + landing model, 2M @5e-5 (train_fast bs512 cuda-graph): DEVICE median 3540 / best 6783
  vs v36 3112/4307 SAME-SESSION (fresh→warm phone, 8ep, hook 16x). FIRST 256-net to BEAT v36 on device —
  the hole fix TRANSFERRED (offline 4793 → device win, correlation RESTORED). `captures/v47_land_16x.jsonl`.**
- **CREVASSE regression (USER-spotted live: "jumps too early on mid-high + high-ground crevasses, spot it
  by the SLOWDOWN"): the landing model was bleeding onto LevelChange crevasses (they're in _MUST_JUMP +
  wide_gap) → rewarded early crevasse jumps. FIXED two ways: (1) landing model HOLES-ONLY, (2) NEW
  `crevasse_apex_all` (config knob, default OFF): the velocity-apex check (was step-UP only) now fires on
  ALL crevasse ledge cells incl step-DOWN/high-ground. Device vel_y ARCS (hook 16x) confirm step-down
  also needs near-apex at the far cell (pt35 cp0.96 vel_y~4=apex). Offline-validated: apex ON drops the
  early-jumper v47 (4131→3440) but leaves good v36 untouched (1810→1825). v48 = v47 + both crevasse fixes:
  device crevasse HIT-RATE (slowdown metric, `scratchpad/crevasse_clips.py`) 19%→15% (v36=16%) — the fix
  WORKS. BUT v48 overall median 2587 (degradation-confounded — ran ~2.5h into session) and offline≈v47:
  the stumble-penalty GRADIENT IS WEAK (crevasses too sparse in standard training). NEXT (v49): a
  CREVASSE-DENSE episode_profile so the policy gets enough exposure to learn the fine timing + re-eval
  v48/v49 vs v47 on a FRESH (cooled) phone. `captures/v48_land_16x.jsonl`.**
- **HOOK 16x (`clean-16x`): + Velocity Vec3 `vel_x/y/z` (pure read, MovingObject.Velocity @0x34/38/3C) =
  arc ground-truth. `vel_y` = vertical jump arc (takeoff ~75 → apex ~0 → negative). capture_calib now
  logs vel_x/y/z (added to LOG_FIELDS). Device serial 3f80a3a; recover flaky phone = force-stop+restart+
  re-inject (the injector host MUST stay alive — if it dies the agent unloads → capture reads "degraded").**
- **METHOD that now works (the loop the user asked for): device capture → forensic mine the death
  signature (death_type + velocity-slowdown = a survivable hit) → measure the real mechanic (span/arc/cp)
  → add a config-gated sim knob calibrated to it → equivalence-gate (default off) → retrain (holes/apex
  in device-true cfg) → SAME-SESSION phone eval. Each closed gap lifts the device median; offline now
  predicts device again. Remaining death modes to close for a 10k MEDIAN: narrows-at-speed, multi-hole
  rhythm, flame-tower slide-under, junction reliability @v200+.**
- **★ TURN DEAD-LETTER (2026-07-06, USER-observed "the policy fires an action but it is never launched"):
  v47 junction/off deaths (pt-1) = the policy DECIDES a turn on the junction (v47 ep3: TL @observed
  cp0.36 @v210) but the byte arrives ~1 frame later, and at high speed 1 frame = ~0.47cp, so it PROCESSES
  at cp~0.83 > the 0.56 consume deadline → DEAD-LETTER → runs off the track (fires R/R/L into the void,
  dies). v14 survived by deciding @cp~0.1 (early). The gap: turn_buffer_late_cp (0.56) is measured on the
  PROCESSING cp (velocity-independent), but the sim gated on the DECISION cp with NO latency → let v47
  fire on-junction turns that dead-letter on device. FIX = `turn_send_lat_frames` (config knob, default 0):
  proc_cp = decision_cp + velocity*lat*dt/cell; gate on proc_cp. lat 0.4 = v47 median 4189→2252 (pressure,
  still trainable; lat 1.0 → 256, device-plausible but crushes the untrained base). Parity 16/16 on the
  junction scenarios. The latency-robust play the policy must LEARN = PRE-SWIPE on the approach cell
  (turn_approach_buffer, from_buffer → fires centered, IMMUNE). v50 = v47 warm-start + all 3 fidelity fixes
  (holes+crevasse+turn_send_lat 0.4) + junction/crevasse/narrow-dense profile mix (`ppo_v50_turnlat`).
  v50 DEVICE (8ep fresh phone): median 2044 / best 5314 — REGRESSED vs v47 3540/6783. Still 1 junction/off
  death. LESSON: stacking 3 fidelity changes + a 4-way dense-profile mix OVER-SPECIALISED (same as v49) —
  the dense grind hurt the standard distribution more than the turn fix helped. NEXT: isolate the turn fix
  = v47 + turn_send_lat ONLY (mild, no dense profiles) to test it cleanly. v47 REMAINS the device champion.**
- **GAP FORENSIC (death_mechanic.py, v47 device vs sim): the raw medians are CLOSE now (sim ~3344-4200 vs
  device 3540 — the hole+crevasse fixes worked) but it's a COINCIDENTAL cancellation: sim kills v47 on
  NARROWS 48% (~4x device 0.125/run — sim narrows likely too HARSH at v200+) but has ~0 FLAME deaths
  (device kills on flame — the slide-under-at-speed timing is unmodelled). Next fidelity targets: measure
  device narrows-at-speed (is [3,7] too strict at speed?) + flame slide-under. NEW config knobs this
  session (all default-OFF, equivalence-gated): wide_gap_landing_model/_span_a/_span_b/_land_grace,
  crevasse_apex_all, turn_send_lat_frames. Hook 16x = +vel_x/y/z. UNCOMMITTED (per norm).**
- **`scripts/trajectory_diff.py` (NEW, multi-frame comparator — the cheap sim-vs-game tool replay_diff
  couldn't be): reconstructs the track from a capture + replays recorded actions through the sim.
  VELOCITY half WORKS (sim-vs-device vel MAE 0.5-1.4 = physics/curve/clip are FAITHFUL). Death-locator
  half is BUGGY (reports constant "sim dies @~90" every ep = track-reconstruction alignment artifact,
  the action(i)=resp-to-state(i-1) + fired-turn-resets-cp gotchas). DO NOT trust its death verdict until
  the frame alignment is fixed. NEXT: hook player world-pos/height per-frame (pure read) → exact arc
  ground-truth → fix the comparator → ground arc/chain mechanics in real physics vs statistical fits.
- **FLEET IN FLIGHT AT SESSION END (2026-07-06 evening): v44 (v43+2M ext), v45 (v43 surgery-widened
  256→512, `runs/ppo_v43_512.zip` init), mixg (from-scratch profile mix + clip band, 6M) — all
  train_fast bs512 --cuda-graph. GATE each with eval_ckpt_torch (n>=96) then SAME-SESSION phone eval
  vs v36 AND v43 (`captures/v43_device_16w.jsonl` = its 8-ep baseline, median 1820). A subagent (git
  worktree) is making the torch-sim step branch-free + CUDA-graph-captured (equivalence-suite-gated) —
  check/merge its worktree branch. Iteration loop: model gap → 3-min train → 1-min offline gate →
  8-ep phone eval → capture forensics → repeat. Device serial on the GPU box = 3f80a3a (Redmi Note 7,
  screen brightness 0 + keep-awake set; game restart drops the hook → kill old injector PID, re-run
  scripts/setup_device.py).
- **SPEED (2026-07-06 final): minibatch A/B PROVED bs512 quality == bs64** (same recipe/warm-start,
  1M-step arms: medians 3675/3830/3498 within noise — the earlier "bs512 broke from-scratch" was the
  from-scratch WALL confound, not bs). **+ CUDA-graph minibatch capture in train_fast.py (--cuda-graph;
  needs Distribution.set_default_validate_args(False)) → 10.3k steps/s end-to-end = 2M finetune ~3 min**
  (was 610/s = 55 min). Next lever: branch-free sim step → sim graph capture (equivalence-suite-gated).
- **SIM CUDA-GRAPH (2026-07-06 night, subagent, VERIFIED-INDEPENDENTLY, NOT merged): branch-free
  torch-sim step + opt-in `enable_step_graph()` / `train_fast --sim-graph`. Bit-identical — my OWN check:
  eager-vs-graph reward+obs streams max|diff|=0.00e+00 @B512×200, + equivalence 1638/1638 GREEN re-run on
  the branch. Per-seed RNG preserved via CUDAGraph.register_generator_state (torch 2.6). Speed only ~1.17x
  @B1024/4096 BUT that's a FLOOR: random-action bench dies ~every step → eager reset fallback
  (uncapturable) runs each step; under a TRAINED policy (long episodes) the win is larger — re-bench
  before judging. Branch `worktree-agent-ad640dbebfa14fa05` commit `0083529` (8 files). MERGE CARE: main
  working dir has UNCOMMITTED same-file edits (clip band, min_obst, new scripts, this CLAUDE.md) —
  reconcile first; the subagent replicated min_obst so its sim_env_torch is ~a superset.
- **UNCOMMITTED STATE (main working dir, 2026-07-06 night): NOT committed** (per norm: commit only when
  asked). New tracked files: sim_env_torch.py, sim_equivalence.py, scorecard_torch.py, train_sim_gpu.py,
  train_fast.py, eval_ckpt_torch.py, trajectory_diff.py; modified: sim_env.py (stepup_clip_band),
  obs/constants unchanged. Weights runs/*.zip gitignored (v41-v46, v36_256, v43_512 all local-only).

## ★ 2026-07-07 — SOURCE-DIG + PRECISE INSTRUMENTATION (hook 16y): the crevasse graze DECODED
- **Method shift (user-directed): mine the game SOURCE (`il2cpp_work/output/dump.cs`) with parallel agents +
  add the MISSING state variables to the hook, instead of the slow train→fail→guess loop.** Bodies are
  native, but fields/signatures/enums/cross-refs are readable. 3 agents mapped: (1) the STUMBLE mechanic,
  (2) crevasse/ledge collision, (3) missing hook vars.
- **THE GAME HAS NO SEPARATE GRAZE — every non-fatal hit IS a Stumble** (`CharacterPlayer.Stumble`,
  velocity ×= `StumbleSlowDownPercent` **0.85** for `StumbleDuration`, escalates to Kill via
  `StumbleKillTimer`/`StumbleKillVelocityPercent` — device-read **stumble_kill_vel_pct = 0.65**). Matches
  the sim's ×0.85. The crevasse "graze" reuses Stumble. `IsGroundHeightChangeHigher` = the step-UP flag.
- **HOOK 16y adds the variables we lacked (the user's point — recordings missed key state):**
  `time_since_stumble` (==0 at the exact clip frame — PRECISE hit onset), `stumble_duration`,
  `stumble_kill_vel_pct`, `is_falling`+`time_since_fall` (came-up-short BEFORE a Fall), `death_run_velocity`,
  and **`world_y`** = `CommonPlayer.PreviousLocation.y` @0x80 (last-frame world-Y — `world_y - ground_height`
  = jump height above track = EXACT arc, cheap field read). capture_calib logs them all now.
  (Doc-bug found: CLAUDE.md's old "JumpVelocity 0x80/GroundHeight 0x88 (CharacterPlayer)" are the ENEMY
  class; the player has NO JumpVelocity, real GroundHeight=0x134, and 0x80=PreviousLocation.y.)
- **★ THE VELOCITY-DROP HIT HEURISTIC WAS WRONG (over-counting).** Precise `is_stumbling` on v47 (16y,
  42 crevasses / 6 clean eps): the "57% mid→high step-up graze" from the −5% velocity heuristic was
  FALSE POSITIVES — step-up caps grazed 0/7 with 0 velocity drop. **The REAL grazes are on HIGH-GROUND
  (path_level 3, raised cap ground_height 15) crevasse caps: 32% (6/19); mid (lvl2) caps 0% (0/23).**
  Rule (device-exact): **graze fires iff `world_y < ground_height` at the cap entry** (arc DESCENDED below
  the raised cap top) — grazes arc_h@entry median −0.6, cleans +3.1 (lvl3) / +4.2 (lvl2). The ×0.85
  slowdown is CHRONIC (v47 grazes 32% of high-ground caps, recovers each time) — the sim keeps full speed
  → sim optimistic on high ground. This is "high-high crevasses" the user saw; NOT a step-up problem, and
  crevasse_apex_all's uniform apply was mis-aimed. **NEXT: model the high-ground cap graze as a Stumble
  when the sim's arc height at the cap < cap top, calibrated to ~32% for a v47-like policy (level 3 caps
  only); re-verify with a fresh-phone 16y capture; retrain ISOLATED (v47 + this one fix).**

## STATUS — what worked / what failed / where we are

**WORKED (device-verified):**
- **Turns FIXED — via the game's own swipe-queue (2026-07-04, the real fix).** ROOT CAUSE of the turn wall
  = **~2-frame control latency** over Frida-TCP (obs is ~1 frame stale: policy acts at device-cp ~0.50 vs
  sim-cp 0.30; + the turn executes ~1 frame late → device-cp ~0.73), which EATS the tight survivable swipe
  window. `predicted_landing` obs (v10) helped but did NOT close it; retrains kept overshooting (lane
  15-22). THE FIX: route the turn byte through the game's native HandleDelayedTurning (`GI_ShouldTurnLeft`
  write) → the GAME times the turn at the corner, immune to the lag. v14+hook-16k device: **turn-overshoots
  0/8** (was the dominant death). See Protocol.
- **Small narrows JUMPED (hook 16p)** — human jumps them 82%; hook flags pt17/18 `must_jump` → v14 jumps
  (device-verified), no retrain.
- **Narrows (long side-strips)** — cracked via a NARROW-ONLY isolation curriculum → warm-start full task.
- **Jump-into-slide death**, **turn-direction** (native CanTurn opposite-origin, hook 16f), **lateral =
  analog tilt** (alpha 0.446/0.1s), **stumble** 2-strike — all modeled + scorecard-guarded.

- **Crevasses LARGELY FIXED (v14, device-verified 2026-07-03)** — was the top device death after turns
  (v10: 5/8 @~600 m, DeathType 4 = Ledge). replay_diff proved it was the **SIM, not the reward** (reward
  rewards jumping, never timing; the sim was too lenient). Fix = model the crevasse as a **2-cell step**:
  gap on cell 1, raised ledge (`ground_height=15`) on cell 2 where a velocity-dependent APEX check fires
  (clip = Ledge if not near jump apex there). Retrained (v14) → **device median 1652, best 2908, crevasse
  deaths 2/8** (down from 5/8). `next_ground_change_higher` obs (pt 33-36, hook 16i) **verified firing on
  device** (7/7 crevasse-approach frames) — prior "0 fires" was a capture-logging artifact, not a bug.

**FAILED / LESSONS (2026-07-03/05):**
- **12 retrain regressions (v15→v22) at LR 3e-4.** Every warm-start+finetune from v14 drifted BELOW it on
  device (high-speed curriculum, off_center_penalty [device-TOXIC], turn_lane_carry "fidelity" [it's a
  load-bearing REGULARIZER, keep 3.0], tail-risk, action_delay, low-ent, buffer models). Offline metrics do
  NOT predict device turns. **AMENDED 2026-07-05: the collapse mechanism was the LR** — 3e-4 knocks the
  policy off v14's sharp optimum; **5e-5 fine-tunes stably** (v25c/v26 verified: no collapse, skills adopt).
- **v25 (2-phase: crevasse-only isolation → full) DEVICE-REGRESSED (median ~650): 6/8 deaths = MISSED
  TURNS.** Crevasse-ONLY training ATROPHIES device turns even after a full-task Phase B restores them
  in-sim. **Isolation curricula: OK for narrows, WRONG for turns — keep turns practiced in every phase.**
  Also 2/8 crevasse jumped too EARLY = the guessed lat 2±1 over-modeled (real = 1). Both fixed in v26.
- Ruled out (device-measured): over-strafe lateral gap (device caps lane ±7, same as sim); 20Hz sampling
  (WORSE); crevasse apex miscalibration (sim clears the human's v210 0.58 takeoff).

**2026-07-05 NIGHT (autonomous session) — the retrain curse + turn mechanics fully decoded:**
- **v14 FRESH-PHONE: median 2010, best 5462, 0 missed turns in 212 junctions** (`captures/
  v14_freshphone.jsonl`). The old 1652 was the DEGRADED phone. ⚠️ ALL cross-model comparisons must be
  same-session fresh-phone (v26/v27's "missed turns" were measured on the tired phone).
- **Turn mechanics (205+ junctions, frame-level):** every policy decides the swipe on the junction's FIRST
  frame; byte goes out 1 control-tick later. Early-phase onsets (cp<0.2) ALWAYS survive; late-phase onsets
  (cp≥0.3) = a latency-jitter COIN FLIP (write lands 0.5-1.5 frames after the onset sample vs the CanTurn
  window end ~0.56). Fresh phone = low jitter = coin lands good (v14 50/50 late onsets); degraded = misses
  (v26/v27: 5/9+ late-onset deaths). Wrong models tried + discarded: v29 constant-0.50 gate (never binds
  in-sim), v30 velocity-arrival lag-2 (refuted: v14 turns fine @v207). Hook drop paths (turn byte):
  debounce 0.3s / invalid-dir(CanTurn false) / not-near-junction — pre-onset turn bytes: 0 in all captures.
- **JUNCTION LEVEL DROPOUT found + fixed** (see Sim section) = why obs30 retrains looked worse on device.
  v31 (v27 recipe + dropout + 0.56 gate) trained; in-sim clean turns WITH dropout. NOTE: in-training
  EvalCallback env ≠ training env (no buffer flags) → its curve says "collapse" when device-true eval says
  healthy. Judge checkpoints with `scratchpad/eval_swipe_deaths.py` (device-true cfg, deaths by mechanic).
- **NEXT WALL @v200+ (both v14 long runs died on it): chained-jump blindness** — a jump spans ~2.75 cells,
  obs lookahead = 1 cell → lands blind into TreeJump (3738m, 5462m deaths). Lever: next2-obstacle obs
  (hook 2-ahead walk device-verified readable on straights; breaks only through turns = dropout above).
- **USER (live, 2026-07-06): low-speed dumb deaths (missed transition/turn) are the MEDIAN killer** —
  high speed works; prioritize low-speed reliability.

**2026-07-06 EARLY — fresh-phone round 2 + JUMP-BYTE RULES (the low-speed dumb deaths):**
- **v31 fresh: median 1331 best 2442 — turn hesitation CURED (1 junction death/91; dropout fix VERIFIED
  on device).** v27 fresh: 1266/3864, still 5/8 missed turns (predates the dropout fix — kills the
  "phone-state explains v27 turns" theory). v14 floor 2010/5462 stands.
- **The game EATS Jump() bytes — 3 device-measured rules (131 grounded attempts):** (1) 56% refused while
  the lane glides >0.8/frame vs 4% settled → sim `jump_refuse_strafing_prob 0.6` DEFAULT (gate
  |target−lane|>1.8, settled narrow-riders unaffected); (2) refused on wide-gap/crevasse cells past the
  lip cp~0.2 → `gap_jump_lip_cp 0.2` DEFAULT (takeoff = APPROACH cell; ALL grounded-at-gap crevasse
  deaths were act-2 spam refused there); (3) "landing lock" = airborne rule + 10Hz phase, already covered
  by jump_exec_lat 1±1. Refusals PERMANENT in context — the lesson is avoid-the-state, not retry-spam.
  These explain v31's 3/8 crevasse deaths (its learned strafe-LEFT-before-crevasse habit — crevasse-zone
  lane −2.56 vs v14 −0.46 — was free in the old sim, lethal on device) and the user-flagged low-speed
  dumb deaths. Crevasse LATERAL sensitivity refuted (n=108: 13% vs 12% death rate).

**WHERE WE ARE (2026-07-06 dawn) — FLOOR SMASHED via obs freshness (hook 16t):**
- **16s diag (turn-write verdicts) closed the junction case:** misses = DEAD-LETTER writes (flag written,
  game's consume deadline live-cp ~0.67, pipeline med 0.21cp/tail 0.40) — policy decisions were perfect
  all along. **16t = 30 Hz state broadcast (control stays 10 Hz, env drains to newest)** → sampling age
  50→17ms → dead letters gone for EVERY model, no retrain.
- **Fresh-phone @16t: v34@2M median 2863 best 4525, junctions 1/254 | v14 median 2833 best 6399
  (ALL-TIME), junctions 3/255, crevasses 44/46.** Obs freshness = the dominant lift (+40% median both);
  v34's trained rules = cleaner junctions.
- **NEW DEVICE CHAMPION (2026-07-06 late night) = v47** (`runs/ppo_v47_land_1703936_steps.zip`, 256-net,
  obs58, + hook 16x): same-session device **median 3540 / best 6783** vs v36 3112/4307 — see the CURSE
  BROKEN entry in the GPU section. v48 (`runs/ppo_v48_land_1703936_steps.zip`, + crevasse fix) improved
  the crevasse hit-rate 19%→15% but its overall median (2587) was phone-degradation-confounded; re-eval
  v48 vs v47 on a FRESH phone to crown. `runs/ppo_best.zip` (v36) stays the SAFE fallback.
- **DEPLOY (2026-07-06 EARLIER, superseded by v47) = v36@2M + hook 16u/16t** (`runs/ppo_best.zip`, obs30, 30Hz): device
  **median 2642, BEST 7040 m — NEW ALL-TIME RECORD** (beats v14's 6399), junctions 1/303, crevasses
  45/46. v36 = v34 self-warm-start + the USER-OBSERVED STACKED-obstacle blindspot (`compound_prob 0.12`:
  narrow-A → jump-barrier → narrow-B-opposite, needs a mid-air/pre-swing). Offline: compound v210 7/50
  vs v34 28/50, standard cfg 4/40 = v34 (ZERO regression). First strict improvement of the session —
  new skill, no cost. Record run archived `captures/best_run_7040m_v36_hook16u.jsonl`. v34@2863 = prior.
- **Remaining walls to 10k (v36 death mix): LOW-SPEED holes 600-1000m (multi-hole jump RHYTHM — v34/v36
  survive 4-cell chains @v205 with staggered jumps but the low-speed rhythm is fragile), NARROWS-AT-SPEED
  (the policy RELEASES the strafe-hold on the LAST strip cell when narrow_target_lane→None → drifts off
  the inner edge @v205; next lever = current-narrow-lane obs / hold-through-last-cell), trees.** Settled:
  single 2-cell FLAT holes are FORGIVING on device (probe 10/10, no sim change). Sim recalib note: gate
  0.56 ≈ deadline 0.67 − 30Hz pipeline 0.10; jump_exec_lat 1±1 slightly over-modeled (fresher obs).
- NN decides all actions. Branch `sim-calibration` (safe to merge; `runs/*.zip`+MODELS.md gitignored).
