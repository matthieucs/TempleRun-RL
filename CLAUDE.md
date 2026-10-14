# Temple Run RL — project memory (minimal)

Goal: an RL policy that reaches **10,000 m** on real **Temple Run** (`com.imangi.templerun` v1.39.0),
trained in a headless sim, deployed on a physical Android phone via **Frida** (read game state + inject
actions over TCP — no screen-scraping, no synthetic input). Full history: `docs/archive/CLAUDE_FULL_HISTORY.md`.

## ⚡ CURRENT STATE (2026-10-14) — START HERE

- **NEW DEVICE RECORD 16,782 m (v120 `models/templerun_runner_v1.zip`, hook clean-17k, phone e4de25a1 =
  Redmi Note 12 `topaz`).** 20-attempt record session, one run per scrcpy recording, keep-best:
  2252..9668 m then 16,782 (10 km crossed at frame 3102, ~5 min). Died Burnt-type 7 on a flame tower at v210.
  README GIF = `assets/demo_10k.gif` (overlay video around the 10 km banner).
- **REPO REORGANIZED (branch `repo-cleanup`):** `scratchpad/` (research) -> `experiments/`;
  `scripts/` -> `scripts/{device,train,validate,analysis,probes}/`; recipes + world -> `configs/`
  (`_v120_cfg.json` -> `configs/sim_world.json`); AGENTS.md / OVERNIGHT_LOG / CLAUDE_FULL_HISTORY ->
  `docs/archive/`. Paths below were rewritten accordingly. Scripts find siblings via a 3-line
  "repo bootstrap" `sys.path` header. `scratchpad/` is now local-only (gitignored).
- **⚠️ PRE-EXISTING (found during cleanup, NOT caused by it): the C++/Python equivalence gate is RED at
  4dfe4db** — `sim_equivalence --backend cpp --det-only` 0/7488, `test_rng_order` 0/60, all on
  `dist_next_must_jump` (Python streams it, C++ returns 0). Fix before trusting a new C++-trained arm.
- **SHIPPED SURFACE RENAMED:** v120 obs67 = `models/templerun_runner_v1.zip` (only shipped model; v133 +
  v120 obs66 zips removed from models/, still in git history). Recipe = `configs/train.sh` (v133's flags,
  fine-tunes runner_v1), world = `configs/sim_world.json` (was _v120_cfg.json). v120/v133 recipes now in
  `experiments/`. README/MODELS.md carry NO vXXX names (user request); keep vXXX only here + experiments/.
- **USB on e4de25a1 drops (cable/port, not vibration).** Temple Run's VIBRATE appop set to `ignore`.

## ⚡ CURRENT STATE (2026-07-31)

- **★★★★★ THE DEVICE MEDIANS IN THIS FILE ARE RIGHT-CENSORED, AND IT HID A NEW ALL-TIME RECORD.
  `capture_calib` ends an episode at 4000 LOGGED FRAMES. Re-audited every w18/w19 capture
  (`death_type` of the LAST frame + its velocity): several episodes were cut off WHILE STILL ALIVE AT
  THE VELOCITY CAP — v120 twice, v126 three times.**
  - **`captures/w18_v120_clean.jsonl` ep5 = 4000 frames, distance strictly monotonic 6.5 -> **15,454.4 m**,
    `death_type` 0, velocity 210.00 on the final frame. THE RUN DID NOT END; IT WAS TRUNCATED. So
    v120's max is **>= 15,454 m** (previous recorded best 15,002) and its true value is UNKNOWN.**
  - **⚠️ AND THE RECORDED v120 TABLE OMITS THREE REAL EPISODES from that same clean batch (15454.4,
    9648.4, 4588.3). Pooled over the three w18 files with exact-duplicate distances dropped, v120 is
    n=17, median 4831, >=8 km 7/17, >=10 km 2/17 — BETTER than the n=14 / 4736 / 5-of-14 / 1-of-14
    recorded below. The n=14 figures are what was reported and are kept for continuity, but they
    UNDER-state the champion.**
  - **⚠️ `captures/w18_v120_device.jsonl` ep4 is CORRUPT: its frames span a run reaching 11538 and then
    a NEW run starting under the SAME ep index (last frame reads 2109.4 at velocity 192.7). The ep
    counter did not advance — the documented wedged-stream failure. Audit per-episode monotonicity
    before pooling any batch.**
  - **⇒ TWO RULES. (1) RAISE THE FRAME CAP before quoting any median; a policy at the velocity cap
    covers ~3.8 m/frame, so 4000 frames is only ~15 km and the best runs hit the ceiling.
    (2) A capture episode is only valid if its distance is STRICTLY MONOTONIC and its last frame has
    a real `death_type`; otherwise it is a wedged stream or a merged pair of runs.**

- **CODE + WEIGHTS PUSHED TO GITHUB (`origin/sim-fidelity-holes-crevasse-turn`). `models/` now ships
  `ppo_v120_obs67_base.zip` (champion, obs67, behaviour-identical to the obs66 original) and
  `ppo_v133_sdonly_10002432_steps.zip` (same world + the step-down fix), with `models/MODELS.md`.
  The README's recipe is now v133's EXACT command, reproducible from the shipped base. `.gitignore`
  keeps scratchpad `*.py` / `*.sh` / `*_cfg.json` (the recipes) and drops the data dumps.**

## ⚡ CURRENT STATE (2026-07-30) — START HERE

- **★★★★★ NEW DEPLOY CHAMPION = v120 (`runs/ppo_v120_chaincoast_10002432_steps.zip`, obs66,
  512-net, hook 17j/17k). DEVICE-VALIDATED, USER-CONFIRMED ("the new policy is pretty good").**
  ```
              n   distances                                              median   max    >=8km   >=10km
    v111      8   2339 2439 2502 3243 4323 4780 5146 5711                 3783    5711    0/8     0/8
    v120     14   3252 3364 3475 3544 3582 3800 4642 4831 6070            4736   11538    5/14    1/14
                  8149 8199 8377 8476 11538
  ```
  - **THE TAIL IS THE RESULT: 5 of 14 runs past 8 km where v111's BEST of 8 was 5711 (Fisher
    p~0.06). Median +25%.**
  - **★ THE TARGETED DEATH MODE FELL AND THE MIX DIVERSIFIED — the same signature as v111's own
    landing_hazard win: ARC-ENDS-IN-HOLE 75% (6/8) -> 47% (9/19), and the mix went from
    {Ledge 7, Fall 1} to {Ledge 12, Fall 3, 7:2, Slide 1, Tree 1}.**
  - **⚠️ IT READ PARITY IN SIM (+1.2%, t=0.73). That verdict was WORTHLESS because the sim
    reproduces this killer at 2.2% vs the device's 24.3%. ⇒ WHEN THE SIM DOES NOT REPRODUCE THE
    DEATH MODE AN ARM TARGETS, ITS SIM VERDICT CARRIES NO INFORMATION — go to the phone.**
  - **RECIPE: v111's exact recipe + `chain_coast_kill_prob` 0.5 (`configs/sim_world.json`,
    `experiments/train_v120_chaincoast.sh`). v111 stays the obs65 fallback.**
  - **⚠️ CAPTURE HYGIENE, cost me a false "2/8": never leave two `capture_calib` processes on one
    stream -- they double-log the SAME physical runs (batch B eps1-3 were identical to batch A
    eps5-7 to the decimal). And `death_type` persists on the trailing frames, so aggregate on
    capture_calib's own `ep N: dist=` lines, not on max-per-episode.**

- **★★★ NEXT LEVER, MEASURED: THE SIM CLUSTERS HAZARDS ~3.4x TOO MUCH. Chains per 1000 CELLS (not
  per km -- the scale-free unit): DEVICE 8.87, SIM v120 world 30.58. Hazard DENSITY is close, so
  this is a SPACING-SHAPE error, and it is why the sim presents "my arc lands on a hazard" on 21.7%
  of takeoffs vs the device's 8.4%.**
  - **The five compound knobs explain only ~8 of the excess (x0.5 -> 22.19, x0.25 -> 22.46 = FLAT,
    so halving is the whole available effect). tight_pair 0 + pair_same_type 0 adds nothing further
    (22.68). The remaining ~2.5x is base generation.**
  - **⚠️ DO NOT chase the apparent density gap (198.7 vs 157.6 hazards/1k cells) without settling the
    basis: the ONSET measure on the SAME captures says exposure MATCHES (must_jump 31.2/1k frames sim
    vs 30.4 device). Device `must_jump` is WINDOW-PRESENTED, so a per-cell census can undercount it.**
  - **v125 = v120's recipe + the flattened world (`experiments/_v125_cfg.json`: compounds x0.5,
    tight_pair 0, pair_same_type 0 -> chains 22.68). Strictly more device-like at NO sim cost
    (median 1280 vs 1238) and the best training curve of any hardened arm (ep_dist 1707 @10M).
    DEVICE-TESTED (USER: "the phone is fine") -- n=10, median 4290, max 11740, >=8km 1/10,
    P(>=10k) 1/10. BETTER THAN v111, NOT BETTER THAN v120 (median 4736, >=8km 5/14).
    ⇒ FLATTENING THE CLUSTERING IS NOT THE LEVER. v120 REMAINS CHAMPION.**
  - **⚠️ DEVICE EVALUATION IS NOW THE BINDING CONSTRAINT, and it is an INFRASTRUCTURE problem: the
    injector dies mid-batch and the state stream wedges (a wedged stream re-logs the LAST run's
    distance verbatim -- ep1 read 6296.7119 identical to ep0 -- and a dead game reads pt=-1/dist=0).
    Three batches were truncated at 2-4 episodes. `nohup ... &` from the shell gets reaped; use
    PowerShell `Start-Process -WindowStyle Hidden -PassThru` (survives) and ALWAYS drop exact
    duplicate distances before aggregating. Recovery = force-stop, restart, 15 s, re-forward all
    three ports, re-inject.**
  - **⚠️ `land_on_hazard_kill_prob` (NEW, default 0) IS THE WRONG LEVER AT EVERY DOSE TRIED: it costs
    ~30% of the sim median for +0.1x of lift (0.05 -> lift 1.8x/median 872; 0.10 -> 2.0x/696;
    0.15 and 0.35 collapse the policy). It DOES work behaviourally (v123 cut arc-lands-on-hazard
    takeoffs 26.9% -> 17.9%), but it prices a configuration the sim OVER-GENERATES 2.6x, so the
    price is disproportionate. FIX THE GENERATION FIRST, then re-dose.**

- **★★★★★ THE STEP-DOWN JUMP IS SOLVED, AND IT NEEDED BOTH HALVES (USER-REQUESTED). v133 =
  `runs/ppo_v133_sdonly_10002432_steps.zip` = v120's EXACT world + ONLY the step-down fix:
  `--stepdown-jump-penalty 8.0 --stepdown-explore-prob 0.5`.**
  ```
    arm   penalty  guided   step-down jump rate      sim median
    v120     -        -       99% (148/149)             958
    v130    8.0       -      100% (112/112)               -
    v131    1.5      0.5      99% (129/130)               -
    v132    8.0      0.5       0% (0/125)   <- but 4 changes at once, device median 3423 = WORSE
    v133    8.0      0.5       0% (0/154)   <- ISOLATED on v120's world, sim median 1159 (+21%)
  ```
  - **★ NEITHER HALF WORKS ALONE. A penalty at 0.20/1.20/8.0 left it at 99-100% because at a 100%
    jump rate THE ALTERNATIVE IS NEVER SAMPLED, so the penalty is a constant offset that advantage
    normalisation removes. Guided exploration at a small penalty (1.5) also fails -- the sampled
    counterfactual carries too little advantage. Together they flip it to 0%. Same shape as the
    turn dead-letter (exploring-starts) crossed with the held-jump (needs an explicit price).**
  - **THE ACTION IS PROVABLY WASTED, measured before spending the arm: forcing NoOp at 619
    step-downs killed 1 = 0.2%, and 99% of them are REAL DECISIONS (took off AT the step-down;
    1-2% already airborne, 0% walked). Device rates: v120 95/95, v126 103/103 vs the HUMAN's 17/40.**
  - **DEVICE, n=12: median 4420, mean 5932, max 12007 (NEW BEST), >=8km 3/12, P(>=10k) 2/12 vs
    v120's 1/14. NOT RESOLVED (Fisher p~0.57) -- better tail, slightly worse median. ⚠️ Its FIRST
    batch read 2-of-6 over 10 km and the second regressed to the mean: the documented n=6 trap.**
  - **⚠️ v132 shows why the one-change rule matters: identical step-down fix, but bundled with
    crevasse_takeoff_std + a crevasse-dense profile -> device median 3423, clearly WORSE.**
  - **NEW INFRA: `sim_stepdown_walk_hint` (C++ oracle), `--stepdown-explore-prob/-anneal-frac`,
    `--stepdown-jump-penalty`. All default-off.**

- **★★★ (SUPERSEDED BY v133 ABOVE, kept for the reasoning) THE STEP-DOWN JUMP RESISTS A PENALTY ALONE.
  USER: "penalize the policy from random jumping the mid to low transition. It is useless and it can
  be a source of death." CONFIRMED: device step-down jump rate v120 95/95, v126 103/103, v111 25/26
  = ~100% vs the HUMAN's 17/40 = 42%.**
  - **AND THE ACTION IS PROVABLY WASTED: forcing NoOp at 619 step-downs killed 1 = 0.2%, and cost
    2.6% of the sim median (1437 -> 1400, inside noise). Walking is free. 99% are REAL DECISIONS
    (took off AT the step-down; only 1-2% were already airborne, 0% walked).**
  - **NOTHING MOVES IT. `stepdown_jump_penalty` (NEW) at 0.20 / 1.20 / 8.0 -> 99/99/100%. Adding
    GUIDED EXPLORATION (`--stepdown-explore-prob` 0.5 + `sim_stepdown_walk_hint`, the exact
    mechanism that broke the turn dead-letter) -> still 98%. The plumbing is VERIFIED, not assumed:
    SHP[17] reads back 8.000 across the FFI, the flag is in the recipe, train_fast forwards it, the
    oracle fires 459/96k env-steps with the policy jumping in 202, and the penalty condition is set
    at 100% of those takeoffs.**
  - **⇒ A verified-firing penalty + a verified-sampled counterfactual + a verified-free alternative
    still leaves the habit at 98%. My model of this situation is WRONG somewhere; do not simply
    raise the dose again. Next diagnostic: log the per-event ADVANTAGE of jump-vs-walk at a
    step-down during training, rather than inferring it from behaviour.**

- **⚠️ EVERY RETRAIN OFF v120 REGRESSES THE USER'S MID->HIGH METRIC. Level-3 graze in the SAME world:
  v120 7.4% (device 1.6-2.2%) vs v126 5.9-6.1% device, v130 24.6%, v131 19.4%. The USER spotted this
  from watching before I measured it. Cause: `crevasse_takeoff_std` was 0.0 in v120's world, so the
  graze is invisible AND unpriced; pricing it (v127) did NOT fix the timing (8.3% vs 8.8%) because a
  graze is a STUMBLE and the gradient is too weak. `crevasse_takeoff_std` 0.1 is now in
  `_v130_cfg.json` so at least every future gate can SEE it.**
  - **⇒ v120 REMAINS CHAMPION. v126/v127/v128/v129/v130/v131 all fail at least one of the three
    gates (step-down rate, graze/velocity-bled, distance).**

- **★★★ WAVE 19 (2026-07-30 late): FOUR ARMS, ONE PARTIAL WIN, AND THE SIM MISPREDICTED THE DEVICE
  EVERY TIME. v120 REMAINS CHAMPION.**
  ```
    arm    what it added (on top of v133 = step-down fix)        DEVICE n  median   graze
    v120   -                                                        14     4736    1.6-2.2%
    v133   stepdown penalty 8.0 + guided 0.5                        16     4708    0-3.5%
    v135   + stack_margin_penalty 1.0                                8     6720    10.6%
    v136   + drill(zone_up 0.8) + crevasse_takeoff_std 0.1           7     3530    9.1%
    v137   v135 + drill ONLY (no std), on a 31C phone               11     1838    15.6%
  ```
  - **★ THE USER DIAGNOSED THE REWARD CORRECTLY: "it gave the same amount of points for jumping over
    an obstacle whether there is an obstacle behind or not." TRUE -- `reaction_bonus` pays a flat
    +0.1 for ANY must_jump takeoff, so the policy is fully paid for reusing ONE trajectory where a
    stacked obstacle needs a different one, and only DYING (0.03% of frames) ever says otherwise.
    NEW `stack_margin_penalty` charges only the stacked case -> v135, the best device median of the
    session (6720 vs v120's 4736). ⚠️ But it costs the mid->high graze (10.6% vs 1.6-2.2%).**
  - **⚠️ A SYMMETRIC version (v134, `margin_bonus`, grading EVERY takeoff) BACKFIRED: tight takeoffs
    rose 13.6% -> 19.4%, sim median fell. Grading all takeoffs pushes timing later everywhere;
    scoping the charge to the stacked case is what works.**
  - **★★ NEW KNOB `zone_up_prob` (default 0.35 = unchanged). The level walk was HARDCODED
    `rnd < 0.65 ? down : up`, so HIGH ground is rare BY CONSTRUCTION -- which is why every
    "crevasse-dense" profile made level-3 caps per km go DOWN (0.55 -> 0.32 -> 0.27). At 0.75
    globally it lifts them 0.52 -> 0.76/km. THE KNOB WORKS; THE DRILL DOES NOT TRANSFER: v136/v137
    halved the sim graze (20.9% -> 11.3%) and were the two WORST device arms of the session.**
  - **⇒ THE SIM MISPREDICTED THE DEVICE FOUR TIMES IN ONE SESSION (v120 read parity and WON; v136
    read fixed and LOST; v137 read best-graze and collapsed). DO NOT crown on a sim median.**
  - **⚠️ PROCESS FAILURE TO NOT REPEAT: v132 and v136 each bundled FOUR changes and both regressed.
    The one-change rule was broken twice in one session after being cited as the reason v132 failed.**

## ⚡ EARLIER 2026-07-30

- **★★★★★ FRESH DEVICE MEASUREMENT OF THE OBJECTIVE (`captures/w18_v111_base.jsonl`, 8 ep, 33->38C):
  v111 = 4323/5711/2502/4780/5146/3243/2339/2439, MEDIAN 3783, and P(dist >= 10,000 m) = 0/8.**
  - **⚠️ THIS IS FAR BELOW THE RECORDED v111 BATCH (median 7799, 2-of-5 over 10 km). Some of it is
    thermal (the phone crossed the documented 38C throttle mid-batch) and some is that n=5 with a
    618..15002 spread has a meaningless median. TREAT 7799 AS UNRELIABLE; the honest current
    estimate of P(>=10k) is ~0.2-0.4 and TODAY measured 0.**
  - **★ 7 OF 8 DEATHS WERE DeathType 4 = LEDGE (the crevasse wall clip). That is a single dominant
    failure mode, and it is the USER's long-standing mid->high complaint.**

- **★★★★★ THE KILLER IS "MY JUMP ARC ENDS INSIDE A HOLE" — NOT THE CREVASSE CLIP. A frame-level
  forensic of all 8 fatal frames (`captures/w18_v111_base.jsonl`) OVERTURNS the DeathType-4 reading
  below: 0 of 81 CREVASSE encounters were fatal. Every dt=4 death is on a HOLE cell (pt 11/12/14/16)
  or the straight just past one, with world_y - ground_height at -7 to -44 = falling INTO the gap.**
  ```
    ep0 pt11  airborne 7 frames -> arc ENDS on the hole cell        -> falls in
    ep1 pt14  same shape                                            -> falls in
    ep2 pt14  lands on the hole, presses JUMP 1 frame later          -> too late
    ep3 pt16  airborne across a HoleLong, arc too short              -> falls mid-gap
    ep5 pt12  same
    ep6 pt11  jumps the CREVASSE (pt35), DESCENDING arc drops into a hole 2-3 cells later
    ep7 pt11  same shape
    ep4 pt2   never airborne; hole visible at t-3, JUMP pressed t-2  -> one frame late
  ```
  - **⇒ 7 of 8 = the ARC-ENDS-ON-A-HAZARD mechanism (the same one the tight-chain "coast" analysis
    isolated), NOT a crevasse timing defect. THE CREVASSE LINE (v121/v122, crevasse_graze_band) WAS
    AIMED AT THE WRONG MECHANISM.**
  - **⚠️ AND THE SIM DOES NOT REPRODUCE IT (tight-chain death 2.2% sim vs 24.3% device), so A SIM
    PARITY VERDICT ON THIS ARM IS UNINFORMATIVE. Judge it on the phone.**

- **★★★ DEVICE, 2026-07-30, same session, 33->38C. THE FIRST POSITIVE SIGNAL ON THE REAL OBJECTIVE:**
  ```
    v111  n=8  2339 2439 2502 3243 4323 4780 5146 5711   median 3783  max  5711  P(>=10k) 0/8
    v120  n=5  3252 3364 3544 8199 11538                 median 3544  max 11538  P(>=10k) 1/5
  ```
  **v120 = v111 + `chain_coast_kill_prob` 0.5 (the death model for exactly the mechanism above) is
  the ONLY arm that produced a 10 km run today, and it did so despite reading PARITY in sim (+1.2%,
  t=0.73). ⚠️ NOT RESOLVED: 1/5 vs 0/8 is p~0.38, and v120's batch was DISRUPTED (the injector died
  mid-run; batch B re-attached and observed the SAME run finishing, which double-counts 11538 if you
  aggregate by max-per-logged-episode -- use the DEATH FRAME). NEXT: one clean uninterrupted v120
  batch on a cooled phone.**

- **★★★★ THE CREVASSE TAKEOFF INVERSION REPRODUCES IN THE SIM — the 2026-07-27 note that it is a
  "SIM-TO-REAL TIMING GAP, NOT A TRAINING FAILURE" is FALSIFIED.** Device slope
  d(cp)/dv = **-0.01051 +- 0.00262** (n=30, the worst ever recorded here; human **+0.00828**).
  The SAME policy in the C++ trainer: **-0.00574 +- 0.00088 over 150 crossings (6.5 s.e.)**.
  ⇒ The defect IS reproducible and therefore trainable. What kept it alive: the sim's level-3 graze
  was **0.7% vs the device's 3.3%** (`crevasse_takeoff_std` sat at 0.0 while this file's own
  calibration says 0.2), so the early takeoff was nearly FREE in-sim. std 0.1 -> 5.1% graze.
  - **⇒ THE REMAINING LEVER, and it is the same shape as every win in this file: the sim charges a
    STUMBLE (5.1% graze) where the device KILLS (7/8 Ledge). Calibrate `crevasse_graze_band` so the
    sim produces device-like LEDGE DEATHS, not just slowdowns, then retrain.**

- **★★★★ PROVEN DEAD END — THE `predicted_jump` LAW CANNOT FIX THE INVERSION, AND THIS RETIRES THREE
  ARMS (v94/v95, v121, v122). For a fixed learned threshold r the implied takeoff distance is:**
  ```
    old law  raw=(2-cp)-0.5*span      -> cells-to-cap = (2r-0.5) + 0.5*span   slope +0.00667
    "vel-inv" law  (2-cp)/span/2      -> cells-to-cap = 2r*span               slope +0.00667  (IDENTICAL)
  ```
  **The flag only RESCALES the threshold; it does not change the velocity dependence at all. Measured
  confirmation: v111 (old) -0.00537, v121 (vel-inv, retrained 10M) -0.00516 = unchanged.**
  - **v122 tried removing the lead term (PJ_LEAD_FRAC 0.0 -> constant implied takeoff, which is what
    the HUMAN does: cells-to-cap 0.70/0.35/0.22 at v140-180/180-200/200-240, slope -0.0029).
    IT MEASURED WORSE (-0.00748) for reasons unrelated to geometry: (1) with lead=0,
    raw=(2-cp) in [1,2] and the mapping (raw+0.5)/2 SATURATES at 1.0 for every cp<=0.5 -- which is
    also PJ_NONE, the "nothing ahead" sentinel; (2) with no span term the cue is a function of cp
    ALONE and duplicates the cell_progress column. REVERTED. ⇒ The lead term is what makes the cue
    informative; the gap is that the sim never CHARGES for getting the fusion wrong.**

- **★★★ TWO REAL BUGS FIXED IN THE C++ TRAINER (both behaviour-verified neutral: v111 scores 2838
  before and after):**
  - **A LIVE STACK BUFFER OVERFLOW. `double raw[RAW_LEN]` was 37 doubles while `snapshot()` writes
    R_soft_stepdown(37)/R_next_soft_stepdown(38), and `shaping_reward` READ prev_raw[37] for the
    step-down gate -- past the array and into the next State fields. Fixed by sizing storage from
    the full enum (RAW_CAP).**
  - **`obstacle` WAS NEVER EXPOSED in the C++ raw state, so every device forensic silently fell back
    to window-presented must_jump/must_slide flags and produced a DIFFERENT chain segmentation than
    the device (a 20x disagreement in tight-chain lethality that was pure basis mismatch). Now
    exposed; RAW_LEN 37 -> 42, aligned both sides. NEW `experiments/cpp_rollout_dump.py` dumps C++
    rollouts in DEVICE CAPTURE FORMAT, so every forensic runs on 192 episodes in ~2 min instead of
    24 episodes in 25 min via the CPU ref.**
  - **⚠️ `experiments/sim_patterns.py` merges `DEVICE_TRUE_CFG` UNDER the passed config, so it scores a
    DIFFERENT WORLD than the trainer. Do not compare its output to a C++ rollout.**

- **★★ v120 = tight-chain coast death model (`chain_coast_kill_prob`, NEW, default off). Device
  calibration: a hazard within 2 cells of the previous one with NO new jump/slide since is 47%
  lethal on device vs 5% when a fresh input is taken; the sim was ~10%. Dose 0.5 reproduces it
  (42.9%). TRAINED -> +1.2% vs v111 (t=0.73, NOT RESOLVED) = parity. The knob is device-true and
  stays (default off); it is not by itself a win.**

## ⚡ PREVIOUS STATE (2026-07-28 LATE)

- **★★★★★ THE DEFECT IS CASCADES, AND IT IS *NOT* A PERCEPTION PROBLEM — MEASURED ON 8 EPISODES.
  USER (2026-07-28): "one or at max two runs are sufficient for me to know what's good and bad."
  CORRECT, and the arithmetic backs it: a run yields ONE distance but ~7 three-obstacle chains,
  ~50-100 hazard encounters and ~200 takeoffs, so a per-encounter RATE has 50-100x the sample of a
  median FROM THE SAME RUN. 48 episodes is only for ranking two similar policies by median — which
  is the expensive, low-power question and the wrong one to ask.**
  ```
                chains   rate/km   deaths in chain   per-encounter   3rd obstacle VISIBLE at commit
    v111 device    55      0.93          6              10.9%                0/55 = 0%
    HUMAN          45      1.09          0               0.0%                0/45 = 0%
  ```
  - **6 OF v111'S 8 DEVICE DEATHS ARE INSIDE A 3-OBSTACLE CHAIN. Exposure is MATCHED (the human meets
    MORE chains per km) and the gap resolves at p~0.03 on 100 encounters.**
  - **★ THE HUMAN IS EQUALLY BLIND — 0/45 chains have the third obstacle inside the +3 horizon at
    commit — AND NEVER DIES. ⇒ MORE LOOKAHEAD IS NOT THE FIX, and that retroactively explains why
    the HORIZON OBS (`dist_next_must_jump/slide`, 6 cells) died across three arms incl. a 40M
    consolidation: it was solving a non-problem, the same shape as the airborne penalty added for a
    device jitter that did not exist. Do NOT revive it on cascade grounds.**
  - **v111's chain deaths cluster at TIGHT h2->h3 gaps (3+1 33%, 3+2 33%, 4+2 50% vs 4+4 0%, 4+3 0%),
    i.e. RECOVERY SPEED, not sight. First probe: v111 is airborne when h2/h3 arrives 59.2% vs the
    human's 51.2% (n=510/726, ~2.7 s.e.) — real but small. ⚠️ The landing->next-action half of that
    probe is INVALID for the human: human inputs live in the `gi_*` swipe buffer, not the `action`
    field, so it read empty. Fix that before trusting a reaction-time comparison.**
  - **★★★★ THE MECHANISM IS "COASTING", AND THE SIM DOES NOT PUNISH IT. Within a chain the killer is
    a TIGHT h2->h3 gap (1-2 cells): device 9/37 = 24.3% vs 1/125 = 0.8% at gaps 3-4 = a 30x cliff.
    The behaviour is riding ONE jump arc from h2 into h3 instead of taking a fresh action:
    device COAST 8/22 = 36.4% lethal vs 1/15 = 6.7% when a fresh input is taken, and ALL 8 fatal
    coasts die exactly ON h3 (4 airborne = could not act, 4 grounded = landed badly). The HUMAN
    coasts 0% of 9 tight chains and takes MORE inputs through the pair (1.11 vs 0.49 onsets).**
    ```
                       chains/km  tight share  TIGHT death   by h2->h3 gap
      SIM (192 ep,431 km)  3.42       12%         2.2%       g1 6% g2 1% g3 2% g4 2%   <- FLAT
      DEVICE (169 km)      0.96       23%        24.3%       g1 19% g2 29% g3 0% g4 1% <- 30x cliff
    ```
  - **⇒ THE SIM GENERATES 3.6x TOO MANY CHAINS, HALF THE TIGHT ONES, AND MAKES THEM 11x LESS LETHAL.
    That is why v113 ("more cascade exposure") failed: it added MORE of a chain the sim does not
    punish. The fix is not exposure, it is making the tight pair as lethal as the device's.**
  - **⚠️ NOT a perception fix: h3 is INSIDE the +3 horizon at 82% of coast takeoffs (18/22), the
    policy already carries landing_hazard at -0.01335 AND a 0.30 penalty, and the human is equally
    blind yet never dies. Takeoff cp does NOT separate coast from safe (0.37 vs 0.41, overlapping).**
  - **⚠️ Do not compare CPU-ref rollouts to C++ ones without checking the config: `sim_patterns.py`
    merges `DEVICE_TRUE_CFG` UNDER the passed config, so it scores a DIFFERENT world (it read
    tight-chain lethality 36.4% at n=22 where the C++ trainer reads 2.2% at n=180).**
  - **⚠️ DETECTOR BUG FIXED EN ROUTE: `scripts/analysis/detect_patterns.py` computed BLIND_LANDING_JUMP with
    the RETIRED span fit `0.0204*v - 0.94`, over-stating the arc 13-17% and flagging the WRONG
    landing cell. Corrected to the shipped law (+ height scaling), the pattern goes from "11 events,
    45% lethal" to "38 events, 11% lethal" against the HUMAN's 28 / 18% ⇒ v111 is at PARITY-OR-BETTER
    and it was never a defect. Every named-pattern verdict computed before this is suspect.**

- **⚠️⚠️ RETRACTION OF THE TWO ENTRIES BELOW, BOTH MY OWN, BOTH BASIS ERRORS. After controlling the
  confounds the sim is WELL CALIBRATED and there is no fidelity gap to exploit here:**
  - **"2.5-7.7x TOO LETHAL PER ENCOUNTER" — WRONG, mismatched ATTRIBUTION. The device side counted a
    death only if it happened ON the hazard cell; the sim side pinned every death to the last hazard
    seen EVER (sticky, never cleared), so 369 of 384 sim deaths were attributed to a hazard. Matched:
    crevasse 1.31%, flame 0.60%, hole 2.89%.**
  - **"2.9x TOO LENIENT AT CREVASSES" — ALSO WRONG, POLICY-STRENGTH confound. The 3.84% device figure
    pools 176 captures spanning years of weaker policies (crevasse was the #1 killer for the v51 era).
    CHAMPION-MATCHED (v107+v111 device vs v111 sim): crevasse 1.31% vs 0/203, flame 0.60% vs 0.44%
    (MATCHED), hole 2.89% vs 1.71%. Nothing separates at these n.**
  - **"FLAMES 6-8x UNDER-GENERATED" — ALSO WRONG, ENCODING basis. A `path_type` census drops
    TreeSlide, which rides a STRAIGHT cell with the obstacle in a field the C++ raw state does not
    expose — the same trap the narrows census hit in July. On the ENCODING-INDEPENDENT basis
    (`must_jump`/`must_slide` onsets, booleans defined identically both sides,
    `experiments/slide_exposure.py`): must_jump 31.2/1k sim vs 30.4 device, must_slide 11.4 vs 12.3,
    slide share 26.8% vs 28.8%. HAZARD EXPOSURE MATCHES TO WITHIN 7%.**
  - **⇒ THREE headline claims in one session, all dissolved. THE RULE, restated: compute BOTH sides
    with the SAME code AND the same attribution rule AND on an encoding-independent quantity before
    believing any sim-vs-device number. See [[aggregate-hides-per-unit-error]].**

- **★★★★ A REAL BUG, FOUND BY THE FAILED ABLATION: `shaping_reward()` in cpp_sim/sim.cpp opened with
  a HAND-MAINTAINED fast-path guard listing the knobs worth running for, and it was NEVER updated
  when `stepdown_bonus_gate`, `landing_hazard_penalty` or `turn_takeoff_penalty` were added — so any
  of those, SET ALONE, returned 0.0 and SILENTLY DID NOTHING.**
  - **⚠️ SCOPE, do not overstate it: v111 was NOT affected. Its recipe also sets narrow_hold_bonus /
    airborne_action_penalty / strafe_penalty / turn_buffer_bonus, all in the list, so the guard
    passed and `--landing-hazard-penalty 0.30` really did apply. The bug only bites an ISOLATED
    ablation — which is exactly the experiment you run to attribute a result to one knob.**
  - **FIXED by scanning the array instead of a list that can go stale. New diagnostics
    `sim_get_shaping(i)` / `sim_shaping_count()` read the coefficients back across the FFI — added
    after every Python-side layer looked correct and the knob still read inert. ⇒ WHEN A KNOB READS
    INERT, READ THE VALUE BACK FROM THE OTHER SIDE OF THE BOUNDARY instead of re-checking the plumbing.**

- **★★★★★ THE CHECKPOINT RULE IS WRONG AND IT INVERTED A VERDICT. "Take the checkpoint with the best
  training `ep_dist`" picked v117@6M, which is the WORST of the three on a paired eval. The LADDER,
  same 8 seeds vs v111: @2M -12.5% (t=-2.43) -> @6M -3.7% -> @10M +2.4%. That is a monotone LEARNING
  CURVE, not noise — the arm was still improving when the rule declared it finished.**
  - **WHY: `ep_dist` is logged DURING training, under an annealing entropy coefficient, so it mixes
    policy quality with exploration noise. It is not a proxy for a deterministic eval.**
  - **⚠️ THIS AFFECTED v113/v114/v115 TOO — all three "peaked at 8M and dipped at 10M" and were
    judged at 8M on this rule. Their verdicts are suspect and should be re-read off a ladder.**
  - **NEW RULE: evaluate the CHECKPOINT LADDER with eval_paired (~17 s per checkpoint-seed, so a
    4-point ladder at 8 seeds is ~9 min) and pick from that — then RE-TEST the winner on FRESH SEEDS
    (`--seed-offset`, added today). Scoring on the seeds you SELECTED on is biased high: v117@10M
    read +2.4% on the selection seeds and +1.2% on seeds 100-115.**

- **★★★ v117 (obs66 cue) @10M vs v111 = +1.2%, t=0.97 on 16 FRESH seeds = INDISTINGUISHABLE. NOT the
  "-3.7% flat-to-negative" the @6M checkpoint suggested. The column is LEARNED, FREE and NEUTRAL on
  outcome, with no erosion of landing_hazard (-0.01482 vs v111's -0.01335).**
  - **v119 = the same recipe at 20M SETTLES IT: the late ladder PLATEAUS at parity — @14M +0.7%,
    @16M -3.3%, @18M -0.2%, @20M -3.5%, none resolved. So v117's rising ladder was the arm RECOVERING
    from warm-start disruption back to parity, NOT a curve that keeps climbing. ⇒ KEEP the obs66
    column as infrastructure (free, learned, no erosion); do NOT crown it. v111 REMAINS CHAMPION.**
  - **RE-CHECKED THE ARMS THE BROKEN RULE MIS-JUDGED (their 10M finals were never scored; widened to
    obs66 by a behaviour-preserving zero-column surgery, 0/300 each): v113 -3.6%, v114 -3.6% (both
    unresolved), v115 -7.9% RESOLVED WORSE (consistent with its original -9.7%). NO HIDDEN CHAMPION —
    the rule inverted v117's verdict but cost us no winner.**

- **★★★ v118 = v117 + `--slide-landing-hazard-penalty 0.30` (the amplifier). WORSE AT EVERY MATCHED
  CHECKPOINT: @8M -4.3% (t=-2.26, RESOLVED), @10M -2.3%. Against v117 at the matched 10M point the
  penalty costs ~3.5 points (+1.2% vs -2.3%). ⇒ v111 REMAINS CHAMPION.**
  - **The penalty did NOT amplify the cue (-0.00615 -> -0.00610) where landing_hazard's went
    -0.0039 -> -0.01163. It DID change behaviour: states whose arc lands on a slide hazard fell
    2.41% -> 1.62%, i.e. the policy avoids the configuration ~33% more — and still scores worse.**
  - **★ WHY, AND IT IS A RULE THIS FILE ALREADY CONTAINS: "before pricing a cue, check the cue EXISTS
    in the stream and that the priced event is DISCRETIONARY." Existence was verified hard (52.6x
    lift over 26,854 takeoffs); DISCRETIONALITY WAS NOT — and the spec itself says these takeoffs are
    100% FORCED by a must_jump. Pricing a forced action is the conflicting-gradient trap that already
    sank slide_jump_penalty. The held-jump and landing_hazard wins both priced a DISCRETIONARY action.
    ⇒ When the action is forced and only its TIMING is free, a flat per-event penalty cannot tell the
    good instance from the bad one. See [[weak-signal-needs-amplifier]] (boundary condition added).**
  - **KEEP: the obs66 column (learned, free, no erosion), `scripts/analysis/cue_response.py`, the guard fix,
    the death-type relabelling. DO NOT deploy v117 or v118.**


- **★★★★★ THE SIM IS 2.5-7.7x TOO LETHAL PER HAZARD ENCOUNTER, AND THAT IS THE AXIS SIX FAILED ARMS
  NEVER TOUCHED. `experiments/per_encounter.py`, identical encounter definition on both sides
  (a TRANSITION onto a hazard kind), v111 in its own world vs 176 device captures:**
  ```
  hazard      SIM encounters/deaths/rate      DEVICE encounters/deaths/rate     sim is
  crevasse       1980 / 193 /  9.75%             4139 / 159 / 3.84%             2.5x too lethal
  flame           166 /  14 /  8.43%             5331 /  58 / 1.09%             7.7x too lethal
  hole           2320 / 162 /  6.98%            11106 / 182 / 1.64%             4.3x too lethal
  ```
  - **VALIDATION: the device column reproduces the 1-3%/encounter this project measured
    independently (v114 hole 3.12%, flame 3.08%), so the harness agrees with prior work.**
  - **AND THE SIM UNDER-GENERATES FLAMES ~7x: flame is 3.7% of sim hazard encounters (166/4466) vs
    25.9% on device (5331/20576) — in v111's world (`_v108_cfg`), which PREDATES v114's mix
    calibration. Two errors that CANCEL in the aggregate: too few flames x far too lethal each.**
  - **⇒ THIS IS THE DISTANCE-UNIT BUG'S TWIN. Difficulty was calibrated on the MEDIAN (v90 sim 4770
    vs device 5116 = "7% gap") and matched, while the PER-ENCOUNTER rate underneath was 3x off,
    because the two errors offset. A per-unit metric that disagrees while the aggregate agrees
    localises the bug to the model, not the amount. See [[verify-units-before-comparing]].**
  - **⚠️ BOTH CAVEATS RUN THE SAFE WAY: the device pool includes many WEAKER policies (inflates the
    device rate) and 10 Hz capture can alias a hazard passed at speed (inflates it again), so the
    true gap is if anything LARGER. NEXT: calibrate the death-model probabilities down per hazard,
    then re-check that the champion's sim median still lands near its device median.**

- **★★★★ DEATH-TYPE LABELLING BUG, FOUND AND FIXED IN BOTH SIMS — every sim-vs-device "death
  signature" comparison ever made in this project was reading a miscoded label.** Five kill sites
  (crevasse cap deep-clip, step-up apex miss, grounded-at-ledge, wide-gap landing-inside-gap,
  non-airborne LevelChange) emitted **DeathType 2 (Tree)** for what the device reports as
  **4 (Ledge)** — and the code comments SAID "-> Ledge" while the code said 2.
  - **BEHAVIOUR-NEUTRAL: `do_kill` only sets a label and the reward is death-type blind, so NO
    trained policy was ever affected. The damage is purely diagnostic — and total: the sim reported
    0.0% Ledge deaths against the device's 36%.**
  - **DEATH MIX after the fix, matched basis (sim started at distance 0 like the device, both
    bucketed by distance at death, `experiments/deathmix_basis.py`):**
    ```
    band        source   Fall    Tree   Slide   Ledge   Burnt
    0-1000      SIM      59.7%   15.1%   7.5%    3.4%   14.3%
    0-1000      DEVICE   52.0%    7.5%   0.9%   35.5%    0.2%
    3000+       SIM      32.8%   11.7%  22.3%    2.0%   31.2%
    3000+       DEVICE   24.9%   11.8%   8.9%   36.3%    3.8%
    ```
    **Fall and Tree match well. Ledge is the device's #2 killer at a steady ~36% in EVERY band and
    the sim barely produces it; Burnt is inverted. The relabelling accounted for only 0.5-3.4 points
    — the rest is the real per-encounter gap above.**

- **★★★ v117 = `slide_landing_hazard` (obs 65->66). THE CUE IS LEARNED — AND IT DOES NOT MOVE THE
  OUTCOME. v111 REMAINS CHAMPION.**
  - **THE CUE, measured with the SHIPPED encoder over 26,854 device takeoffs
    (`experiments/slh_device_check.py`): cue=1.0 -> 8 slide deaths in 51 takeoffs = 15.69%;
    cue=0.0 -> 80 in 26,803 = 0.30%. A 52.6x LIFT, and takeoff cp is 0.50 when it fires vs 0.28
    when it does not — the policy takes off LATE exactly when its arc will end on a slide hazard.
    89% of the victims DID press slide and were still airborne; the takeoffs are 100% forced by a
    must_jump, so it is a TIMING error, not a decision error.**
  - **LEARNED, on the leading indicator: dP(JUMP) = -0.00450 @2M -> -0.00615 @6M -> -0.00631 @10M,
    correct sign, monotone — AHEAD of where landing_hazard sat in v90 (-0.0035 @2M). And it cost
    nothing: v117's landing_hazard response is -0.01482 vs v111's -0.01335 = NO erosion.**
  - **FLAT ON OUTCOME: sim median -3.7% (t=-1.25, not resolved, 8 paired seeds) and the TARGETED
    metric is dead level — slide-type deaths 42.90% (v111) vs 42.71% (v117) over 1536 deaths each
    = 0.15 s.e. That is a real null on a big sample, not noise.**
  - **⇒ THE HONEST READING, and it is a MEASUREMENT lesson: the effect was designed for ~3% of
    deaths (the corrected ceiling — see below), and eval_paired resolves ~2.5% of the MEDIAN. A
    3%-of-deaths fix CANNOT show up there. Worse, the judge itself is a world that is 2.5-7.7x too
    lethal per encounter. Fix the lethality before re-judging this arm.**
  - **⚠️ MY OWN SPEC SAID THE CEILING WAS ~33%. IT IS ~3%: over 1077 pooled device deaths the cue
    flags 32 (3.0%) one jump ahead. The 33% came from a SIM rollout whose flame-death share (33%) is
    nothing like the device's (7.1%) — i.e. it was an artifact of the very lethality bug above.**
  - **⚠️ EXPOSURE WAS NEVER THE GAP, measured BEFORE spending steps (`experiments/slh_exposure.py`):
    the sim already fires this cue on 1.12% of takeoffs vs the device's 0.19% = 6x OVER-served. A
    jumpslide drill moved it 1.12% -> 1.06% = INERT, and was cancelled. This also explains v116's
    null: it drilled a situation already over-served. ⇒ ADDING PRACTICE FOR AN OVER-SERVED SITUATION
    IS NOT A LEVER; the user's "specific episodes" framing needs the exposure CHECKED first.**
  - **KEEP the obs66 column (learned, free, no erosion) but do NOT crown v117.
    Artifacts: `runs/ppo_v111_obs66_slh.zip` (surgery base, 0/400), `runs/ppo_v117_slh_*.zip`.**
  - **NEW TOOL `scripts/analysis/cue_response.py` — flips ONE obs column on states the policy actually visits
    and reports dP(action). This is the kill criterion the horizon obs never had. Validated against
    a known value (v111 landing_hazard reads -0.01335 vs -0.01163 on record). Reference scale:
    v111 landing_hazard -0.01163 | eroded -0.0005 | horizon obs (dead) -0.00005; bar |dP| >= 0.0005.
    ⚠️ Flipping columns on RANDOM vectors reads ~0 and is meaningless — use on-distribution states.**

## ⚡ PREVIOUS STATE (2026-07-28) — START HERE

- **★★★★★ CHAMPION = v111 (`runs/ppo_v111_lh030_final.zip`, obs65, 512-net, hook 17j or 17k).
  DEVICE n=5: 618 / 1730 / 7799 / 13699 / 15002 → MEDIAN 7799, mean 7235.
  ★ NEW ALL-TIME RECORD 15,002 m (prev 12,779), and a SECOND run at 13,699 m — both beat the old
  record IN THE SAME BATCH. 2 of 5 runs over 10 km.** Prior champion v107 (median 5848, max 10955).
  ⚠️ n=5 and bimodal (three short runs), so the median is NOT a ranking — the death MIX is the
  evidence (below). Batch was interrupted; finish it before crowning formally.

- **★★★★ THE WIN, AND IT IS THE PROJECT'S MOST REUSABLE METHOD (3rd time it has worked):
  `--landing-hazard-penalty 0.30` = dock a JUMP whose predicted ARC lands on a must-jump hazard.**
  - **WHY, measured end-to-end: v107 died 5-of-7 with dt=4 Ledge ("my arc landed me somewhere
    fatal"). Frame-level forensic of the fatal takeoff showed `landing_hazard = 1.0` ALREADY SET one
    frame early AND at takeoff — so it was NEVER a perception gap. The counterfactual was: flipping
    that cue on real states moved P(JUMP) by only −0.0039 on a 0.036 base (~10% relative), nowhere
    near enough to veto a jump the rest of the obs demanded. The cue had also ERODED through the
    lineage: v92b −0.0078, v104 −0.0005, v107 −0.0039.**
  - **RESULT: cue response −0.01163 (3.1x v107, the strongest ever measured here, above v90's mature
    −0.0067) at ZERO median cost in sim; on DEVICE the targeted mode fell from 5-of-7 dt=4 Ledge to
    1-of-4, with the remaining deaths spread across other causes.**
  - **⇒ THE PATTERN: a correct-but-quiet cue needs an explicit PRICE. Same shape as held-jump
    (death model + airborne_action_penalty → 10.67 → 0.00/km). See [[weak-signal-needs-amplifier]].**

- **★★★★ THE TRAINER IS NOW C++ AND IT CHANGED THE ECONOMICS: `--backend cpp` (cpp_sim/), 7.07x at
  matched geometry — 26-28k steps/s vs the torch sim's 3.7k. 10M steps in ~6 min (was ~45).**
  - **This is why the 256x16 geometry is back: 1024x4 was only ever a workaround for the launch-bound
    torch sim, and it COST US (v101 was 25x worse on held-jump). C++ makes the CORRECT geometry the
    fastest one. `--num-envs 256 --n-steps 16`.**
  - **GATES: `python scripts/validate/sim_equivalence.py --backend cpp --det-only` → 6552/6552, plus
    `python cpp_sim/test_rng_order.py` → 60/60 lockstep (the deterministic tier injects fixed tracks,
    so it CANNOT see the generator — the RNG test is what covers generation).**
  - **⚠️ THE GATE EARNED ITS KEEP: it caught that landing_hazard_penalty was wired into reward.py and
    the C++ but NOT the CPU sim's constructor — a silent 0.25-reward divergence that would have
    trained a policy against a reward the reference does not implement. Do not drop it.**
  - **TORCH IS RETIRED from the loop (user decision): CPU ref = specification, C++ = trainer. The
    torch sim still exists but is NOT gated and WILL drift. Gate takes ~3 min instead of ~45.**

- **★★★ SIM FIDELITY — everything below is DEVICE-MEASURED with no fitting, all gated:**
  - **TURN DENSITY WAS 2.5x TOO HIGH — the biggest generation error found today. Sim emitted a
    junction every 3.4 cells vs the device's 8.6 (29.3% of cells vs 11.6%). Cause: `turn_prob_late`
    0.5. Fixed to 0.2 → 11.8%. ⚠️ Counter-intuitive consequence: FEWER turns makes the world HARDER
    (junction cells carry no obstacles, so the excess was diluting hazard density).**
  - **OBSTACLE MIX, in STRUCTURE units (a 2-cell crevasse/HoleLong is ONE obstacle): crevasse was
    2.12x OVER (34.2% vs 16.1%), flame 0.46x UNDER (8.5% vs 18.7%). Recalibrated via
    `experiments/calib_mix.py` (iterative: weights are NOT the emitted share). This also corrects the
    2026-07-27 claim that the mix was "within 3 points" — that check used per-CELL counts on
    non-crevasse obstacles and missed both the crevasse share and the density.**
  - **THE GAME REPEATS THE SAME OBSTACLE TYPE, and only for SLIDES: device's two most common
    consecutive pairs are TreeSlide→TreeSlide (9.8%) and Flame→Flame (8.1%) = 17.9% of all pairs,
    vs 2.9% in sim. Type-agnostic repetition is the WRONG model (Hole→Hole is BELOW its independent
    rate on device). New `pair_same_type_prob` 0.6, slides only → ~14%.**
  - **POST-TURN: the device concentrates hazards at +3 (36.6% of junctions) not +1 (3.2%).
    `turnobstacle_prob` could only ever fill +1 → new `turnobstacle3_prob`.**
  - **STEP-DOWN TO LOW GROUND IS FOOT-CROSSABLE (`stepdown_grounded_ok`): human walks 17 of 26
    step-downs to low ground and NEVER dies; every policy jumped 100% of 55. The device streams
    must_jump=True there too, so the OBS is faithful — it is purely a DEATH-RULE gap.**
  - **JUMP AIRTIME SCALES WITH THE HEIGHT CHANGE (`jump_dur_stepdown_scale` 1.17, step-up 0.74
    measured but HELD OUT). Step-down flies 3 cells not 2; step-up is ~26% SHORTER. Two independent
    pipelines agree (human 30 Hz n=50, three policy sets 10 Hz).**
  - **`landing_hazard` now uses that height-scaled span (`LAND_SPAN_LEVEL_SCALED`) — it was wrong by
    a whole cell exactly at crevasses.**

- **★★★★★ THE GENERATOR IS DRAW-THEN-SUBSTITUTE (REJECTION SAMPLING), NOT i.i.d. WEIGHTS — READ FROM
  THE GAME (dump.cs, 2026-07-28). Our sim is the WRONG SHAPE, which is why no dose of
  `pair_same_type_prob` can reach the device's clustering.**
  - **WHAT THE GAME HAS (all SEEN in dump.cs, not inferred):
    `PathElement.ACertainTypeCheck(PathType currentType, PathType parentType)` @283485 RVA
    0x12E3DF4 (84 B) = an explicit PAIRWISE admissibility rule; `OriginalPick` 0x98 and `FinalPick`
    0x9C = the generator DRAWS a type and then SUBSTITUTES it; `SpacesSinceLastObstacle` 0x7C and
    `BackToBackObstacleCount` 0x80 per node. `GetRandomPathElementForSide` @283488 RVA 0x12E3E48 is
    3500 B (~875 instrs) = a hardcoded if/else chain of `Random.value < literal`, far too big for a
    table; `HalfTheTime()` (36 B) is an ad-hoc 50/50 helper.**
  - **WHAT IT DOES NOT HAVE (exhaustive negatives, all file-wide): NO authored patterns/templates/
    combos (zero `ObstacleType[]` / `List<PathType>` collections; all 74 path prefabs are ONE cell);
    NO string literal containing "obstacle"; NO per-type probability field of any kind — the game
    uses named probability statics ONLY for bonus items (`sProbabilityCoinBonus/Invincibility/
    Vacuum/Boost`) and deliberately none for obstacles, so the type numbers are COMPILED-IN
    LITERALS. Per-type control is only the boolean gates `sAllowPathHoles/…/sAllowTreeSlide`.**
  - **★ THE STRUCTURAL KEY, and it explains BOTH device deviations with ONE mechanism: holes,
    narrows, flames and level-changes are `PathType` (the CELL GEOMETRY) and must survive
    ACertainTypeCheck + DoesPieceTypeFitOnSide + WillPieceTypeConnectOnSide + the spacing budget;
    TREES (Stumble/Jump/Slide) are `ObstacleType` placed onto an ALREADY-CHOSEN straight by a
    separate `AddObstacle` pass those filters never touch. A filtered type is pushed BELOW chance
    (Hole->Hole 6.1% vs 10.3% independent); an unfiltered one sits ABOVE it (TreeSlide->TreeSlide
    9.8% vs 1.5%). That is the signature of CONSTRAINED REJECTION SAMPLING.**
  - **MEASURED SIM-vs-DEVICE on the JOINT pair basis (structure units, `config=` verified):
    slide-repeat 7.9% vs 17.9%, slide-led share 30.8% vs 50.7%, Hole->Hole 7.9% vs 6.1%.
    `pair_same_type_prob` SATURATES at 10.4% / 33.3% even at prob 1.0 — it can never close the gap,
    because it only fires when the PREVIOUS obstacle was a slide and slides lead too few pairs.
    ⇒ THE FIX IS STRUCTURAL: implement draw -> pairwise-reject -> substitute, with trees on a second
    unconstrained pass. The true literals are recoverable only by disassembling RVA 0x12E3E48.**
  - **NEW KNOB `pair_anti_repeat_prob` (+ `pair_anti_repeat_types` (1,4)) is IN and default-off; at
    0.6 it lands Hole->Hole on 6.34% vs the device's 6.1%. It is a SYMPTOM-LEVEL patch on the above
    and should be retired once rejection sampling is implemented.**
- **★★★★ FIVE FIDELITY ARMS, ZERO GAINS — AND THE MEASUREMENT IS NOW SHARP ENOUGH TO BELIEVE IT.
  Judged with eval_cpp/eval_paired (paired s.d. 108 m, resolves 2.5%): v113 cascade exposure
  NEGATIVE; v114 device-calibrated obstacle mix PARITY; v115 airborne-slide-survivable -9.7%;
  v116 faithful base + 3 edge-case drills -2.1% at t=-2.41, RESOLVED WORSE (its @8M ckpt is -0.5%,
  not resolved). v111 REMAINS CHAMPION.**
  - **⇒ GENERATION FIDELITY IS NO LONGER THE BINDING CONSTRAINT. Five consecutive nulls on an
    instrument that can now see 2.5% is real evidence, not absence of evidence. The 2026-07-26
    strategic conclusion ("the problem is the simulation, not the training") has been WORKED OUT —
    the mix, the pair statistics and the airborne-slide death rule are all device-calibrated now.**
  - **⚠️ SOFTENING THE WORLD COSTS PERFORMANCE (v115, -9.7%). Making a lethal situation survivable
    without adding TARGETED practice removes pressure and teaches nothing. And adding the drills
    (v116) did not recover it. USER's framing is the right one and is now tested: a faithful base
    AND specific drills — but the drills as built are not the lever.**
- **★★★★★ THE C++ EVALUATOR CUT PAIRED VARIANCE 11x — THE MEASUREMENT PROBLEM IS SOLVED.
  `scripts/validate/eval_cpp.py` scores both policies in the C++ trainer with MATCHED env seeds; paired s.d.
  falls from 1237 m (torch evaluator) to 108 m. At 8 seeds that resolves a 2.5% effect instead of
  ~30%, in ~3 minutes. This is the common-random-numbers win that the abandoned track-injection
  attempt was chasing — it came for free from evaluating where the policies train.
  ⇒ JUDGE EVERY ARM WITH `scripts/validate/eval_paired.py` (which now calls eval_cpp), 8+ seeds.**
  - **CHECKPOINT RULE, fixed in advance to stop post-hoc cherry-picking: take the checkpoint with
    the best training `ep_dist`, ties to the LATER one. v113/v114/v115 all peaked at 8M and dipped
    at 10M; v116 rose monotonically so its 10M final is the pick.**
- **★★★★★ EVERY SIM VERDICT WAS MEASURED IN THE WRONG SIMULATOR. `scripts/validate/eval_ckpt_torch.py`
  evaluates in `TorchTempleRunVecEnv` — the RETIRED torch sim, which is NOT gated and silently
  lacks anything added to the C++/CPU sims since. It has ZERO implementation of
  `airborne_slide_kill_prob`, so v115 (trained WITH that death rule) was judged in a world WITHOUT
  it. v113, v114 and v115 were all called on measurements taken somewhere the policies never
  trained. ⇒ USE `scripts/validate/eval_cpp.py` (NEW): same output format, runs the C++ TRAINER.**
  - **⚠️ AND THE TRAINER'S CONFIG IS NOT JUST `--sim-config`. train_fast.py builds a BASE cfg first
    — `domain_rand`, `turn_approach_buffer`, `turn_linger_frames`, `curriculum_max_start_dist` —
    then merges the JSON on top. An evaluator that passes only the JSON runs a DIFFERENT WORLD:
    without `turn_approach_buffer` the pre-swipe never latches and v111 dies at the FIRST junction,
    scoring 109 m instead of ~2400. eval_cpp.py mirrors that base cfg; keep them in step.**
  - **eval_cpp.py takes ONE EPISODE PER ENV on purpose: with auto-reset, harvesting "the first N
    finished episodes" oversamples SHORT ones and biases the median down. num_envs == episodes.**
- **★★★★★ A SINGLE-SEED n=128 SIM EVAL IS WORTHLESS FOR RANKING — MEASURED, AND IT INVALIDATES THE
  WAY EVERY ARM THIS WEEK WAS JUDGED. The SAME model (v111) scored 2188 / 2383 / 3222 / 3565 / 3644
  across five eval seeds = a spread of 49% OF ITS OWN MEAN. So two EQUAL policies can read anywhere
  from -37% to +68% on one seed, and they did: v114 read +29.6% on seed 0 and -28.9% on seed 1.
  Over five seeds the honest answer was +2.7%, t=0.14 — INDISTINGUISHABLE.**
  - **⇒ USE `scripts/validate/eval_paired.py` (NEW). Statistic = MEAN OF PER-SEED MEDIANS with a s.e. across
    seeds, plus a paired t against the baseline. Paired s.d. is ~1240 m on a ~3000 m mean, so
    seeds ~= (1240 / (0.5 * effect * 3000))^2:  50% -> 3 seeds (~6 min), 30% -> 8 (~16 min),
    20% -> 17 (~35 min), 10% -> 68 (~2.3 h). 35 min to resolve 20% is nothing next to a 10M arm.**
  - **⇒ THE "+-20% NOISE FLOOR AT n=128" NOTE ELSEWHERE IN THIS FILE UNDERSTATES IT: that was one
    model over five seeds of the same config; the RANKING error between two models is ~+-30-40% on
    a single seed. v113's "+16%" and v114's "+29.6%" were both inside pure noise.**
  - **⚠️ THE REAL FIX IS COMMON RANDOM NUMBERS (score every policy on IDENTICAL injected tracks —
    track luck then cancels and a handful of seeds suffices). ATTEMPTED AND ABANDONED 2026-07-28:
    `sim_probe_track` (cpp_sim/sim.cpp) generates cells correctly but its `distance_real` will NOT
    accumulate (`s->distance_real += step` reads back as ~5 m after 4000 cells, cause unexplained),
    so every generated track stays in the early ramp and slides never switch on. Seeding both
    policies alike is NOT a substitute — generation and the event rolls share one RNG stream, so a
    different action sequence shifts the draws and the worlds diverge. Worth another attempt: it
    would cut evaluation cost ~10x.**
- **★★★ v114 = THE OBSTACLE MIX IS NOW DEVICE-CALIBRATED (keep the WORLD, the POLICY is parity).
  `experiments/_v114_cfg.json`, weights {1:0.2872, 2:0.1349, 3:0.1009, 4:0.0484, 5:0.1681, 6:0.2605},
  fitted as weight *= device/emitted from REAL ROLLOUTS (weights are NOT the emitted share — holes
  also arrive via wide-gap/compound paths that bypass the type draw).**
  - **FIDELITY, measured on the device's own basis (structure units, types 1-6), TOTAL ABSOLUTE
    ERROR 18.8 -> 7.6 points: Narrow 8.9 -> 14.2 (dev 14.7), Stumble 14.6 -> 10.1 (dev 11.0),
    Flame 18.4 -> 17.5 (dev 17.7), TreeSlide 11.7 -> 12.7 (dev 12.0), Hole 38.3 -> 36.3 (dev 33.1),
    TreeJump 8.2 -> 9.3 (dev 11.4). This is device-calibrated, not fitted to an outcome — KEEP IT
    as the training world.**
  - **POLICY = PARITY, and the first read was NOISE: seed 0 gave v114@8M 4177 vs v111 3222 (+29.6%,
    outside the +-20% floor and tempting); seed 1 gave 2535 vs 3565 (-28.9%). POOLED over n=256:
    v114 3356 vs v111 3394 = dead even. ⇒ ONE n=128 POINT AT ~1.5 SIGMA IS NOT A RESULT. Always run
    the second seed before crowning; it cost 3 minutes and prevented a false champion.**
  - **DEVICE, run 2026-07-28 (hook 17k, 37C, `captures/v114_device.jsonl`): v114@8M n=6 =
    4258/2461/3584/2262/3643/3211, MEDIAN 3398, mean 3236. Same-day v111 reference (n=3, 36C,
    `D:\Robotique\Temple-run RL\captures\video\run.jsonl`): 3094/2915/4294, median 3094.
    +9.8% on the median = NOT RESOLVABLE at n=6 vs n=3.**
  - **PER-ENCOUNTER (the higher-power read, and it is also flat): v114 hole 3/96 = 3.12%,
    narrow 0/54, flame 2/65 = 3.08%, crevasse 0/37. v111 hole 0/46, narrow 0/21, flame 1/40 = 2.50%,
    crevasse 0/37. Nothing separates: 3-vs-0 hole deaths on 96-vs-46 encounters is p~0.3.**
  - **⚠️ THE ONE DIRECTIONAL HINT, worth a bigger batch and NOT worth acting on yet: v114 died
    3-of-6 on HOLES where v111 died 0-of-3, and HOLES are also the largest surviving mix error
    (36.3% vs the device's 33.1%, 1.10x over, after the correction cut it from 1.16x). The narrow
    increase produced ZERO narrow deaths in 54 encounters, so the extra narrow exposure cost
    nothing. NEXT MIX PASS: push holes down and TreeJump up (9.3% vs 11.4%).**
  - **v111 REMAINS CHAMPION on accumulated evidence. v114 is not better and not worse on EITHER
    sim or device; its value is the WORLD. Ranking these two properly needs ~48 eps/arm (~2.5 h
    phone time each) — do not attempt it with 6.**
- **⚠️ `sim_probe_cells` (new C++ export) CANNOT measure anything DISTANCE-GATED: `distance_real`
  only advances when the sim is STEPPED, so a bare generator loop always sees 0 and never reaches
  the late ramp (slides stay off, mix reads Hole-heavy). Use a POLICY ROLLOUT
  (`experiments/sim_patterns.py`) for any mix/pair calibration.**
- **⚠️⚠️⚠️ THE PAIR-STATISTIC TARGETS IN THIS FILE ARE ON AN UNSTATED BASIS AND MUST NOT BE USED
  UNTIL IT IS PINNED DOWN. Re-measured 2026-07-28 on a STRICTLY MATCHED basis (obstacle types 1-6,
  STRUCTURE units, ALL consecutive pairs, 15 device captures / 3316 structures / 3315 pairs) —
  ie. the identical statistic computed the identical way on both sides:**
  ```
                       DEVICE      sim v111     sim + rejection "fix"
    slide-led           29.7%       31.1%            31.8%
    slide-repeat        10.3%        7.9%             8.0%
    Hole->Hole          13.1%        7.4%             6.2%
  ```
  - **⇒ THE 50.7%-vs-39.5% SLIDE-LED GAP DOES NOT REPRODUCE. Slide-led is ALREADY MATCHED (31.1 vs
    29.7). The 50.7% is defined over 3-OBSTACLE CHAINS, not pairs, so it was never comparable to a
    pair statistic. The whole "the sim under-serves slide-led cascades" premise rests on that
    mismatch.**
  - **⇒ AND THE HOLE DIRECTION IS INVERTED: the device repeats holes ~1.8x MORE than the sim
    (13.1% vs 7.4%), not less. So `pair_anti_repeat_prob` and the `type_reject` hole entry are
    pointed THE WRONG WAY — tuning them to the "6.1%" target moved Hole->Hole further FROM the
    device (7.4% -> 6.2%). The "Hole->Hole 6.1% vs independent 10.3%" figure in
    `_sample_obstacle_type_paired`'s comment is on yet another basis.**
  - **⇒ THE ONE GAP THAT SURVIVES A MATCHED COMPARISON IS THE EMITTED MIX (device vs sim weights):
    TreeJump 11.4% vs 3.4%, Narrow 14.7% vs 8.0%, Flame 17.7% vs 26.6%, TreeSlide 12.0% vs 16.1%,
    Hole 33.1% vs 32.7%, Stumble 11.0% vs 13.1%. Fix THAT before touching any pair rule.**
  - **⇒ RULE: a pair/chain statistic is meaningless without its BASIS (structure vs cell; joint vs
    conditional; pairs vs chains; which types are counted). State the basis next to every number, and
    compute BOTH sides with the SAME code before comparing.**
- **⚠️⚠️ HARNESS TRAP THAT COST FOUR FALSE RESULTS IN ONE SESSION: `gym.make('TempleRunSim-v0', ...)`
  takes `config=`, NOT `sim_config=`, and the C++ `CppTempleRunVecEnv` likewise takes `config=`. An
  unknown kwarg is SWALLOWED, so the env silently runs DEFAULTS and every dose looks identical. This
  produced a false "pair_same_type_prob is inert" and a false "_cells_since_hazard gate bug" — both
  RETRACTED; under the real config the knob moves slide repeats 13.5% -> 31.6% and the counter sits
  at 0-6, well inside the <=4 gate. Cell dicts also key the obstacle as `obstacles` (PLURAL).
  ⇒ EVERY generation probe MUST assert `env.cfg[k] == v` after construction before it measures.**
- **★★★ NEGATIVE RESULTS — do not repeat these:**
  - **THE JUMP-INTO-TURN PENALTY CANNOT WORK AS A REWARD FIX — THERE IS NO JUNCTION CUE AT TAKEOFF.
    `--turn-takeoff-penalty` is BUILT and gated GREEN (6864/6864, variant `turntakeoff_pen`) but is
    DEFAULT-OFF and was never trained, because the pre-training measurement kills it. Over 2064
    device takeoffs pooled across 13 captures: only 10 (0.5%) have a junction in the jump arc — NOT
    the 5.0% this file claimed — and 8 of those 10 (80%) are jumps DEMANDED by a must_jump hazard.
    An ungated penalty therefore prices mostly NECESSARY jumps (the slide_jump_penalty
    conflicting-gradient trap); a gated one fires on 2 events in 2064.**
    - **ROOT CAUSE, and it retires the whole reward direction: `next_is_turn` fires 0/2064 on
      device — the documented junction obs dropout — and `dist_to_turn` streams 750..1000 with a
      FLOOR of 750, so it is not metres-to-junction and is unusable. The policy has NO
      junction-ahead signal at the moment it commits to a takeoff, so the counterfactual is not
      learnable no matter how it is priced. ⇒ THE ONLY REMAINING ROUTE IS PERCEPTION: make the
      landing prediction FOLLOW THE TURN (the hook's turn-aware walk already does this for the
      level fields). ⚠️ Budget it against the horizon-obs precedent, which was also "obviously
      needed" and turned out not to be learnable.**
    - **⇒ RULE THIS CONFIRMS: before pricing a cue, check the cue EXISTS in the stream and that the
      priced event is DISCRETIONARY. Both checks are ~20 lines and cost minutes; the arm they
      cancelled would have cost 10M steps and produced an unattributable result.**
  - **EPISODE-PROFILE OVERRIDES ACCEPT ONLY 11 KEYS (`sim_profile_key_id` in cpp_sim/sim.cpp):
    obstacle_prob, turn_prob, turn_prob_late, min_space_between_turns(_late),
    narrows_after_distance, zone_after_distance, zone_change_every_cells,
    min_space_between_obstacles, compound_prob, wide_gap_prob. NOT obstacle_type_weights and NOT
    pair_same_type_prob — so a SLIDE-WEIGHTED cascade profile is impossible without a C++ change.
    This is why the 50.7%-slide-led (device) vs 39.5% (sim) chain skew is STILL OPEN.**
  - **THE HORIZON OBS (obs 65→67, `dist_next_must_jump/slide` over 6 cells) IS BUILT AND GATED BUT
    NOT LEARNABLE in this setup. Three arms (10M @5e-5, 10M @2e-4, 40M consolidation) all left the
    cue flat: among states where the +3 booleans are ALL False (the only place it carries
    information) P(JUMP) at d=4/5/6 was 0.0004/0.0004/0.0004 — d4−d6 = −0.00005 against a +0.0030
    bar. The policy learned "be twitchier in the dark", never "the hazard is 4 cells away".
    `obs.HORIZON_OBS = False` (reverted); the infra (obs, CPU, C++, hook 17k, surgery) stays — it is
    a one-line experiment when there is reason to believe the regime can develop zero-init columns.
    ⚠️ `HORIZON_OBS` in obs.py and `HORIZON_OBS` in cpp_sim/sim.cpp MUST be flipped together (the
    gate catches it instantly: shapes (37,) vs (39,)).**
  - **v113 = MORE CASCADE EXPOSURE DOES NOT FIX CASCADES (n=128 + 24-episode rollouts). v111's
    recipe + a dedicated cascade profile (w=10: obstacle_prob 0.85, min_space 3, compound 0.5;
    hole-chain profile cut 12->8). SIM MEDIAN parity — v113@8M 3140 vs v111 2700 = +16%, INSIDE the
    +-20% n=128 noise floor, so NOT resolvable (@10M 2290 = the usual end-of-schedule dip, keep @8M).
    ON THE TARGETED METRIC IT IS SLIGHTLY WORSE: chain death 8.2% -> 9.5% per encounter, at speed
    6.9% -> 9.3%. ⇒ CONFIRMS the skill-gap reading: the sim already presents cascades at the right
    rate and lethality, and MORE of the same teaches nothing.**
    - **⚠️ BUT THE ACTUAL HYPOTHESIS WAS NEVER TESTED. The device's chains are 50.7% SLIDE-LED vs
      the sim's 39.5%, and `Hole->Hole->Hole` is the sim's #1 chain (n=15/148) while being ABSENT
      from the device's top five. v113 could not change that because episode profiles cannot carry
      `obstacle_type_weights` (see the 11-key list below), so it only changed cascade FREQUENCY.
      NEXT, and it is device-grounded rather than fitted: the device's `Hole->Hole` sits BELOW its
      independent rate, i.e. the game ANTI-repeats jump obstacles, and the sim models no such rule
      (it only has `pair_same_type_prob`, a PRO-repetition rule for slides). A `pair_diff_type` rule
      for jump-type obstacles is the missing mechanic.**
  - **CASCADE GUIDED EXPLORATION DOES NOT TRANSFER (v112: 8.39% vs v111's 8.16%, median worse).
    The turn precedent does not apply: that was a NEVER-SAMPLED action (P(turn|approach) pinned at
    0.000000 through five arms); a cascade is a TIMING-PRECISION problem — the policy already jumps.
    `--cascade-explore-prob` + `sim_cascade_hint` stay in the code, default off.**
  - **n_epochs 20 = n_epochs 10** (median 1575 both, cue flat) at matched steps, for 2x wall-clock.
  - **v106 FAILED because I set `tight_pair_prob` 0.25 when this file already recorded the
    device-matching dose as ~0.08-0.10. It alone crushed v104 from 4728 → 672. Calibrated dose is
    0.05 (v104 sim median 3464 @0.04 / 3764 @0.08 vs its device 3249; 0.12 cliffs to 1358).**

- **⚠️⚠️ TWO MEASUREMENT BUGS FOUND TODAY — both had already corrupted conclusions:**
  - **THE EVAL HARNESS WAS CENSORED. `scripts/analysis/hole_landing_sim.py` passed `max_episode_steps`
    through `gym.make()`, which the registered env IGNORES → every distance capped at ~2607 m, so
    every world looked identical for hours and difficulty comparisons were meaningless. Fixed +
    asserted. ⇒ If two configs give the SAME median to the metre, suspect the harness.**
  - **WAVE-15's "device +2 = 0.7%" WAS AN ARTIFACT — and it had driven a real config change
    (`min_clear_after_turn` 2→3). `postturn_obstacles.py` used `obst.setdefault(cell, ...)` = FIRST
    frame wins, against WINDOW-PRESENTED device flags. Re-measured any-frame on the SAME captures:
    5.3-9.5%. Its "+3 = 21.7%" vs my 36.6% was simply MUST_SLIDE being excluded. Both capture eras
    agree at every definition level ⇒ methodology, not a change in the game.
    `experiments/postturn_reconcile.py` runs all four definitions side by side.**

- **★★ THE TWO REMAINING BOTTLENECKS (USER-identified, both measured):**
  1. **OBSTACLE CASCADES — "the only thing that really prevents it going far". 5 of v111's ~6 device
     deaths are inside a 3-obstacle chain: 9.62% per encounter on DEVICE, 8.16% in SIM (MATCHED —
     so NOT a fidelity or coverage gap), vs the HUMAN's 0 of 45. The sim presents cascades at the
     right rate AND lethality and the policy still cannot do them = a genuine SKILL gap.
     RESIDUAL SKEW WORTH FIXING: device cascades are 50.7% SLIDE-led vs the sim's 39.5%, and
     `Hole→Hole→Hole` is the sim's most common chain but is absent from the device's top five ⇒ the
     policy over-trains jump-land-jump and under-trains slide-recover-slide.**
  2. **★ JUMPING INTO A TURN — a STRUCTURAL defect, and the best lead. USER: "it jumped in a turn,
     turned and landed in a hole." MEASURED: 5.0% of takeoffs happen with a junction within 2 cells,
     and 8.7% of THOSE kill (vs ~1% for takeoffs generally). MECHANISM, visible in code:
     `landing_hazard` and the whole lookahead are computed ALONG THE CURRENT HEADING with an explicit
     junction dropout — so when the turn fires mid-air the runner lands on a path the observation
     NEVER MODELLED. FIX DIRECTION: either price a takeoff near an unresolved junction (exactly like
     landing_hazard_penalty, which just worked), or make the landing prediction follow the turn the
     way the hook's turn-aware walk already does for the level fields.**

- **NEW TRAINER KNOBS (all default-off, all in `scripts/train/train_fast.py`): `--backend torch|cpp`,
  `--landing-hazard-penalty`, `--stepdown-bonus-gate`, `--ent-coef-final` + `--ent-schedule`,
  `--gae-lambda-final` + `--gae-schedule`, `--cascade-explore-prob` + `--cascade-explore-anneal-frac`.**
  - **`--gae-lambda-final` exists because n_steps is the WRONG dial for a growing credit horizon:
    num_envs is fixed at construction and batch-size is capped at 512, so ramping n_steps changes
    num_envs*n_steps = updates per DATA COLLECTION (the invariant that gave 1036 vs 3262). Ramp
    lambda instead — episode length triples during a run (ep_len ~260 → 800+).**

- **NEW TOOLS: `scripts/analysis/chain3_forensic.py` (3-obstacle chains, STRUCTURE units),
  `pair_coverage.py` (what follows what, device vs sim), `calib_mix.py`, `calib_pairs.py`,
  `stepdown_forensic.py`, `stepdown_arc.py`, `stepdown_sim.py`, `postturn_reconcile.py`.**

- **HOOK 17k = 17j + `dist_next_must_jump/slide` (from the existing turn-aware 6-cell walk).
  BACKWARD-COMPATIBLE with obs65 policies: it only ADDS stream fields, and obs.py builds from an
  explicit field list, so v111 sees byte-identical inputs on 17j or 17k.**

## ⚡ PREVIOUS STATE (2026-07-27 LATE)
- **★★★★ TRAINING IS 2.2x FASTER, VALIDATED: 10M in ~16 min (was ~35). `train_fast.py` now defaults to
  `--num-envs 1024 --n-steps 4` instead of 256x16.**
  - **THE TRAINER IS LAUNCH-BOUND, NOT COMPUTE-BOUND. Measured: the sim step is 96.2% of wall-clock
    (63.7 ms vs 1.27 ms for a full PPO update), issues ~2000 tiny kernels/step, pegs exactly ONE CPU
    core (100% of 800%), and the GPU draws 59 W of a ~180 W TDP. ⚠️ `nvidia-smi utilization.gpu` is a
    LIAR here — it reports the fraction of time a kernel is resident, so it reads 60% while the card
    is ~10% busy (Task Manager's CUDA graph and the power draw are the honest numbers).**
  - **Step cost is therefore nearly FLAT in num_envs (70.7 / 68.1 / 70.9 / 77.9 ms at 256/512/1024/2048),
    so the win is making FEWER, WIDER step calls.**
  - **⚠️ KEEP num_envs * n_steps == 4096. Widening num_envs ALONE (1024x16) is 3x faster but 3x WORSE
    (@3M median 1036 vs 3262). Gradient-update COUNT is identical either way; what breaks is STALENESS —
    a 16384-sample rollout applies 128 sequential updates to ONE policy snapshot and PPO clips most of
    them away. The invariant is updates per DATA COLLECTION, not per sample.**
  - **VALIDATED at matched steps (n=128): @3M 3258 (1024x4) vs 3262 (256x16) = parity. @1M it lags
    (1666 vs 2546) — shorter GAE horizon learns value more slowly early — then catches up.**
  - **`torch.compile` IS PERMANENTLY UNAVAILABLE on this box: Triton does not support the GTX 1080
    (sm_61). Don't retry it. Dynamo also shows 53 graph breaks in the step, ALL from passing an
    explicit seeded `torch.Generator` to every RNG call.**
  - **NEXT SPEED LEVERS (untried): (1) batch the 107 per-step GPU->CPU syncs into ONE transfer — they
    are 31% of step time (`aten::_local_scalar_dense`, ~191us each) and come from `if bool(x.any())`
    guards; ⚠️ simply DELETING those guards is a REGRESSION (63.7 -> 84.2 ms) because the skipped work
    then always runs — batch the decisions instead. (2) CUDA-graph the step (works on Pascal, unlike
    Triton) — needs a sync-free step first, so (1) is a prerequisite.**
- **★★★★ THE MID->HIGH CREVASSE IS A SIM-TO-REAL TIMING GAP, NOT A TRAINING FAILURE. This reframing is
  the most important result of the day and it retires four failed arms.**
  - **TAKEOFF cp AT SPEED: the policy is ALREADY WELL-TIMED IN SIM (0.65-0.71, where the sim graze is
    0-3%) and only mistimes ON DEVICE (0.49-0.56, where the device graze is 20%). No sim-side training
    change can fix a gap that only exists on the phone — which is exactly why v93 (velocity-drop
    penalty), v94/v95 (predicted_jump obs law), v96 (graze jitter) and v97 (crevasse exposure) ALL
    FAILED. v97 was actively WORSE (graze 16.9%, slope -0.0059 resolved).**
  - **★★★ THE MEASURED CULPRIT (USER-challenged: "why add it if it does not even exist on the real
    game?"). `jump_exec_lat_jitter: 1` randomises the press->takeoff delay by +-1 frame. The GAME has
    zero jitter (probe_jump_exec 25/25 at exactly 1 frame); the +-1 was added on 2026-07-05 as an
    UNVERIFIED margin for TCP/obs-staleness variance and has been in every recipe since.
    NOW MEASURED END-TO-END from the captures (grounded press -> airborne, ~1400 presses, 4 policies):
    v92b 341/341, v90 392/392, v78 394/394, v88 313/314 ALL IN ONE BUCKET = ZERO JITTER, TCP INCLUDED.
    So the sim has been teaching the policy to hedge against a hazard the phone does not have, and
    hedging means TAKING OFF EARLY = the user's mid->high complaint. (Caveat: 10 Hz sampling resolves
    jitter only to 100 ms; sub-frame jitter would not show — but is far less harmful.)**
  - **v98 = v96 world + `jump_exec_lat_jitter` 1 -> 0. JUDGE ON: takeoff cp must RISE with speed
    (`scripts/analysis/takeoff_timing.py`, which now regresses on ALL crossings with a standard error).**
- **⚠️ TWO MEASUREMENT ERRORS OF MINE TODAY, both corrected — read before trusting any earlier entry:**
  - **THE SLOPE ESTIMATOR was a fit through 3 bucket medians whose top bucket held n~7. On that basis I
    reported v93/v94 "improvements" that DO NOT EXIST. Rebuilt to regress on every crossing with a
    standard error; it now says NOT RESOLVED unless |slope| > 2 s.e. Re-judged: human +0.00828+-0.00214
    (correct), v92b device -0.00638+-0.00137 (inverted), all sim arms -0.002..-0.004.**
  - **THE OBSTACLE MIX: see the note in sim_env.py. I counted `obstacle == 2` and concluded narrows were
    2.2x UNDER-served; real sim strips are `path_type` 17..30 with obstacles LEFT AT 0, so I counted only
    the FAKE (laneless, harmless) ones. Counting properly the sim emits ~41% narrows vs the device's 14%
    = ~3x OVER. My "fix" also could not change narrow frequency at all (the narrow branch fires BEFORE
    the weight table) — it only swapped real hazards for no-op cells, which regressed v94 (USER: "it
    strafes randomly to the left"; L/R strafe 0.75 -> 1.78 -> 2.74 across v90/v92b/v94). REVERTED, and
    `experiments/hazard_density.py` now normalises both encodings.**
- **DEPLOY STATUS: v92b (`runs/ppo_v92b_corrected_final.zip`) is the only model to ship today — it fixed
  the user's #1 complaint on DEVICE (held-jump 10.67 -> 3.64/km, user-confirmed). v94..v97 REJECTED.**

## ⚡ EARLIER 2026-07-27 — the held-jump fix and the fidelity bugs
- **★★★★★ THE USER'S #1 COMPLAINT IS FIXED, ON DEVICE, USER-CONFIRMED. v92b
  (`runs/ppo_v92b_corrected_final.zip`, obs65, 512-net, hook 17j). USER, live: "it's our best policy!
  The jump after a crevasse jump problem is now fixed!"**
  - **DEVICE-MEASURED (`scripts/analysis/detect_patterns.py`, 8 eps / 31 km): HELD_JUMP_SPAM 10.67/km (v90)
    -> 3.64/km (v92b) = 2.9x fewer, POST_CREVASSE_REJUMP 0% lethal. The sim predicted 3.55/km — the
    fix transferred almost exactly. Device distances 971/1814/1893/1930/3412/3593/7501/9243
    (median 2671, 34-39C) — but n=8 has ~11% power, so do NOT read the median as a ranking.**
  - **★ THE MECHANISM THAT WORKED — and it is a GENERAL LESSON. The held-jump DEATH MODEL alone did
    NOTHING (v91a/b/c: hold rate 105.8 -> 113.3 / 116.8 / 99.3 per 1000 cells = flat). The habit only
    broke when the death model was PAIRED with `--airborne-action-penalty 0.08`:
    v92a (no penalty) 11.53/km vs v92b (penalty) 3.55/km, and on HIGH GROUND 44 -> 8 events.**
  - **WHY: a JUMP pressed while airborne is a pure no-op in-sim apart from the rare landing re-jump,
    so the true gradient is ~0.1-0.2% death per press — real, correct, and far below what PPO at
    ent 0.01 will chase. A death model makes the world TRUTHFUL; an explicit penalty makes the
    truth LOUD ENOUGH TO LEARN. Neither works alone: v58/v60/v61's penalties failed because the
    world never punished the behaviour (so they were gamed by 1-frame gaps), and v91's death model
    failed because the signal was too weak to find. ⇒ **WEAK-SIGNAL RULE: when a device-true
    mechanism is faithfully modelled and the policy still ignores it, check the SIZE of the gradient
    before concluding the model is wrong.** The penalty cost NOTHING: v92b beat v92a on all 3 eval
    seeds (3505/3660/3695 vs 2994/2496/3399, ~+22%).**
- **★★★ THREE MEASUREMENT/FIDELITY BUGS FOUND AND FIXED TODAY (details in the sections below):
  (1) the sim's METRE was ~1.4x too short at speed; (2) the JUMP-SPAN law used by landing_hazard and
  the landing death rules was over-long by 13-17%; (3) the OBSTACLE MIX was badly wrong (Narrow 0.08
  vs device 0.173). All three were found by expressing a metric in a unit both sides define
  identically, and all three were corrected from measurement with NO fitting.**
- **★★★★★ THE MID->HIGH CREVASSE — ROOT CAUSE FOUND, AND IT IS THE `predicted_jump` OBSERVATION
  ITSELF. This is why weeks of reward shaping, death models and geometry knobs never fixed it: the
  policy was being TOLD to jump earlier the faster it runs.**
  - **`obs.py predicted_jump_obs` returns `raw = (LEDGE_CELLS_AHEAD - cp) - 0.5*span` in CELLS. The
    comment claims "velocity-invariant". IT IS NOT. Since `raw = span*(frac - 0.5)`, a FIXED learned
    threshold r means `frac = 0.5 + r/span` — so as speed (span) grows, the same obs value sits ever
    CLOSER TO 0.5, the graze boundary. The safety margin SHRINKS exactly when the arc is longest.**
  - **IT PREDICTS THE DEFECT QUANTITATIVELY. Inverting the mapping, the chosen takeoff is
    `cp = LEDGE_CELLS_AHEAD - 0.5*span - r`, which FALLS ~0.4 from v150 to v210. MEASURED on device:
    the policy's takeoff cp falls 0.81 -> 0.49 (slope -0.0054) while the HUMAN RISES 0.53 -> 0.85
    (+0.0054). The observation was teaching the inversion.**
  - **FIXED: `PJ_VELOCITY_INVARIANT = True` emits the DIMENSIONLESS arc fraction (0.5 = apex exactly
    over the ledge, below = still ascending = safe), so one learned threshold means one physical
    margin at every speed. Both sims, CPU<->torch parity EXACT (0.00e+00). Set the flag False to
    reproduce a pre-2026-07-27 model's inputs — v78..v93 ALL trained on the old law, so pin it (and
    `--land-span 0.0204,-0.94`) when evaluating or deploying them.**
  - **v93 CONFIRMS the reward cannot fix it (10M each, crevasse-stress cfg, 72-86 level-3 caps):
    v93a (velocity-drop 0.01) slope -0.0016 -> +0.0000 FLAT, graze 19.7% -> 18.1%; v93b (0.03) slope
    -0.0035, graze 27.9% = WORSE. Pricing the hit was NECESSARY (the user was right that it was
    unpriced) but is NOT SUFFICIENT while the cue points the wrong way. Keep the low dose, drop 0.03.**
  - **v94 IN FLIGHT = v93a's world + the corrected obs law (the only change). Judge with
    `scripts/analysis/takeoff_timing.py`: the slope MUST turn POSITIVE.**
- **★★★ THE MID->HIGH CREVASSE — the USER DIAGNOSED THE REWARD HALF CORRECTLY.**
  - **USER: "The mid to high transition is not fixed. The policy jumps way too early... Is it because
    it has no penalty to do so in the sim? Have you modelled the case when the character takes a hit
    and is slowed down? This should be penalized. It is clearly not." EXACTLY RIGHT.**
  - **`reward.py` HAS a `velocity_drop_penalty` — written during the crevasse wave, commented "docks
    the too-early crevasse jump that clips the raised wall + gets slowed (user-reported)" — and BOTH
    reward.py AND the torch `_shaping` implement it. It was NEVER PLUMBED INTO `train_fast.py`, so
    NO recipe has ever used it. The sim models the wall-clip (Stumble x0.85) and already grazes
    HARDER than the device, so the physics was never the gap: a graze costs a few units of distance,
    i.e. the SAME weak-signal failure as the held jump. NOW WIRED (`--velocity-drop-penalty`).**
  - **NEW DETECTOR `scripts/analysis/takeoff_timing.py` — names the defect as a SLOPE, which is the whole
    story (a jump spans 0.0143*v cells, so takeoff must be DELAYED as speed rises to still be near
    apex over a raised cap):**
    ```
                 takeoff cp by speed          slope d(cp)/dv   lvl-3 graze   velocity bled
    v92b device  0.81 -> 0.56 -> 0.49            -0.0054           16.3%        5.5 /km
    HUMAN        0.53 -> 0.74 -> 0.85            +0.0054            0.0%        0.0 /km
    ```
    **Equal and opposite. The policy takes off EARLIER the faster it runs — the exact inversion.**
- **★★★ THE NARROW REGRESSION IS REAL AND THE USER SPOTTED THE RIGHT BASELINE ("v78 didn't die on
  narrows like v92"). `scripts/analysis/narrow_forensic.py`, per-ENCOUNTER (high power):**
  ```
                 narrow deaths/encounter     strafe reversals off-narrow
  v78 (pooled)        2.47%  (4/162)                 ~0.1 /km
  v90                 7.62%  (8/105)                  0.7 /km
  v92b                8.00%  (6/75)                   1.1 /km
  HUMAN               0.95%  (1/105)                  0.0 /km
  ```
  **v78 -> v90/v92 is a 3.1x regression, p=0.028 — it arrived with the v90 GENERATION, not with
  v92b's recipe. ROOT CAUSE FOUND: the sim's obstacle MIX, not the recipe (see below).**
- **★★★★ THE OBSTACLE MIX WAS WRONG — and this is why a density-only audit missed it. Pooled over
  28,600 device cells / 4112 structures (6 capture sets), reproduced independently by the 41 km human
  set to within ~0.02 on every entry:**
  ```
  non-crevasse share   sim(old)   DEVICE      verdict
  Narrow                0.08      0.173     2.2x UNDER   <- the narrow weakness, directly
  TreeJump              0.22      0.103     2.1x OVER
  TreeSlide             0.22      0.110     2.0x OVER
  Hole                  0.22      0.327
  FlameTower            0.14      0.186
  ```
  **TOTAL hazards/km was already CORRECT (16.4 sim vs 15.2 device) — the MIX is a separate axis from
  the DENSITY, and `sim_real_audit.py` only ever checked density and spacing. The policy was getting
  HALF the strip practice it needed and DOUBLE the tree practice. CORRECTED (device target scaled once
  by target/observed, since holes also arrive via wide-gap/compound paths): now every hazard lands
  within 3 points of the device composition. Also fixed: a JSON `--sim-config` passes STRING keys for
  this dict, which mixed with the int defaults and crashed the torch env — both merges now coerce.**
- **THERMAL PROTOCOL RELAXED (USER 2026-07-27: "You can safely play the game on the phone. The current
  threshold is very restrictive. It's summer here"). Do NOT block a batch waiting for <=36.0C; record
  the temperature alongside the result instead. v92b's batch ran 34->39C.**
- **IN FLIGHT (10M each, ~12:45): v93a / v93b = v92b's recipe + `--velocity-drop-penalty` 0.01 / 0.03
  + the corrected obstacle mix + narrow-heavy episode profile 6% -> 17% (standard anchor held at 45%
  per the anti-drift lesson). JUDGE ON: (1) takeoff slope must turn POSITIVE (`takeoff_timing.py`);
  (2) narrow deaths/encounter must fall toward v78's 2.47% (`narrow_forensic.py`); (3) HELD_JUMP_SPAM
  must STAY low (~3.5/km) — do not trade the fixed complaint for the new one.**

## ⚡ EARLIER 2026-07-27 — the fidelity bugs in detail
- **★★★★★ THE SIM'S METRE WAS ~1.4x TOO SHORT AT SPEED. This is the single biggest fidelity error
  found in the project, it is FIXED, and it silently corrupted ~20 models' worth of sim-vs-device
  verdicts. `dist_per_velsec` was a FLAT 0.107; the real game accrues DistanceTraveled as
  ~8.47e-4 * velocity^2.**
  - **MEASURED, not fitted: pooled over 154 capture files / 770k frame pairs, `dist_per_velsec/v` is
    8.56, 8.46, 8.47, 8.48 e-4 across v105..v215 — one constant, no freedom. DEVICE and HUMAN captures
    (independent pipelines, 10 Hz vs 30 Hz) agree within 5% in EVERY velocity band:**
    ```
    v-band     80-119  120-159  160-199  200-239
    device     0.0970   0.1317   0.1616   0.1753
    human      0.0934   0.1242   0.1550   0.1751
    sim (old)  0.1062   0.1069   0.1070   0.1070   <-- flat; only correct near v=136
    ```
  - **HOW IT HID: the flat constant is RIGHT at low speed, so every early-run check passed. A run
    spends most of its life at v170-210, where the sim under-reports distance ~1.5x. Consequence: the
    sim needs ~1.39x MORE CELLS to report the same distance as the phone (cells/km 157 vs 113).**
  - **THE THREE "PROBLEMS" IT MANUFACTURED, all one bug:** (1) the chronic "sim is 1.5x too easy"
    (and this week's opposite "1.8x too hard"); (2) "hazard density 1.44x over-packed" — hazards per
    CELL were CORRECT ALL ALONG (0.152 sim vs 0.146 device = 1.04), only hazards per *metre* were off;
    (3) the distance-keyed difficulty ramp firing ~1.4x too late in cell terms, i.e. the sim applied
    the game's late-game tight spacing far too late. **DO NOT retune `obstacle_prob` — generation is
    right.**
  - **THE CLINCHER (how to localise this class of bug): TIME per cell already matched EXACTLY
    (400 ms @v120-159, 300 ms @v160-199, in sim AND device AND human) and velocity-vs-distance matched
    to ~1% — so the mechanics were faithful and only the conversion was wrong. A per-CELL metric that
    agrees while the per-METRE metric disagrees localises the bug to the unit, not the physics.**
  - **SHIPPED: `dist_per_velsec_slope` (both sims, default 0.0 = old flat behaviour; set 8.47e-4 for
    the measured law). The replay path's flat `cell_length_real` 6.4 is corrected the same way
    (device: 5.4 @v100 -> 10.2 @v215 = slope*cell_length*v).**
  - **★ RESULT — THE BEST SIM-TO-REAL DIFFICULTY ALIGNMENT EVER MEASURED, WITH ZERO FITTING:
    v90 sim median 2908 -> 4770 against its MEASURED DEVICE median of 5116 = a 7% gap (was ~1.8x).
    Verified end-to-end: cell length 8.7 vs device 8.8, hazards/km 15.5 vs 16.6, dist-rate within
    2-4% in every band.**
- **★★★ THE HELD-JUMP CALIBRATION IS SETTLED — AND YESTERDAY'S BLOCKER WAS NOISE. The 2026-07-26 note
  "do NOT train until the sim median lands near the device median" is WITHDRAWN, on evidence:**
  - **the n=64 sweep that motivated it (6224 -> 2153 at the raw device rate) does NOT reproduce. At
    n=128 in the same world the response is smooth, monotone and MILD: 0.00 -> 2908 | 0.02/0.06 ->
    2843 | 0.04/0.11 -> 2782 | 0.07/0.20 -> 2634 | 0.10/0.28 -> 2642 (= -9% at the RAW DEVICE RATE).
    The "collapse" was the exact failure this file's own eval-power correction warns about.**
  - **the knob is BEHAVIOUR-CONDITIONAL, unlike a generation knob — verified selective (a HOLDING
    policy gets unwanted re-jumps, a TAPPING policy is untouched). "The sim median drops" is the cost
    of a habit we want removed, not evidence of an over-harsh world. So it is set to the DEVICE-
    MEASURED 0.10 solid / 0.28 gap, like the difficulty ramp, and is NOT a free parameter.**
  - **FIDELITY CHECK PASSED: the sim now reproduces the device's held-jump LETHALITY — 2% of
    HELD_JUMP_SPAM events precede a death in-sim vs 3% on device.**
- **★★ THE SIM-SIDE HOLD RATE (the measurement yesterday asked for) — and a CORRECTION to my own first
  answer. Measured per KM it looked like a 1.54x inflation (sim 16.5 vs device 10.7). That was the
  distance-unit bug again: per km is not comparable when the sim's km is 1.39x too short. PER 1000
  CELLS (scale-free) the sim and the device AGREE: sim 105.8 vs device 94.5 = 1.12x. HUMAN 0.0.
  So there was never an event-rate discrepancy to compensate for, which retires the reasoning behind
  the v91b "event-matched" dose and leaves the device-measured rate as the only defensible one.
  ⚠️ RULE: never compare a per-km rate across a change in the distance scale — use per 1000 cells.**
- **★★★ v91 VERDICT (all three arms, 7M each): THE HELD-JUMP DEATH MODEL IS FAITHFUL BUT DOES NOT
  TEACH RELEASE. HELD_JUMP_SPAM per 1000 cells: v90 105.8 -> v91a 113.3 / v91b 116.8 / v91c 99.3,
  i.e. FLAT (the human is 0.0). The indicator was flat at 2M (17.4/km), 4M (17.56) and 7M — read
  early, exactly as the wave-13 discipline requires. Performance: in v91's own training world
  v91a 2889 vs v90 2642, which is INSIDE the noise floor (see below) = no cost, no gain, at half the
  steps. v91b (lower dose) 2291 = worst; the device-true dose wins.**
  - **WHY IT FAILS, honestly: a JUMP pressed while airborne is a pure no-op in-sim except for the rare
    landing re-jump, so the gradient on that action is ~0.1-0.2% death per press — far below what PPO
    at ent 0.01 will chase. This is NOT the wave-13 exploration deadlock (releasing IS sampled
    constantly); it is a WEAK-SIGNAL problem. More dose is not the answer: the rate is device-measured.**
  - **NEXT (v92b): pair the death model with `--airborne-action-penalty` as a GRADIENT AMPLIFIER. The
    2026-07-08 retirement of that penalty ("airborne presses are dropped on device, so it solves a
    non-problem") is REFUTED by the wave-12 landing-lock forensic: the press is dropped while airborne
    but fires a fresh jump ON TOUCHDOWN. Crucially this is NOT a repeat of v58/v60/v61 — those taxed a
    behaviour the world never punished, so the policy gamed them at zero cost; now the world punishes
    it truthfully and the penalty only amplifies a gradient that is genuinely there. Keep the
    coefficient modest (0.08): on the LANDING frame prev.is_jumping is still true, so it also docks the
    DELIBERATE landing-frame re-jump that post_land_gap_rejump models as a real gap-chain affordance.**
- **★★★ SIM EVAL NOISE FLOOR, MEASURED (do not skip — it invalidates small verdicts): the SAME model in
  the SAME config at n=128 over five seeds gives 2098 / 2642 / 2664 / 2679 / 2728 = ~+-10% at 1 s.d.
  with outliers to -20%. **DIFFERENCES UNDER ~20% AT n=128 ARE NOT RESOLVABLE.** This is the sim-side
  twin of the device-side power correction, and it retires today's own n=64 dose sweep as noise.
  PREFER RATE METRICS: HELD_JUMP_SPAM per 1000 cells and per-encounter death rates count hundreds of
  events per rollout and separate cleanly where medians cannot.**
- **THREE ARMS IN FLIGHT (7M each, compressed cosine 8e-5 -> 5e-6, consolidation from
  `runs/ppo_v66_obs65_land.zip` = v90's own base, NOT a finetune off v90b):**
  - **v91a `experiments/train_v91a_heldjump.sh` — v90's exact recipe + held-jump at the DEVICE rate.**
  - **v91b `train_v91b_matched.sh` — the EVENT-matched dose 0.065/0.18 (= device rate x 10.7/16.5), the
    control that separates "reproduce the device MECHANISM" from "reproduce the device EVENT RATE".**
  - **v91c `train_v91c_distfix.sh` — v91a + the distance-unit fix (so also the corrected ramp timing).**
  - **⚠️ LEADING INDICATOR AT 2M: v91a's sim HELD_JUMP_SPAM is 17.4/km = NOT falling yet (v90 16.5).
    Expected-weak: holding is only punished when the re-jump both fires (10%) and lands somewhere
    fatal (~2%), so the per-press gradient is ~0.1-0.2% death. Re-read at 4M/7M. If it stays flat, the
    honest read is that the death model alone is too weak a signal for an action that is otherwise a
    pure no-op in-sim, and the next lever is the wave-13 pattern (make the counterfactual SAMPLED),
    NOT more reward shaping (v58/v60/v61 all failed/were gamed).**
- **JUDGE v91 ON: (1) sim HELD_JUMP_SPAM per km (must fall from 16.5 toward the human's 0) and its
  lethality; (2) per-ENCOUNTER death rate (`scripts/analysis/device_compare.py`, n=400-800 encounters/batch);
  NOT the 8-episode device median (11% power to see a 20% gain).**
- **NEW TOOL: `experiments/hazard_density.py` — hazards per km AND per cell, banded by distance. This
  is the leg the wave-15 audit missed: that audit matched the spacing SHAPE (share of pairs inside the
  jump arc), which is CONDITIONAL on a pair existing and says nothing about how many there are.**

## ⚡ PREVIOUS STATE (2026-07-26 end of day)
- **★★★★ THE USER'S #1 COMPLAINT IS SOLVED AT THE MECHANISM LEVEL (built, gated, NOT yet trained).**
  - **USER, ~50 times: "on the high-ground it jumps for no reason just after having jumped over a
    crevasse". Also: "how do you analyze the logs to detect them? Can't you build a script?"**
  - **`scripts/analysis/detect_patterns.py` (NEW — build named-pattern detectors, not just aggregates. Every
    prior analysis was an AGGREGATE, which averages away a specific recurring mistake: a pattern that
    fires 6x/run and kills once is invisible at 1.5%/encounter but is what the user sees EVERY run.)**
  - **THE ANSWER (v90 device 37 km vs HUMAN 41 km): `HELD_JUMP_SPAM` = the policy holds JUMP through
    its whole arc **10.7 times per km**; the HUMAN does it **ZERO times in 41 km** — they TAP.
    In the sim a JUMP pressed while airborne is DISCARDED by the airborne guard, so holding is FREE
    and nothing ever teaches release. On DEVICE the still-held press fires a NEW jump the instant the
    runner lands (wave-12: 10% solid / 28-46% gap) — on high ground that second jump drops into the
    next crevasse/hole. THE USER'S REPORT, EXACTLY. It is a SIM-FIDELITY gap, NOT reward shaping —
    which is why v58/v60/v61's penalties failed (they taxed a behaviour the world never punished and
    were gamed by 1-frame gaps).**
  - **SHIPPED IN BOTH SIMS: `held_jump_rejump_solid` / `held_jump_rejump_gap` (default 0 = old
    behaviour). A JUMP pressed while airborne is remembered; any other action releases it; on
    touchdown it fires a fresh jump with the context-dependent probability. Equivalence GREEN
    (312/312 with the model at prob 1.0, plus 1248/1248 on the core variants). VERIFIED SELECTIVE:
    a HOLDING policy gets unwanted re-jumps, a TAPPING policy is completely unaffected — exactly what
    the old reward penalties could not achieve.**
  - **⚠️ CALIBRATION IS UNFINISHED — DO THIS FIRST TOMORROW. At the raw device rates (0.10/0.28) the
    model is far too harsh: v90's sim median 6224 -> 2153, against its MEASURED device median of 5116.
    Dose-response (v90, n=64, tomorrow's full cfg): solid/gap 0.02/0.06 -> 3799 | 0.04/0.11 -> 4761 |
    0.07/0.20 -> 4383. ~0.04/0.11 lands nearest 5116, but the curve is NON-MONOTONE at n=64 (noise),
    so re-measure with n=128 before fixing the value. Likely cause of the gap between the device rate
    and the sim outcome: the policy holds the button MORE in-sim (no feedback) than on device, so the
    same per-landing probability converts into more events — worth measuring directly with
    detect_patterns.py on a SIM rollout before trusting either number.**
- **★★ v90 = THE LANDING-ARC OBSERVATION (obs 64->65, `landing_hazard`). DEVICE: 1934/2977/4453/5041/
  5191/5249/5510/6677 = MEDIAN 5116, mean 4629, zero sub-1000 — the BEST median measured this week
  (v78 4424) but NOT separable at n=8 (see the protocol correction below). Weights
  `runs/ppo_v90b_final_final.zip` (11M + a 2M annealing tail at LR 3.66e-5 -> 5e-6).**
  - Built from the behavioural death budget (`scripts/analysis/death_behaviour.py`, 44 deaths): 55% COMPOUND
    TRAP + 5% RANDOM-JUMP TRAP = **60% of deaths are "my arc landed me somewhere fatal"**, with the
    takeoff->fatal distance clustering at 2.0-3.5 cells = exactly the jump span; 84% of deaths had
    another hazard within 5 cells behind the fatal one (33/37 at gap 1-2 = the user's "combinations").
  - `landing_hazard` = one velocity-invariant scalar: span = 0.0204*v - 0.94, landing offset =
    cell_progress + span, emit 1.0 if a must-jump hazard sits at the landing cell / 0.0 clear / 0.5
    beyond the 3-cell lookahead. HOST-DERIVED from already-streamed fields => **no hook change**.
    Verified speed-adaptive; obs parity CPU<->torch 1248/1248; surgery base
    `runs/ppo_v66_obs65_land.zip` (0/400 action mismatches).
  - **MECHANISM CONFIRMED LIVE during training: dP(jump) when landing_hazard flips 0->1 was
    -0.0035 @2M, -0.0051 @4M, -0.0067 @6M (max |dP| 0.33) — correct SIGN and GROWING, i.e. the policy
    reads the cue and jumps LESS when its arc would land on a hazard. (Contrast wave-13's v76/v77,
    killed early because the leading indicator stayed flat at 0.000000.)**
- **★★ TRAINING-SPEED FINDINGS (the GPU sits ~12-40% because the PPO optimizer loop is the wall, not
  the env): BIGGER BATCHES DO NOT PAY. Measured 256/512 -> 5.4k steps/s, 1024/2048 -> 12.3k,
  2048/4096 -> 18.8k, BUT at matched steps the learning is ~55% (1024/2048 ep_dist 794/945/1340 @1/2/3M
  vs 256/512's 1741/2096/2459) — it trades gradient updates for throughput ~1:1. KEEP 256/512.
  ★ THE REAL SAVING IS SCHEDULE LENGTH: v90's curve is 1741@1M, 2459@3M, 2520@6M, 2682@10M, 2494@11M —
  essentially ALL learning happens by 3M and the rest is LR annealing. A 2M annealing tail on the 11M
  checkpoint lifted ep_dist 2494 -> 2867 in 7 minutes. NEXT RUNS: ~6-8M with the cosine compressed into
  that budget (~20 min instead of 70), then verify against a 20M control ONCE.**
- **TOMORROW, in order: (1) re-calibrate held_jump_rejump_* at n=128 (and measure the sim-side hold
  rate with detect_patterns.py) — do NOT train until the champion's sim median lands near its device
  median; (2) train v91 = v90's recipe + the calibrated held-jump model, ~6-8M compressed schedule;
  (3) judge it with detect_patterns.py (HELD_JUMP_SPAM per km must fall toward the human's 0) and on
  PER-ENCOUNTER death rate, NOT the 8-episode median.**

## ⚡ EARLIER 2026-07-26 — read this first
- **★★★★ PROTOCOL CORRECTION (2026-07-26, USER-driven, MEASURED — read before trusting ANY device
  verdict in this file). USER: "the record of v78 is based on randomness... any of the models
  deployed after v78 had the potential to do as great." CORRECT, and the numbers are brutal:**
  - **An 8-episode median has a 95% CI of ~[3235, 8092]. v78 vs v88: p=0.50. v78 vs v79: p=0.23.
    NONE of this week's device "verdicts" (v79, v83, v86, v87, v88 all "rejected") were
    statistically resolvable. They were coin flips presented as conclusions.**
  - **POWER at n=8 to detect a TRUE 20% median improvement: 11%. A genuinely 20%-better model is
    MISSED 9 times out of 10. 80% power needs ~48 episodes/arm (~2.5 h phone time each).**
  - **Per-ENCOUNTER rates (`scripts/analysis/device_compare.py`) give 40-70x the sample (n=341-569 hazard
    encounters per 8-ep batch vs 8 distances) BUT deaths are RARE (1-2%/encounter), so a batch still
    holds only ~5-8 death events — v78 vs v88 is p>0.47 on EVERY per-encounter metric too.**
  - **THE RULES THAT FOLLOW:**
    1. **Judge SIM FIDELITY by direct measurement (`scripts/validate/sim_real_audit.py` coverage / spacing /
       champion's sim-vs-device median). Those are high-power and do NOT depend on noisy episode
       medians. That is what validated the wave-15 ramp, and it still stands.**
    2. **Do NOT rank policies on an 8-episode batch, and do NOT write "REJECTED" from one. Either
       spend ~48 eps/arm for a real crowning decision, or defer the ranking.**
    3. **Prefer HIGH-RATE metrics: the lvl-3 crevasse wall-clip runs ~20%/encounter (vs 1-2% for
       deaths) so it resolves far faster — conveniently the mechanic the user most wants fixed.**
  - **CONSEQUENCE FOR THE RECORD: v78 remains the DEPLOY DEFAULT (it has the most accumulated
    evidence), NOT the demonstrated best. v88 in particular has the highest floor ever measured
    (min 1333, zero sub-1000, 6/8 above 3400) and is statistically tied with v78 — it should NOT be
    treated as a failure, and the earlier "regression" language for v79/v83/v87/v88 is overstated.**

- **★★★★ THE GAME'S OWN DIFFICULTY RAMP, READ DIRECTLY OFF THE DEVICE (2026-07-26). This supersedes
  every statistical estimate of the generation law and is almost certainly THE sim-to-real gap.**
  - **USER: "isn't it things that you can find in the source code instead of collecting data on the
    phone? I was looking for a cheap way to estimate the difference between the simulation and real
    data without having to run a policy on the phone." CORRECT — and far better than fitting.
    `PathElement` keeps its WHOLE generation law in STATIC fields (dump.cs:283287). libil2cpp.so
    exports are stripped and `DumpStatics()` crashes without IL2CPP context, but an IL2CPP object's
    first word is its Il2CppClass* and `static_fields` sits at klass+0xB8 — so a LIVE PathElement
    (which the hook already holds every frame) reaches them. Hook 17j samples them every ~2s and
    streams `s.gen`; `capture_calib` logs it. ONE 3-episode run gives the exact schedule.**
  - **THE RAMP (one run to 10,768 m; distance-keyed, exact — NOT a fitted distribution):**
    ```
    dist(m)      minSpaceObstacles  minSpaceTurns  doubleObstaclePct  narrows flame/slide
    8..254             6                 12              0.00            no      no
    253..505           5                 10              0.00            no      no
    507..1006          4                  8              0.05            no      YES
    1109..1499         4                  6              0.05            no      YES
    1502..1999         3                  5              0.10            YES     YES
    2002..3253         3                  4              0.15            YES     YES
    3256..4506         3                  3              0.17            YES     YES
    4509..10768        2                  2              0.20            YES     YES
    ```
    (sMaxBackToBackObstacles stays 2 THROUGHOUT — the training recipes set 6, which is WRONG.)
  - **★ THE SIM HOLDS `min_space_between_obstacles = 6` FOR THE WHOLE RUN. The real game ramps it
    6 -> 5 -> 4 -> 3 -> 2. Past ~4.5 km the game packs obstacles THREE TIMES TIGHTER than anything
    the policy has ever seen in training. `sDoubleObstaclePercent` (0 -> 0.20) is NOT MODELLED AT ALL.
    `sMinSpaceBetweenTurns` ramps 12 -> 2 (sim: 12 early / 5 late).**
  - **THIS ONE TABLE EXPLAINS THE OPEN ITEMS:** the device's tight pairs (not a probabilistic sprinkle
    — it is the spacing LAW past 1500 m); the 17 pair-types with ZERO sim coverage incl. the whole
    "-> Crevasse" family; the sim being ~1.5x too easy (v78 6786 sim vs 4425 device); the deaths
    clustering in the mid band (where the real ramp bites and the sim flatlines); and the USER's
    "succession of obstacles" death (at 3-5 km the game runs minSpace 2-3, the sim says 6).
  - **IMPLEMENT IT AS AN EXACT SCHEDULE, not a knob to calibrate.** `tight_pair_prob` was a bad
    approximation of this and should be retired. NOTE the ramp is distance-keyed, so
    `curriculum_max_start_dist` must set the ramp state too, not just the velocity.
  - **★★ RAMP IMPLEMENTED + v88 TRAINED ON IT (2026-07-26). SIM MEASURABLY BETTER, POLICY NOT YET.**
    - **Shipped in BOTH sims as `device_difficulty_ramp` + `difficulty_ramp_table` +
      `double_obstacle_percent` (default OFF). Equivalence subset GREEN 1248/1248.**
    - **VALIDATION (all three parts, `scripts/validate/sim_real_audit.py`): COVERAGE 17 missing device
      pair-types -> 0 (every "-> Crevasse" compound now exists). DIFFICULTY v78 sim median
      6786 -> 5200 vs its measured device 4425 = the gap closed 1.53x -> 1.18x WITH NO FITTING —
      the first fidelity change ever validated without tuning a knob toward the answer.**
    - **v88 = v78 spine + ramp + turn_aware_lookahead + hole trap. Offline in the corrected world:
      6766 vs v78's 3808 (+78%), turn skill intact (PRE 100%, P 0.9973), and the crevasse graze
      HALVED in-sim (lvl-3 30% -> 14.3%, speed bled 3.2 -> 1.3 per km; takeoff inversion shallower
      0.64->0.45 vs v78's 0.68->0.40). DEVICE: median 3674 vs v78's 4424 = FIFTH offline->device
      inversion of the week. v78 REMAINS CHAMPION.**
    - **★ BUT THE FAILURE SHAPE CHANGED AND NAMES THE NEXT FIX. v88 is the most CONSISTENT policy
      yet — all 8 episodes in 1333..6853, the highest floor of any model, zero sub-1000 — and it lost
      on the CEILING (max 6853 vs v78's 12066), never producing a long run. That is the signature of
      a policy trained in a world that is TOO TIGHT. The audit said so and I moved on: hazard pairs
      inside the jump arc are 14.3% in the ramped sim vs 10.8% on device, i.e. ~30% over-packed.**
    - **★ AND IT SEPARATES TWO THINGS I HAD CONFLATED AS "DIFFICULTY": the sim is simultaneously TOO
      EASY overall (1.18x on the champion's median) and TOO TIGHT in spacing. Different dimensions,
      different fixes. The ramp fixed the spacing LAW (min_space floor); what remains is the
      obstacle PLACEMENT RATE on top of that floor — `obstacle_prob` was tuned back when the floor
      was a flat 6 and is now emitting too eagerly against a floor of 2-3.**
    - **NEXT: fit `obstacle_prob` so the arc-share lands on the device's 10.8% (not 14.3%), re-check
      that v78's sim median stays near 4425, then retrain. Do NOT touch the ramp table — it is read
      from the game and is not a free parameter.**
  - **NEXT SESSION STARTS HERE:** (1) implement the distance-keyed ramp in both sims (minSpace,
    minSpaceTurns, doubleObstaclePercent, the allow* gates, b2b=2); (2) re-run
    `scripts/validate/sim_real_audit.py` — coverage and the spacing histogram should close by construction;
    (3) verify DIFFICULTY: v78's sim median should fall from 6786 toward its device 4425 WITHOUT any
    fitting; (4) only then retrain. Probe: `scripts/probes/probe_generation_statics.py` (standalone, needs
    fixing for stripped exports) and hook 17j's `s.gen` (works — use this).

- **★★★ STRATEGIC CONCLUSION (2026-07-26, USER-stated, now QUANTIFIED): "the problem is not really the
  training but rather the simulation... We need to close the sim-to-real gap." CORRECT. The binding
  constraint is the sim's GENERATION LAW, not its death models and not the training recipe. Four
  offline->device inversions in ONE day (v79 -40%, v83 -26%, v86, v87 -29% while +9% offline) is not
  bad luck; it is what happens when you optimise against a world that lacks the situations that kill.**
- **★★★ NEW STANDING GATE: `scripts/validate/sim_real_audit.py` — run it BEFORE proposing any training arm.**
  It splits the sim-to-real gap into the three parts that were previously conflated:
  1. **COVERAGE** — does the situation EXIST in the sim? (device pair-types with ZERO sim coverage)
  2. **DENSITY** — is the hazard spacing distribution right?
  3. **DIFFICULTY** — does the champion score in-sim what it scores on the phone?
  **THE RULE THIS ENFORCES: a perfect death model for a situation the sim never generates is worth
  NOTHING. Ask "does this situation exist in the sim?" BEFORE "is its death model right?". ~20 models
  were spent on the second question. Two coverage checks on 2026-07-26 both returned ZERO.**
- **★★★ AUDIT RESULT 2026-07-26 (device n=2900 hazard pairs / 20090 cells vs the v78/v87 training gen):**
  - **COVERAGE: 17 device pair-types have ZERO sim coverage = 4.59% of all device pairs. The ENTIRE
    "-> Crevasse" family is missing: TreeSlide->Crevasse 28 device/0 sim, Hole->Crevasse 15/0,
    FlameTower->Crevasse 13/0, TreeJump->Crevasse 10/0 (66 occurrences). ROOT CAUSE = ONE LINE,
    `sim_env.py` zone-crevasse block: the crevasse placement itself demands
    `_cells_since_obstacle >= min_space_between_obstacles`, so a crevasse can NEVER be generated
    within 6 cells of an obstacle. `tight_pair_prob` does NOT help — it relaxes only the NORMAL
    obstacle path. THIS IS THE USER'S REPORTED DEATH ("an obstacle that forced a jump, followed by a
    crevasse; it jumped as usual and landed directly into the crevasse") — structurally untrainable.**
  - **DENSITY: the device spacing law is SMOOTH, the sim's is lumpy with a hole where the device is
    densest. gap 1: 1.7% vs 0.0% | gap 2: 2.5% vs 0.5% | gap 3: 6.6% vs 7.8% | gap 4: 11.9% vs 10.3%
    | gap 5: 19.7% vs 0.6% (!!) | gap 6: 18.9% vs 12.9%. The hard `min_space_between_obstacles` floor
    of 6 forbids everything below it, and the only sub-6 spacings come from the bespoke compound tails
    which emit fixed 3-4. FIX DIRECTION: replace the hard floor with a device-fitted spacing
    DISTRIBUTION (closes gap 1-2 AND the gap-5 hole in one change) instead of bolting on more
    hand-built compounds.**
  - **DIFFICULTY: v78 scores 6786 in-sim vs 4425 on device = the sim is ~1.5x TOO EASY.**
- **★★ THE OTHER 2026-07-26 FINDING — PERCEPTION (shipped, hook 17h + `turn_aware_lookahead`):** the
  hook's PathElement neighbour walk used ONE fixed heading offset for EVERY step, so it died at every
  corner — 72% of near-junction device frames streamed next/next2 = -1 and NOTHING could be seen past
  a turn. A heading-aware walk (follow forward, else the single available turn; REFUSE at a true
  T-junction where the continuation is undecided) resolves the full 6-cell horizon on 100% of the
  frames where the old walk fell short. **Device-verified: junction blindness 72.1% -> 27.4%** (the
  residual being the deliberate T-junction refusal). Shipped in BOTH sims (`turn_aware_lookahead`,
  default OFF) + hook 17h (`TURN_AWARE_LOOKAHEAD` const). ⚠️ **The hook flag MUST match the policy:
  v78 and earlier trained blind and must run on 17f (or the const set false).** Turn skill verified
  intact under the new obs (PRE% 100.0, P(turn|approach) 0.9995). v86/v87 trained on it; v87 was the
  best offline of the wave (6031 vs v78 5535, Falls 9 vs 14) and STILL lost on device (3136 vs 4424)
  — the perception fix is real and measured, but it is not sufficient while the generation law is wrong.
- **★★ THE USER'S TWO OPEN COMPLAINTS, both traced to the SAME root (the min_space floor):**
  1. **"On high ground it makes a second jump right after jumping an obstacle, for no reason" —** the
     6-cell floor GUARANTEES a clear landing zone after every jump, so a stray second jump is FREE
     in-sim. (This also explains why v60/v61's reward shaping failed: they penalised a behaviour the
     world never punished.)
  2. **"An obstacle that forces a jump, followed by a crevasse" —** zero sim coverage, see above.
- **★★ MID->HIGH CREVASSE ("it works but it's not clean; the character often hits the wall which slows
  it down") — MONITOR BUILT (`experiments/crevasse_monitor.py`), and this one is NOT a coverage gap:**
  level-3 cap graze v78 device 19.5% (8/41) vs HUMAN 0.0% (0/49); v78 bleeds 5.1 velocity units/km to
  grazes vs the human's 0.8. **The SIM ALREADY PUNISHES IT HARDER than the device (30% graze, 15%
  kill) — so this is NOT a missing mechanic and NOT fixable by another sim knob.** The separator is
  TAKEOFF TIMING: the human is LATE and SPEED-ADAPTIVE (cp 0.47 -> 0.66 -> 0.82 as speed rises), the
  policy is EARLY and INVERTED (0.56 -> 0.40 -> 0.32). Physically the arc spans more cells at speed,
  so takeoff must be DELAYED to still be near apex at the raised cap; the policy learned the opposite.
  Secondary measured fact: the policy takes off ~0.13 cp EARLIER on device than in-sim at every speed
  (likely obs freshness — device streams 30Hz and the env drains to the newest frame). OPEN.

## ⚡ PREVIOUS STATE (2026-07-25)
- **★★★ CHAMPION IS STILL v78 (`runs/ppo_v78_guided_18001920_steps.zip`, obs64, 512-net, hook 17f).
  v79 (wave-14) DEVICE-REGRESSED and is REJECTED. Nothing beat v78 on 2026-07-25.**
  - **v79@18M device (8 ep, 34->38C, hook 17f): 611/701/932/2143/3149/5098/7390/11317 = MEDIAN 2646
    vs v78's same-week fresh 4425. THREE sub-1000 dumps.** Offline said v79 ~= v78 (standard cfg
    10645 vs 10472) — the phone said otherwise. The offline->device inversion again; the phone wins.
  - **DEATH SIGNATURE: pt2 x3 (plain STRAIGHT cells) + an early crevasse pt36 @611 — i.e. the USER's
    long-reported "random jump" and "mid->high not nailed", both visibly WORSE in v79.**
  - **WHY (honest): v79 stacked TWO sim changes (hole trap + 67.5% one-cell crevasses). Both are
    individually device-measured and correct; the STACK still regressed. The project's own
    one-change-per-retrain rule exists for exactly this. Prime suspect = crevasse_single_cell_prob
    0.675: device-true, but it made 2/3 of crevasses trivial and thinned the practice on the HARD
    2-cell mid->high cap. The wave-14 knobs stay in the code (equivalence-gated, default OFF) and
    stay in gate_battery.DEVICE_TRUE, but NO policy trained with them has earned the phone yet.**
- **★★★ WAVE 15 (2026-07-25, USER-DIRECTED) — THE POST-TURN JUMP: root-caused to a SIM GENERATION
  ERROR, one config value, not yet trained to completion. THIS IS THE STRONGEST LEAD FOR TOMORROW.**
  - **USER (live, repeatedly over many sessions): "the policy tends to jump randomly after a turn. No
    reason involved... I saw a useless jump after a crevasse on high ground... a random jump is often
    the reason of death. If those erratic moments are mastered, the reliable 10,000m is in the
    pocket." CONFIRMED AND ROOT-CAUSED.**
  - **⚠️ MEASUREMENT BUG FIRST (cost an hour, worth knowing): `is_turn` is only presented across the
    EXECUTABLE window (turn_exec_early..late), so it is ALREADY FALSE on the LAST frame of a junction
    cell. Reading it there found 3 junctions in 4590 cells (0.1%; truth ~4%) and made post-turn
    analysis silently report ZERO. Mark a cell a junction if ANY frame on it says is_turn.**
  - **THE METRIC (`experiments/wasted_jump.py`): a WASTED jump = a takeoff whose arc (span = 0.0204*v
    - 0.94 cells) lands SHORT of the next must-jump cell -> it cleared nothing, cost ~0.77s airborne
    and an action-locked landing frame. Result — post-turn is where they live, for EVERYONE:**
    | | wasted/km | post-turn share | post-turn lethality |
    | v79 | 0.8 | 61.5% | 6.2% |  | v78 | 0.6 | 70.0% | 0% |
    | v74 | 0.7 | 68.4% | 7.7% |  | v70 | 0.7 | 61.9% | 7.7% |
    | **HUMAN** | **2.6** | 61.9% | **0/65** |
    **The HUMAN hedges post-turn 2.4x MORE and NEVER dies from it; the policies hedge less and die
    ~7% of the time. The extra jump is NOT the sin — where it LANDS is.** (Cause of the hedge itself
    is the documented junction obs dropout: the hook's neighbour walk breaks through turns, so
    next/next2_path_level stream -1 and NOBODY can see past a corner.)
  - **★ ROOT CAUSE (`experiments/postturn_obstacles.py`) — P(must-jump obstacle) by cell offset after
    a junction, DEVICE vs SIM:**
    ```
    DEVICE                    +1: 4.3%  +2:  0.7%  +3: 21.7%  +4: 15.0%  +5: 4.8%
    SIM min_clear_after_turn=2 +1: 4.7%  +2: 14.4%  +3: 17.4%  +4:  9.2%  +5: 8.5%   <-- +2 is 20x TOO DENSE
    SIM min_clear_after_turn=3 +1: 5.2%  +2:  0.0%  +3: 19.8%  +4: 22.4%  +5: 9.2%   <-- MATCHES DEVICE
    ```
    **The sim has been TRAINING a jump at post-turn cell +2. On device that cell is EMPTY, so the
    jump is wasted — and a 2.5-3.3-cell arc launched there lands at +3/+4, exactly where the device
    concentrates its post-turn obstacles. The sim was teaching the user's "random jump after a turn".
    FIX = `min_clear_after_turn` 2 -> 3. EXISTING knob in BOTH sims: no code change, no equivalence
    risk, one value. Verified NOT a difficulty distortion (v78 6786 -> 8236; it removes a
    non-existent obstacle, it does not add pressure).**
  - **⚠️ MY CALIBRATION ERROR — READ BEFORE ADDING ANY GENERATION KNOB. I set the new
    `tight_pair_prob` to 0.55/0.65 by matching ONE statistic (device's 6.2% of must-jump pairs within
    the jump arc). That matched, and TOTAL obstacle density blew up ~6x. Dose-response on v78 (n=64,
    8000-step cap): `tight_pair 0.0 -> 6786 | 0.15 -> 3188 | 0.35 -> 1205 | 0.55 -> 772`, against
    v78's ACTUAL DEVICE median 4425. Two 10M-step arms (v81, v82) were training against a sim ~6x
    harder than the real game before I caught it — both killed. LESSON: calibrate a generation knob
    on the OUTCOME (does the champion score in-sim what it scores on the phone), not only on a
    distribution statistic. Device-matching dose is ~0.08-0.10.**
  - **★ SIDE-FINDING, possibly the most valuable of the day: at tight_pair 0 the sim gives v78 6786
    vs 4425 on device — THE SIM IS ~1.5x TOO EASY. That is the chronic offline->device correlation
    failure quantified in one number, and it explains why offline wins keep failing to transfer. A
    ~0.08-0.10 tight_pair dose brings overall difficulty onto the device AND is independently
    justified (device really does place 19% of consecutive must-jump pairs closer than the sim's hard
    min_space floor of 6; gaps 1:23 2:49 3:95 4:74 5:272 of n=2684).**
  - **★★ v83 VERDICT (2026-07-26, device, 8 ep, 32->36C = COOLER than v78's own batch): REJECTED.
    791/1348/1928/2745/3766/5395/6267/9272 = MEDIAN 3256 vs v78's 4424 (mean 3939 vs 5248, max 9272
    vs 12066). Better than v79's 2646 but still under the champion. THE HYPOTHESIS IS FALSIFIED, and
    cleanly: the generation fix WORKED (sim +2 density 14.4% -> 0.0%, matching device 0.7%) yet the
    policy's post-turn hedge on DEVICE did not move at all — wasted 0.7/km vs v78's 0.6/km, post-turn
    share 69.6% vs 70.0%. So the sim's +2 obstacle density was NOT what taught the post-turn jump.
    min_clear_after_turn 3 is still the device-TRUE value (keep it as fidelity), it just is not the
    lever for the user's complaint.**
  - **★★★ WHAT THAT LEAVES = THE JUNCTION OBS DROPOUT (now the leading hypothesis for the user's #1
    complaint, and it is a PERCEPTION bug, not a generation one). Two facts pin it: (a) NEITHER v78
    NOR v83 wastes a SINGLE jump in 74 km of SIM rollout (0/675 takeoffs) while both waste 6.5-7.3%
    of takeoffs on DEVICE — the behaviour does not exist in the sim, so no generation knob can train
    it away; (b) 61-70% of the wasted jumps sit within 3 cells of a junction, for the HUMAN too. The
    documented cause: the hook's PathElement neighbour walk BREAKS through turns, so next/next2
    (path_level AND must_jump) stream -1/False across a junction — NOBODY can see past a corner, so
    everyone hedges. The human hedges MORE (2.6/km) and survives because their hedge is well-timed.
    NEXT LEVER = make the hook walk the neighbour chain THROUGH a junction (PathElement neighbours
    0x50/58/60/68; find which is the turn-direction link) so the policy SEES the post-turn cells.
    Precedent says this is the highest-value class of fix in the project: the last perception fix
    (next2_must_jump/slide, obs 58->60, v51) was the single biggest device gain ever measured
    (+108%). If the walk cannot be fixed, the fallback is an obs that marks "blind through junction"
    explicitly so the policy can learn a SAFE hedge rather than a blind one.**
  - **ARMS STOPPED MID-FLIGHT at user request (resume with `--init-from <ckpt>`, do NOT restart from
    the v66 base): `runs/ppo_v83_postturn_16003072_steps.zip` (v78 recipe + ONLY min_clear_after_turn
    3, reached 16M/20M) and `runs/ppo_v84_both_4001792_steps.zip` (+ tight_pair 0.10, reached 4M/20M).
    v84 vs v83 isolates what the tight_pair dose adds on top of the post-turn fix.**
  - **TOMORROW, in order: (1) finish v83 to 20M -> gate_battery vs v78 -> device batch vs the 4425
    baseline (NO alternating A/B — the user judged it a waste of time; run the candidate and compare
    to stored v78 numbers). (2) finish v84. (3) STILL OPEN, the user's #2 ask: the mid->high crevasse
    at DIFFERENT SPEEDS — the user's own suggestion is "craft special episodes at different speeds".
    The v81 attempt bundled it with the bad tight_pair dose and a suspect spine; redo it CLEAN on the
    v78 spine as crevasse-dense episode_profiles at 3 zone_after_distance bands (0/1500/3500) +
    curriculum-max-start-dist 9000. (4) The wave-14 hole trap remains unproven on device — retry it
    ALONE on the v78 spine (NOT stacked with crevasse_single_cell_prob) before judging it.**
  - **PROCESS NOTE the user set today: run PARALLEL TRAINING CONFIGS (the GPU idles ~70% during one
    run; 2-3 concurrent 20M arms fit in ~1.1GB/55%) and do NOT spend phone time on alternating A/B
    batches. Every arm must change ONE thing against a spine that is device-proven, or its result is
    unattributable — v80a/v80b were both built on the regressed v79 spine and had to be discarded.**
- **★★★ WAVE 14 (2026-07-25) = THE AIRBORNE TRAP: the hole wall is a SIM-GEOMETRY gap, and it is the
  cleanest device-measured find since the turn dead-letter. Two knobs shipped in both sims, equivalence
  GREEN (deterministic 4992/4992 + statistical PASS); v79 = v78's exact 20M recipe + this one mechanism.**
  - **THE DEATH MODE (`scripts/analysis/hole_landing_forensic.py`, 1240 device hole crossings / 4 policies):
    12 of v78/v74/v70's 15 device hole deaths happen with the runner ALREADY AIRBORNE — it jumps
    obstacle A, the hole sits INSIDE its jump arc, it cannot re-take-off, and its DESCENDING arc drops
    into the hole. Taking off on the APPROACH cell is FREE on device: 0 deaths in 858 crossings.
    THE SEPARATOR IS ARC PHASE, not landing position: every death entered at frac = d/span >= 0.51
    (past apex, descending); every one of the 858 free crossings is at frac <= 0.41. `d` = cells from
    takeoff to the hole cell. Killed v78 @612 and its 12,779 record @12,770; killed v74 @1722/@1996/
    @3794/@4926. OUT-OF-SAMPLE CONFIRMED on the same-day fresh capture (`captures/w14_v78_holes.jsonl`,
    175 crossings): the ONE airborne hole death sits at frac 0.82, and 0/134 at d<=1.**
  - **WHY IT WAS UNLEARNABLE — TWO independent gaps, both now closed:**
    - **(a) PHYSICS: a 1-cell hole is not `wide_gap`, so the wide-gap arc-landing model SKIPPED it
      (`sim_env.py` must-jump branch) and the ONLY gate was binary `_jump_t > 0` — an almost-expired
      arc crossed a hole FOR FREE.** FIX = `hole_landing_model` + `hole_trap_apex_frac` 0.5 +
      `hole_trap_kill_prob` 0.35 (probabilistic, device 6/20 = 30%; a sharp threshold was the v55
      mistake). It CANNOT touch the d<=1 regime (frac<=0.41 < 0.5) so v78's hole skill is structurally
      safe. Long holes were already priced by `wide_gap_landing_model` — unchanged.
    - **(b) GEOMETRY (the real root): the sim made EVERY crevasse 2 cells; on DEVICE 67.5% are ONE
      cell (n=409).** That extra cell pushes the next obstacle beyond the ~2.5-3.3-cell jump arc, so
      the trap could never form: across ~900 sim crossings in FIVE configs (standard / dense / high-
      speed / gap-2 tail / gap-1 tail), `d` NEVER exceeded 1.0. FIX = `crevasse_single_cell_prob` 0.675.
  - **⚠️ THE TRAP I NEARLY WALKED INTO (and the measurement that saved it): my first 1-cell model gave
    it the full apex/graze physics on its single cell -> the oracle died 25% MORE (dt2 34->52, mean
    1038->779). DEVICE SAYS THE OPPOSITE (`experiments/crevasse_cells_outcome.py`, n=827): the 1-cell
    form is ~6x SAFER — 1.1% deaths / 0.0% grazes vs the 2-cell 7.0% / 33.7%. And the obvious
    confounder is RULED OUT by a length x level cross-tab: at MATCHED level>=3, 1-cell = 0% deaths /
    0% grazes (n=99) vs 2-cell = 7.0% / 33.7% (n=285). It is the LENGTH, not the height. (The 33.7%
    independently reproduces the documented 32% level-3 cap graze — which correctly stays with the
    2-cell form.) So a 1-cell crevasse is modelled as a plain NON-wide must-jump cell: airborne-at-
    entry is the whole requirement, no apex check, no cap graze, no gap-lip jump refusal.**
  - **CALIBRATION (sim vs device, v78 scored in each): TRAP% (crossings entered past apex) sim 0.77%
    vs device 2.31%, median frac 0.30 vs 0.26 — same order, right shape. Scoring v78 in the corrected
    sim: standard median 5535 -> 5045 with Fall deaths 10->14 = the trap BITES without cratering it.**
  - **⚠️ BATTERY + EVAL CHANGES (do not skip — the v74 false-RED lesson): the wave-14 knobs are now in
    `gate_battery.DEVICE_TRUE` (they describe the DEVICE, like post_land_gap_rejump), the default
    champion is v78, and `--max-episode-steps` is threaded through with default 8000 — under the
    corrected sim v78 SATURATES the old 4000-step cap (median 8110 / max 8454 = censored). ALL
    champion baselines from before 2026-07-25 were measured in the uncorrected sim and are NOT
    comparable; re-measure through the battery.**
  - **DEVICE BASELINE for the A/B (v78, hook 17f): fresh 8-ep capture at 36->39C = 1055/3235/3953/4300/
    4549/4735/8092/12066, median 4425 (3 of 8 deaths on SINGLE holes = the target mechanic; SECOND
    12k-class run of the project). Pooled with the wave-13 cooled batch (n=12): median 4642.**
  - **v79 (`experiments/train_v79_holetrap.sh` -> runs/ppo_v79_holetrap) = v78's EXACT 20M consolidation
    from `ppo_v66_obs64_base` + ONLY this mechanism (consolidation, NOT finetune — the v52/v63/v71c/v73
    erosion lesson). Turn skill VERIFIED PRESERVED mid-flight: @3M PRE% 100.0 / MISS 0.0 /
    P(turn|approach) 0.9859. Gate: `gate_battery.py <ckpt>` (crevasse config is the one to watch — the
    1-cell form changes crevasse geometry) -> cooled device A/B vs v78.**
  - **METHOD NOTE for the next session: the standard config's trap rate is too rare (0-1 events per 130
    crossings) to read a checkpoint mid-flight. Use a TRAP-STRESS cfg (min_space_between_obstacles 3,
    obstacle_prob 0.7 + the wave-14 knobs) the way wave 13 used the isolated turn-cfg. Remaining
    structural spacing gap (NOT yet modelled): device puts 19% of consecutive must-jump obstacles
    CLOSER than the sim's hard `min_space_between_obstacles` floor of 6 (gap 1:23, 2:49, 3:95, 4:74,
    5:272 of n=2684) and 6.2% within the jump arc (<=3 cells); the sim's compound tails only ever
    produce gap 3-4. A general `tight_pair_prob` is the next fidelity lever if v79 under-delivers.**
- **★★★ PRIOR CHAMPION = v78 (`runs/ppo_v78_guided_18001920_steps.zip`, obs64, 512-net, hook 17f). The
  TURN DEAD-LETTER — the last tail wall + v74's #1 device ender — is SOLVED in 100% RL via GUIDED
  EXPLORATION (exploring-starts). Device A/B: v78 median 5455 / 0 pt-1 deaths vs v74 3914 / 3-of-8 pt-1;
  NEW ALL-TIME RECORD 12,779 m (re-crosses the 10k goal). The pre-swipe the policy could NEVER discover
  on-policy (P(turn|approach) collapsed to 0.000000) is now 99.7% learned. See WAVE 13 below. NEXT median
  lever = HOLES/long-holes (pt11-16) — the turn wall is gone, holes are the new ~5.5k median cap. Prior
  champion v74 (device median 4531) is the obs64 fallback.**
- **★★ WAVE 13 TURN FIX = A REWARD BUG, NOT AN RL LIMIT (2026-07-14, USER-challenged "RL is powerful,
  there must be a way; I want 100% RL"). The "5 arms plateau → on-policy RL can't discover the pre-swipe"
  conclusion was WRONG. Root cause found in CODE (`reward.py:124`, the reaction_bonus turn branch): the
  dense shaping REWARDS the REACTIVE junction swipe (`prev.is_turn`) — the exact behavior that
  dead-letters under latency — while the latency-robust APPROACH-cell pre-swipe earned ZERO shaping.
  MECHANISM PROBE (`experiments/preswipe_fraction.py`, isolated turn-cfg lat0.7+drop+debounce, n=40 eps):
  v75 AND v74 pre-swipe on 0.0% of junctions (react 61-62%, MISS 38-39% = the reactive swipes that
  dead-letter). The pre-swipe is COMPLETELY ABSENT — not underused — despite the policy HAVING the
  direction obs (next_can_turn_*), the latch (turn_approach_buffer), safe exploration (turn_offjunction_
  drop), and the cue (predicted_turn=1.0). It pre-swipes 0% because there's no reward gradient toward it
  and a positive one toward reacting. The policy learned exactly what it was paid to do. FIX = LEVER A:
  `turn_buffer_bonus` (reward.py + both sims + train_fast `--turn-buffer-bonus` + equivalence variant
  `turn_bufbonus`) credits a valid-direction pre-swipe on the approach cell — un-gameable (spam eaten by
  debounce, a reactive swipe doesn't latch), same shape as the jump/slide reaction_bonus that WORKS.
  v76 (`experiments/train_v76_bufbonus.sh`) = v75's EXACT recipe + ONLY --turn-buffer-bonus 0.5 = a clean
  A/B (v75 = bonus OFF, 0.0% pre-swipe, median 1033-1110) at matched steps. Read the mechanism early
  (preswipe_fraction at 4/6/8/10M): if PRE% climbs from 0 → the fix works → 20M consol → gate → device.
  DEMOS (the human gi_* swipe buffer) stay in reserve as demo-AUGMENTED RL (DAPG: PPO + an annealed BC
  aux that only fixes early exploration, reward-max dominates), NOT supervised replacement — but try pure
  Lever A first to keep it 100% RL. v75 = 5th plateau, negative result, NO phone (won't beat v74's turns).**
  - **★★ DEADLOCK REFINEMENT (2026-07-14, the leading-indicator read — the KEY finding): Lever A ALONE
    is INSUFFICIENT, and the reason is deeper than reward alignment. `preswipe_fraction.py` now also
    reports P(correct-turn | approach cell) = the policy's PROBABILITY mass on the pre-swipe (a LEADING
    indicator; the deterministic PRE% is LAGGING — it only flips once the pre-swipe becomes the argmax).
    Measured: P(turn|approach) = 0.0000 for v74, v75, AND v76@1M/2M (bonus ON) — FLAT. The policy assigns
    ~0 probability to turning on the approach cell, so the pre-swipe is NEVER SAMPLED, so turn_buffer_bonus
    NEVER FIRES, so zero gradient. v76 (bonus-only) KILLED at 2M (confirmed dead: 2M of the bonus moved the
    leading indicator not at all). THIS IS THE HONEST ANSWER to 'why can't on-policy RL discover it':
    vanilla PPO exploration (ent 0.01) COLLAPSED the approach-turn probability to ~0, and a success-only
    reward cannot reinforce an unsampled action. It is a HARD EXPLORATION DEADLOCK — a known limitation of
    NAIVE exploration, NOT of RL itself. Reward alignment is necessary but not sufficient; the missing
    piece is SAMPLING. v77 (`train_v77_explore.sh`) = v76 + ONLY ent-coef 0.01→0.05 (no sim/reward change,
    no re-equivalence) tests the direct pure-RL fix: does forcing exploration lift P(turn|approach) off 0?
    Read at 1-2M (leading indicator). If yes → proper annealed-entropy run → gate → device. If flat even at
    high entropy → the global-entropy tension (too low=collapse, too high=chaos-before-junctions) means this
    LOCALIZED deadlock needs GUIDED exploration (approach-cell action injection = exploring-starts, still
    100% RL) or DAPG (demo-bootstrapped exploration, RL-optimized). ALL scripts/probes in scratchpad;
    turn_buffer_bonus + preswipe_fraction.py committed-worthy infra (equivalence-gated).**
  - **★★★ GUIDED EXPLORATION = THE BREAKTHROUGH (2026-07-14, v78). The deadlock is 100%-RL solvable —
    the USER was right that RL can do it; the missing piece was EXPLORATION, not reward/entropy.** New
    trainer knob `--turn-explore-prob` (+ `--turn-explore-anneal-frac`) in train_fast.py + `approach_turn_
    hint()` in sim_env_torch.py (returns the correct-direction turn on approach cells; reads state, mutates
    nothing → NO sim/equivalence impact — it's an exploration procedure like ent-coef, lives in the TRAINER
    not the sim). EXPLORING-STARTS: during rollout, FORCE the correct pre-swipe on approach cells with prob
    eps, store the EXECUTED action + its policy log-prob (PPO clipping absorbs the off-policy injection),
    so turn_buffer_bonus IS sampled and its positive advantage (survive latency-immune + bonus) propagates
    → pi(approach-turn) rises off 0. eps ANNEALS to 0 over anneal_frac*timesteps so the back half is PURE
    on-policy RL (policy must SUSTAIN the pre-swipe unforced = proof it LEARNED it). v78 = v76 recipe +
    --turn-explore-prob 0.5 --turn-explore-anneal-frac 0.5 (`train_v78_guided.sh`). **RESULT @1M (eval is
    DETERMINISTIC, NO injection → measures the policy's OWN behavior): PRE% 0→98.9%, MISS(dead-letter)
    38→0.4%, P(turn|approach) 0.000000→0.670491. THE DEADLOCK IS BROKEN — the policy pre-swipes on its
    own. ISOLATED TURN-CFG DISTANCE @1M = median 2261 (mean 2575, max 7570) vs the ~1050 PLATEAU that
    capped ALL SIX prior arms = +115%, MORE THAN DOUBLE. Death mix flipped turn→obstacle-dominated (the
    60 type-1 Falls are now HOLE falls, not dead-letters; MISS 0.4%) = the turn wall is GONE.**
    **★★ SELF-SUSTAIN TEST PASSED @10M (forcing FULLY ANNEALED TO 0 = pure on-policy RL): PRE% 100.0%,
    MISS 0.0%, P(turn|approach) 0.997054, turn-cfg median 7048 (mean 5047, max 8233, deaths {1:12, 2:36}
    = obstacle-limited, ~half survive to the 4000-step eval cap). The pre-swipe HELD with zero forcing =
    GENUINELY LEARNED, not propped up. Obstacle skills INTACT (dies on Trees/holes, not turns). ~7x over
    the 1050 plateau. The turn dead-letter — v74's #1 device ender — is SOLVED, 100% RL.**
    **★★★ GATE BATTERY @18M vs v74 (n=96) = CLEAN GREEN SWEEP, NO EROSION ANYWHERE: standard 5535 vs 2719
    (2.04x), crevasse 1450 vs 1344 (1.08x — the v71c erosion-risk config HELD), stacks 3244 vs 2120 (1.53x),
    latstress 4646 vs 976 (4.76x). type-1 Falls tell it: v74 standard {1:74} vs v78 {1:10}; v74 latstress
    {1:87} vs v78 {1:10} — the dead-letter was silently capping v74's median EVEN IN STANDARD (junctions are
    everywhere), so killing it ~DOUBLED the standard median. Fixing turns lifted the WHOLE distribution.
    v78 = the new deploy candidate (obs64, 512-net, hook 17f). DEVICE A/B vs v74 JUSTIFIED.**
    **★★★ DEVICE A/B VERDICT (2026-07-14, same-session cooled 33→36°C, hook 17f, ckpt @18M): v78 = THE
    NEW DEPLOY CHAMPION. v78 batch (n=4, coolest): 12779 / 5579 / 5332 / 612, median 5455, ZERO pt-1
    turn deaths (deaths = holes pt11/16 + obstacles pt1/2). v74 control (n=8): median 3914, 3/8 = pt-1
    TURN DEAD-LETTER (5498/4034/4280 all pt-1). Same-cool round-1: v78 5455 vs v74 4157 = +31% median.
    THE DEATH SIGNATURE IS THE PROOF (thermal-independent): v74 dies 3/4 on the turn dead-letter, v78
    0/4 — the guided-exploration turn fix TRANSFERRED TO DEVICE exactly as the gate predicted.**
    **★★★ NEW ALL-TIME RECORD + 10k GOAL RE-CROSSED: 12,779 m (v78@18M, hook 17f, A/B batch ep0, died
    on a long hole pt16) — beats the prior 11,731 (v74's best was 9202). Archived
    `captures/best_run_12779m_v78_hook17f.jsonl`.** REMAINING WALL for a reliable 10k MEDIAN: HOLES +
    obstacles (v78's median ~5.5k is now hole/obstacle-limited, NOT turn-limited — the 612m ep was an
    early hole dump; v74's new deaths were pt13/pt35/pt1 too). The turn wall is GONE; the next median
    lever is the hole/long-hole death model (pt11-16) + obstacle stacks. Weights
    `runs/ppo_v78_guided_18001920_steps.zip`. NEXT:
    (1) [DONE @1M: 2261; @10M: 7048 vs 1050] turn-cfg DISTANCE broke the plateau;
    (2) watch PRE% HOLDS through the anneal region (~8-12M) as forcing→0; (3) gate battery — VERIFY the
    obstacle skills (crevasse/stacks) did NOT erode (the injection is turn-specific but the policy shifted
    a lot) → cooled device A/B vs v74. The RL lesson: 'RL is powerful' = exploration is the hard part;
    reward shaping + global entropy CANNOT resurrect a specific collapsed action, exploring-starts CAN.**
- **★ WAVE 13 (2026-07-14): the two tail walls (flame + turn) diagnosed as ONE root — device action-send
  latency (~1 frame, free in-sim) → the policy reacts ON the obstacle/junction cell instead of PRE-acting
  on the approach cell it already observes.**
  - **FLAME slide-under = NOT a sim gap (measured, `experiments/smoke_slide_lat.py`): a prototyped
    `slide_send_lat_frames` knob was INERT — the sim ALREADY forces the pre-slide because the must_slide
    death resolves DURING the cell crossing in `_advance`, before the next step's must_slide obs updates
    (an on-cell slide is already fatal in-sim; v74 survivors pre-slide on next_must_slide 100%). The 2
    record flame deaths = airborne-into-flame (jump chaos, jumpslide territory) + a 1-frame coin flip
    (irreducible). Knob REVERTED (both sims, equivalence variant removed) — dead end caught pre-training.**
  - **TURN dead-letter = CONFIRMED real gap (`turn_death_forensic.py` on v74 @1549): junction first
    seen cp 0.46 (phase-unlucky), policy reacts cp 0.82 → write lands giL=ON, never consumed → pt-1.
    v74 CARRIES the predicted_turn cue (obs64) but trained it DEAD (no latency in v74's recipe = nothing
    to predict). The 4 wave-10/11 arms each lacked one ingredient (v71a/b/c: no cue; v72: cue but weak
    10M finetune). v75 = the UNION: v74's champion recipe (20M consol, rejump 0.46) + turn_send_lat 0.7 +
    drop/debounce → activates the cue (latency punishes late junction swipes, drop makes the approach
    pre-swipe explorable, debounce prices spam). Full 20M consolidation IN FLIGHT
    (`experiments/train_v75_turncue.sh` → runs/ppo_v75_turncue). Gate: gate_battery CREVASSE HARD (v71c
    device-eroded crevasse; the battery now catches it) → device A/B vs v74.**
    **v75 MID-FLIGHT (14M isolated turn-cfg median 1052 = v71c's 1106 EXACTLY): the predicted_turn cue
    is NOT the breakthrough — pressure(v71a/b)/drop(v71c)/exposure(v71d)/cue(v72,v75) ALL plateau
    ~1050-1200 isolated (+25-40% over the 846 floor). CONCLUSION FORMING: on-policy RL fundamentally
    struggles to DISCOVER the approach-cell pre-swipe (the counterfactual is rarely sampled, the reward
    gradient weak) — 5 arms now confirm the ceiling. v75 finishing to 20M for the record + gate battery
    (if it passes AND cuts pt-1 deaths on device, a partial gain on v74's base is still forward). ★ THE
    REAL NEXT LEVER (if v75 plateaus on device too): BEHAVIORAL CLONING from the human demos — the human
    PRE-SWIPES turns (13 human eps carry the gi_* swipe buffer = exact human turn timing). A BC-auxiliary
    loss (state→swipe-on-approach-cell, supervised) bypasses the weak RL gradient entirely. Flagged
    wave-9, never built. This is the design for the next session if v75 confirms the plateau on device.**
- **★★ WAVE 12 CANDIDATE: v74@16M = THE FIRST FULL-BATTERY GREEN (2026-07-14).** v73 (10M finetune
  w/ the rejump knob) failed stacks at n=192 (0.89-0.92) → the finetune-vs-consolidation lesson held →
  **v74 = v70's EXACT 20M recipe from a v66-obs64 base (`surgery_v66_obs64.py` → ppo_v66_obs64_base,
  0/400) + `post_land_gap_rejump_prob 0.46` in the BASE cfg** (`train_v74_consol_rejump.sh`).
  **BATTERY FIX (important): the device-true knob now lives in ALL battery configs (`DEVICE_TRUE`
  dict in gate_battery.py) — scoring a corrected-sim candidate in an uncorrected sim produced a false
  RED (v74@20M standard 0.86→0.96 once fixed). Champion baselines under the corrected cfg: standard
  2582 / crevasse 1362 / stacks 1867 / latstress 965.**
  Keep-best ladder: 20M latstress 0.90 FAIL; **@16M (`ppo_v74_consol_rejump_16003072_steps.zip`):
  standard 1.05 / crevasse 0.99 / stacks 1.14 (the chain-recovery WIN, exactly the median-band
  killers) / latstress 1.01 → ALL PASS → device A/B justified** (phone pre-cooled 32.0°C).
  A/B protocol: alternating 4-ep cooled batches vs ppo_v70_obs64_pturn, crown on pooled median ≥3130
  + the re-jump death signature (blind re-jumps into holes should drop).
  **★★★ DEVICE A/B VERDICT (2026-07-14, cooled 36°C, held all 16 eps, alternating 4-ep batches ×2):
  v74@16M POOLED MEDIAN 4531 (n=8: 1548/2698/2778/3900/5162/5221/5446/9202, mean 4494) vs v70 2984
  (n=8: 602/805/2351/2796/3172/3447/5375/9338, mean 3486) = +52% MEDIAN. v74 = THE NEW DEPLOY
  CHAMPION (`runs/ppo_v74_consol_rejump_16003072_steps.zip`, obs64, 512-net, hook 17f). This is the
  BEST device median of the project (prev best v70 ~3130) and the offline stacks 1.14 → device +52%
  correlation HELD. v74's floor lifted hard (min 1548 vs v70 602 — no early crevasse/hole dumps);
  its two upper-band deaths were the PARKED turn dead-letter (pt-1), not hole/flame chaos = the
  post_land_gap_rejump affordance is doing exactly its job. Near-record 9202m archived
  `captures/best_run_9202m_v74_hook17f.jsonl` (ender flame pt31). Both policies now hit the SAME
  tail ceiling (v70 also 9338 on a pt-1) = flame slide-under + turn dead-letter are the last two
  walls for BOTH. NEXT: the flame slide-under model (3+ record captures) is now the #1 lever for a
  10k median — the median-band killers are largely closed; the tail is flame+turn.**
- **★ WAVE 12 TARGET LOCKED (2026-07-09 night, `experiments/flame_forensic.py` 466 flame encounters +
  v70 death-tail decodes): the median killers CONVERGE on ONE root = POST-JUMP ACTION CHAOS.**
  Flame taxonomy (8 deaths/466 = 1.7%/encounter; survivors slide-at-entry 98%, onset med −0.2s):
  5/8 = AIRBORNE-INTO-FLAME (jumped, flame arrived mid-air; incl a flame→flame chain w/ a blind
  jump between); the 11,559 record-ender = slide onset +0.1s late at v195 = ONE control frame =
  the same sampling-phase coin flip as the turn dead-letter (sim already ≥ device harshness at
  entry — NOT a sim gap). v70 @2872: crevasse-jump landed → SLIDE sent at flame cp 0.09 = 1 frame
  late (survivors pre-slide −0.2s) → Burnt. v70 @3387: crevasse-jump landed → HELD JUMP 5 straight
  frames → blind/refused re-jump into pt11 → dt1. **The human exits jumps CLEAN and taps (swipe
  latency 0.000s, instant re-jumps 2.9% vs policy 5.3%, airborne hold 80.4% vs taps); the policy
  exits jumps spamming.** Flames per-se are 98% solved; turns PARKED (4 falsified approaches);
  the leverage = the post-landing action window. DESIGN QUESTION for wave 12 (fresh session):
  how to price/teach the clean-exit — candidates: (a) device-true post-landing action-refusal
  window (landing lock — measure it on device FIRST: does the game eat actions in the N frames
  after touchdown? probe like probe_jump_exec), (b) BC-aux on the human's post-landing action
  timing (the 433 gap-jumps + 152 slides are clean for this), (c) NOT reward shaping (v58/v60/v61
  all failed/gamed). Measure (a) before choosing.
  **★ (a) DEVICE-MEASURED from existing captures (`experiments/landing_lock_forensic.py`, policy
  captures, n=266 post-landing grounded JUMP attempts): the game EATS 75% OF LANDING-FRAME JUMPS —
  fire rate delay0 25% (50/202), delay1 4% (2/45), delay2 0% (0/17), delay4 2/2. The escalation fits
  a PER-LANDING LOCK DRAW (a locked landing eats ALL retries; an unlocked one fires immediately) —
  NOT per-attempt noise; consistent with 'refusals are PERMANENT in context'. If the sim fires these
  ~90-100%, the instant re-jump is a FALSE AFFORDANCE in-sim.
  **★ SIM-SIDE MEASURED (fixed harness, v70 40 eps, matched capture semantics): INVERTED — the sim
  fires landing-frame jumps 0/235 (0%!) and delay-1 retries 0/52, vs device 25%/4%. Mechanism (code):
  an airborne-decided press arrives at the landing step while `_jump_t>0` at apply time (actions apply
  BEFORE the timer decay) → the airborne guard eats it 100%; WHY delay-1 (grounded decision) is also
  0/52 is UNRESOLVED — instrument `_apply_action` refusal reasons (which guard: strafing-refusal 0.6?
  pending+jitter mistiming? sliding?) as the NEXT step. IMPLICATION: in-sim the held-jump is harmless-
  but-useless (never fires → no reward, no cost) = why the habit persists; on device 25% DO fire
  (sometimes into a hole = the re-jump deaths) and refusals then eat the NEEDED next action. The fix
  direction is now TWO-sided: (a) make the sim's landing-frame semantics DEVICE-TRUE (25% fire /
  per-landing lock draw — may require moving the timer decay or a grace re-queue), (b) THEN the
  policy can actually learn tap-precise exits. Wave-12 next session: instrument refusal reasons →
  post_land model calibrated to 25/4/0 → equivalence → v73 → gate battery → device A/B.**
  **★★ RESOLVED (instrumented + context-split): the sim's 0% = ALL 44 post-landing attempts hit the
  GAP-LIP guard (v70 only re-jumps in gap chains in-sim). Device split: delay-0 GAP-context fires
  46% (41/90) vs sim 0% — THE REAL GAP: a LANDING-frame re-jump on a wide-gap cell is a real device
  affordance (the hole-chain recovery the human uses, wave-9 ep10 pattern) that gap_jump_lip_cp 0.2
  structurally DENIES in-sim → unlearnable. Solid-context delay-0 fires only 10% on device (the
  hedge re-jump, ~consistent with strafing-refusal + landing state). WAVE-12 FIX: `post_land_gap_
  rejump_prob 0.46` — on a wide-gap cell, a jump within 1 frame of landing bypasses the lip refusal
  with p=0.46 (per-landing draw, rng-gated on prob>0). Needs `_frames_since_land` state in both sims
  → equivalence → v73 (v70 recipe + knob) → gate battery → device A/B.**
  **SHIPPED + IN FLIGHT: `post_land_gap_rejump_prob` in both sims (`_frames_since_land` /
  frames_since_land state, lip-guard bypass, prob>0-first rng), equivalence GREEN 4004/4004 +
  statistical PASS. v73 (`experiments/train_v73_landrejump.sh` → runs/ppo_v73_landrejump) LAUNCHED:
  v70@20M(obs64) init, 10M @5e-5 cosine, exact v70 mix + the knob at 0.46. Same-day warm-phone v70
  baseline `captures/ab_v70_base_r2.jsonl` (8ep @38°C, median ~2225 — thermally depressed, mechanics
  data only). Gate: gate_battery.py → cooled device A/B.**
  **v73 END-OF-DAY VERDICT (gate battery vs ppo_v70_obs64_pturn — NOTE: battery default champion
  FIXED to the obs64 artifact + missing-result now FAILS LOUDLY; the first run silently compared vs
  nothing): @10M = RED (standard 0.82 = end-of-schedule dip). **@9M = the keep-best ckpt: standard
  0.99 PASS / crevasse 0.93 PASS / latstress 1.11 PASS (a real turn-robustness gain!) / stacks 0.88
  MARGINAL FAIL (2120 vs 2399, bar 0.90)** → RED, no phone. TOMORROW'S OPENING: (1) resolve the
  marginal with n=192 stacks re-eval of v73@9M (noise band ~±6% at n=96); if ≥0.90 → cooled device
  A/B vs v70; (2) also battery v73@8M (its stacks may differ); (3) if truly eroded, next arm =
  the knob at lower prob (0.3) or +2M more anchor-heavy steps.**
- **★ WAVE 11 IN FLIGHT (2026-07-09 late): `predicted_turn` OBS (63→64) — the dead-letter fix as an
  OBSERVATION** (wave-10 verdict: pressure alone plateaus +40% and erodes crevasse skill; the missing
  piece is the cue). `obs.py predicted_turn_obs` (TURN_DEADLINE_CP 0.56, TURN_SEND_FRAMES 1.0):
  approach cell → 1.0 (swipe now = buffered, always safe), junction → deadline margin scaled 0..0.75,
  else 0 — host-derived from streamed fields, hook 17f UNCHANGED. torch `_base_obs` b+12 mirrored.
  Surgery `experiments/surgery_v70_obs64.py` → `runs/ppo_v70_obs64_pturn.zip` (0/400) = v70-exact on
  obs64 = the new deploy artifact for the current pipeline (obs60/62/63 models need their own zips).
  NEXT: equivalence re-run → v72 = v70's EXACT recipe (NO mix changes — the v71c lesson) + obs64 init
  + lat 0.7 + drop/debounce (the cue makes the deadline VISIBLE; the latency makes it BIND) →
  `scripts/validate/gate_battery.py` (MANDATORY, validated: retro-rejects v71c) → cooled device A/B vs v70.
  **v72 VERDICT (equivalence was GREEN 4004/4004; 10M @5e-5 cosine from v70@20M): FAILED — lat-0.7
  ladder 975/882/879, BELOW v71c's 1203 and barely above v70's untrained 846. The predicted_turn cue
  at conservative warm-start LR did not bite (zero-init col + 5e-5 too slow, or the swipe counterfactual
  is just too rarely sampled on-policy). FOUR falsified approaches to the pre-swipe now: pressure
  (v71a +34%), pressure+explorable latch (v71c +42% but crevasse-eroded), exposure (v71d), obs cue at
  low LR (v72 ~0). **DECISION: the turn dead-letter fight is PARKED — v70 survives ~7/8 junctions;
  marginal gain not worth more compute now. v70 REMAINS CHAMPION. The obs64 infra + gate battery +
  drop/debounce knobs stay (all equivalence-gated). PIVOT: FLAME SLIDE-UNDER model (ended ALL THREE
  record runs: 10545/11560/11731 — captures archived, `best_run_*` ×3) = the next death-model wave,
  starting with the three-capture forensic (slide timing vs flame pt31/32 entry at speed).**
- **★★ WAVE 9 (2026-07-09): THE HUMAN-DEMO PAIRED ANALYSIS (USER-directed pivot; the user recorded 13 eps /
  41 km by hand on hook 17c/d — `captures/human_20260708_*.jsonl`, human median 3054 / max 9051 on the
  EVENING throttled phone). Analyzers: `experiments/human_vs_policy.py`, `landing_analysis.py`,
  `human_death_forensic.py`, `slidegap_survival.py`. FINDINGS (human vs pooled v66 device, n=433/847 jumps):**
  - **(1) TAKEOFF TIMING = the #1 behavioral diff.** Human takeoff cp on the approach cell before a gap is
    LATE and SPEED-ADAPTIVE: med 0.52 @v130-200 → **0.65 @v200+**; the policy is EARLY and FLAT-to-INVERTED
    (0.27/0.35/**0.31 @v200+**). Consequence: on late takeoffs the policy lands ON the gap lip 20% vs human
    4.1% (human landing cp med 0.39, deliberate margin). Isolated holes forgive this (the old 7/7 probe);
    STACKS don't — early/lip landings eat the reaction window for the next obstacle.
  - **(2) The human dies on the SAME stacks that kill the policy** (9/12 real deaths decoded via the gi_*
    swipe buffer, which records REFUSED swipes): 2× SLIDE→HOLE (sliding when the gap arrives, jump swipe
    refused mid-fall), 1× JUMP→HOLE airborne-refused (@9051!), 1× HOLE→HOLE chain re-jump refused (@4650),
    1× jump-lands→TREE instantly (@2882), 1× POST-TURN HOLE no-reaction (@4802, lvl3 v210), 1× SLIDE→missed
    turn, 1× crevasse cap graze kill (@1277), +3 fumbles. The stack taxonomy is CONFIRMED as the frontier —
    the human's edge is TIMING MARGIN, not a different strategy.
  - **(3) SLIDE→HOLE = the one UNMODELED stack** (`slidegap_survival.py`, 14 events): slide commits ~0.73s
    (sim's 0.7 ✓, and sim already refuses jump-while-sliding ✓ sim_env.py:1271); human survives spacing
    ≥1.1s via slide-end→IMMEDIATE jump (takeoff −0.33..+0.07s rel. hole entry); spacing <1.1s or a missed
    instant re-action = death (2/14 died; ep0 survived 1.20s spacing, ep2 died at 1.20s = the human reaction
    limit). Generation NEVER makes it (min_space 6; jumpslide is the REVERSE order) → free in-sim → no
    signal. **NEXT KNOB: `slidegap_prob` compound (slide obstacle 1-2 clear cells BEFORE a hole), the
    jumpslide pattern reversed.**
  - **(4) Policy holds JUMP on 80.4% of airborne frames** (human taps: swipe→takeoff latency med 0.000s);
    policy instant re-jumps (≤0.15s grounded) 5.3% vs human 2.9% (chains 8.7% vs 6.3%); policy stumbles
    0.33/1000m vs human 0.12 (3×). Long-hole per-encounter death rate: human 0.70% vs policy 1.16%.
  - **(5) COINS ARE RECORDED after all** — the cell-level fields work (`has_coins/coin_loc_start/end/
    coin_in_air/coin_count` + next-cell); only `t_last_coin` (0x248) is broken-monotonic. **75% of the
    human's big lane excursions (|lane|>4.5) have coins on cell/next** → the user's tilt-for-coins claim is
    CONFIRMED and quantified; lateral human data is coin-contaminated for BC, but jump/slide TIMING is clean.
  - **DERIVED PLAN: (a) add `slidegap_prob` (both sims, equivalence-gated) to the v70 consolidation base;
    (b) the takeoff-timing gap is a TRAINING target, not a sim gap — candidate: BC-auxiliary on the 433
    human gap-jumps (state→jump-timing pairs; obs reconstructable from capture fields) or a takeoff-cp
    prior; (c) v70 (`experiments/train_v70_consolidated.sh`, staged: v66-obs62 base, 20M, penalties STRIPPED
    per the v69 signal, all scenario knobs + anchored profiles) is the vehicle — add slidegap before launch.**
  - **★ WAVE 9 IMPLEMENTATION (2026-07-09, same session): ALL SHIPPED + GATED.**
    - **`slidegap_prob` compound (both sims)**: slide obstacle → `slidegap_gap_cells`(2) clear →
      `slidegap_hole_cells`(1) hole; single hole mirrors normal gen (OBST_HOLE on straight, non-wide;
      >1 → pt12+wide like jumpgap; torch o_wide now type-gated on 12). Trigger = any must-slide emit
      (normal + turnobstacle placements), prob>0-first rng gating, rides `_pending_post_gap`/pend_postgap
      threshold mode. Smoke-verified both sims; new `slidegap`/`slidegap_flame` equivalence tracks.
      **Equivalence 3718/3718 GREEN + statistical PASS; scorecard GREEN.**
    - **`next2_narrow_lane` obs 62→63 fork (the jump→narrow MID-AIR TILT fix)**: USER-decoded — the human
      tilts DURING the jump toward the coming strip; `narrow_target_lane` fires only 1 cell ahead
      (sim_env.py ~1970) so at takeoff the policy is structurally BLIND to the side → the observed
      last-second strafe (v54's jumpnarrow trained the scenario; THIS obs gap kept the commit late).
      Plus the reward strafe-dock (reward.py:207) taxed early tilts — already dead in v70 (idle 0).
      Added end-to-end: obs.py (extra field, `narrow_lane_to_obs` encoding, junction dropout), both sims'
      snapshot, torch `_base_obs` b+11, constants STATE_DEFAULTS, hook **17f** (cachedNext2PathType +
      narrowStrafeDirFor on the 2-ahead cell), equivalence diff sentinel. Surgery
      `experiments/surgery_v66_obs63.py` → `runs/ppo_v66_obs63_n2nl.zip` (0/400 mism) = v66-exact on
      obs63/17f = the deploy artifact for the current pipeline.
    - **⚠️ 17e HOOK BREAKAGE (found by the wave's own capture failure): 17e used an UNDEFINED `readS32`
      → the snapshot threw EVERY frame → ZERO state stream ("device degraded" with a live injector +
      good forwards). FIXED in 17f (`readI32` + renamed `coin_count_total` @0x148 so the per-cell
      `coin_count` from PE_CoinRunCount is not clobbered — BOTH coin signals now stream). Verified
      streaming on device. LESSON: smoke-test a staged hook edit on-device before relying on it.**
    - **v70 TRAINED + OFFLINE-GATED (2026-07-09): the consolidation PAID.** 20M @4.8-6.2k steps/s (~60
      min, 512-net, GPU 803MiB/59% — headroom for a 2nd concurrent run). NO late collapse (cosine
      8e-5→5e-6 + 45% standard anchor): the BEST checkpoint = the FINAL one.
      **Offline device-true cfg (n=96): v70@20M median 3002 vs v66-obs63 2550 = +17.7%** (mean 3223 vs
      2717, max 7365; Tree deaths 28→20). **Stack-heavy cfg (slidegap .4 / jumpnarrow .3 / jumpgap .3 /
      jumpslide .2 / turnobstacle .12): 2399 vs 2232 (+7.5% med, +13.5% mean), Burnt dt5 6→1** = the
      airborne-into-slide-obstacle stack skill directly visible. Weights `runs/ppo_v70_consolidated_
      20000768_steps.zip` (obs63, 512-net, hook 17f).
    - **★★ DEVICE A/B (cooled-phone alternating 4-ep batches ×2 rounds, thermal protocol observed:
      cooled to 36.0°C before each round): v70 POOLED MEDIAN 3130 (n=8: 702/1731/2455/2872/3387/4733/
      5632/5995) vs v66 2603 (n=8: 601/613/1182/2541/2665/3227/6142/11560) = +20%, CONSISTENT with the
      offline +17.7% — offline↔device correlation HOLDS. v70 = THE NEW DEPLOY CANDIDATE.**
      Death mixes: v70 = 3× hole pt11 + 1× long-hole + 2× flame + 1× crevasse@5632 + 1× turn pt-1;
      v66 = 3× crevasse (two at 601/613 EARLY) + 2× long-hole + 1× narrow + 2× flame. v70 had ZERO
      narrow deaths + ZERO early crevasse deaths.
      **★★ NEW ALL-TIME RECORD: 11,560 m — v66(obs63) on hook 17f, A/B batch 4 ep3, died Burnt on a
      flame tower pt32 (`captures/best_run_11560m_v66_hook17f.jsonl`). Beats the 10,545 goal run. The
      flame slide-under at speed remains the record-run ender (same as 10,545) = still the #1 unmodeled
      ceiling for the tail; the median ladder = v70's territory.**
      Same-day fresh-phone v66-obs63 control batch (`captures/ab_v66_wave9_r1.jsonl`, 8ep, 35°C start):
      605/805/1347/2585/2844/2926/3624/4190, median ~2715; deaths incl 2× narrow pt24 (the jump→narrow
      theme live), 1× long-hole pt16, 1× crevasse pt35, 1× flame pt32.
      SESSION OPS NOTE: mid-A/B the adb forwards dropped AGAIN (empty `adb forward --list`, established
      sockets kept flowing → state stream wedged half-open, capture "degraded" mid-run); recovery =
      re-forward + force-stop/restart + re-inject (the hook's broadcast thread does not survive its
      client socket erroring). One v70 round-2 ep was lost to it (batch resumed as `ab_v70_r2b`).
  - **★ WAVE 10 (2026-07-09 evening): TURN DEAD-LETTER fix — v71 twin arms IN FLIGHT.** Death decoded
    frame-exact (`experiments/turn_death_forensic.py`, ab_v70_r2b ep0 @4733): the policy turns ONLY on the
    junction cell; phase-unlucky first sighting (cp 0.52 @v209) → TURN write lands cp 0.92, `giR=ON` but
    never consumed → pt-1. v70 never pre-swipes on the approach cell (0/4 junctions in the tail) because
    late junction swipes are FREE in-sim (decision-cp gate 0.56, no latency).
    **⚠️ CALIBRATION CORRECTION (adversarial-review subagent + measured sweep): the documented
    "turn_send_lat 0.4 = pressure" was WRONG-arithmetic — in-sim a first-frame reaction decides at
    cp < v·dt/cell ≈ 0.35 @v210, so lat ≤ 0.4 NEVER crosses the 0.56 gate. SWEEP (v70@20M, n=96/dose):
    lat 0.4 median 3002 = EXACTLY lat-0 (INERT); 0.7 → 846 (binds, 86/96 turn deaths); 1.0 → 272
    (96/96). The review saved 2×20M inert training steps.** Design doc:
    `docs/design/2026-07-09-turn-deadletter-fix-design.md`.
    Arms (v70 recipe + ONE knob, init v66-obs63, 20M each, concurrent): **v71a lat 0.7** (binding
    pressure w/ survival gradient), **v71b lat 1.0** (device-plausible; pre-swipe or die). Gates:
    standard cfg parity vs v70 + **lat-1.0 stress eval** (junction-reactor floor = 272; a pre-swiper is
    latency-immune → arm's lat-1.0 ≈ its lat-0 distance) → cooled-phone device A/B vs v70 (turn pt-1
    deaths + median ≥ 3130).
    **MID-FLIGHT FINDING (5M/4M): both arms PINNED at their floors (794/307) — the pre-swipe is not
    being discovered. Cause = the critic's RISK 3: the approach-latch (safe, exists in-sim via
    turn_approach_buffer) is SURROUNDED by the turn-on-straight=FATAL rule, so exploration toward it is
    killed. AND that fatal rule is STALE fidelity: measured 2026-07-02 under DIRECT fn.TurnLeft — hook
    16o (2026-07-04) rerouted turns through HandleDelayedTurning, and the 2026-07-05 probe MEASURED the
    hook DROPPING off-junction turn bytes (debounce 0.3s / invalid-dir / not-near-junction; 'pre-onset
    turn bytes: 0 in all captures'). On TODAY'S device an off-junction turn byte is a harmless no-op;
    in-sim it's death. PLAN: (1) hold the 8M early-stop line on v71a/b; (2) build knob pair
    `turn_offjunction_drop` (off-junction turn = silent drop, not kill, when hook_turn_buffer on) +
    `turn_drop_debounce_s` 0.3 (ANY turn action arms a debounce; a junction/approach swipe inside it is
    EATEN = the real device spam-price) — both sims, default off, equivalence-gated; (3) v71c = v70
    recipe + lat 0.7 + drop&debounce → the latch is explorable and spam still costs. If v71a/b stay flat
    at 8M → kill v71b, keep v71a to 20M as control, launch v71c on the freed GPU.**
    **EXECUTED: knobs shipped both sims (`turn_offjunction_drop`/`turn_drop_debounce_s`, sim_env.py
    _attempt_turn + torch mirror w/ turn_debounce tensor), equivalence GREEN 4004/4004 + statistical
    PASS (new `turn_dropdeb` variant: drop/eaten/latch/dead-letter parity). v71b killed @9M (pinned
    377); **v71c (`experiments/train_v71c_latdrop.sh` → runs/ppo_v71c_latdrop) launched** in its slot;
    v71a continues to 20M as the pressure-only control (pinned ~800-900 through 11M = the barrier
    story confirmed). Gate for v71c: standard-cfg parity vs v70 + lat-1.0 stress (junction-reactor
    floor 272) + lat-0.7 (v70 floor 846) → device A/B.
    **RESULTS SO FAR (isolated evals — the POOLED training curve HIDES turn progress, always eval the
    ckpt under its own lat cfg): v71a@20M = lat-0.7 median 1136 (+34% over v70's 846 floor; Falls 85/96
    still turn-misses) at NO standard cost (2848/3442 vs v70 3002/3223) — pressure alone climbs the
    barrier SLOWLY. v71c@7M = 1014, @14M = 1106 (reached the control's 20M endpoint 6M early, max 6156,
    Falls 73/96 — but growth flattening: grind, not breakthrough). v71c runs to 20M (keep-best);
    **v71d (`experiments/train_v71d_turndense.sh` → runs/ppo_v71d_turndense) launched IN PARALLEL** =
    v71c + a 12% TURN-DENSE profile (turn_prob 0.6, min_space_between_turns 4; anchor 45→38, gauntlet
    15→10) — the wave-3 curriculum trick aimed at pre-swipe SAMPLING. Endgame: best arm must hold
    standard parity vs v70 AND beat it decisively under lat stress → cooled-phone device A/B vs v70
    (turn pt-1 deaths + median ≥ 3130). If all arms plateau ~1100: consider the pre-swipe may need
    obs support (e.g. dist-to-junction scalar) or accept the partial gain + move to the flame model.
    **v71c FINAL: best ckpt = @18M (`ppo_v71c_latdrop_18001920_steps.zip`): lat-0.7 median 1203 (+42%
    over the 846 floor; ladder 16M 965 / 18M 1203 / 20M 1187) AND standard-cfg 2939/3208 ≈ v70 parity
    (3002/3223). = v70-equivalent skills + meaningfully better latency-turn robustness, but turn misses
    still 71/96 of its lat-cfg deaths — PARTIAL, not the pre-swipe breakthrough. v71d (turn-dense) is
    the last arm; then verdict.**
    **v71d@9M isolated eval: 852 = AT the floor, BEHIND v71c's pace (1014@7M) — the turn-dense
    exposure hypothesis is NOT accelerating pre-swipe discovery (profile dilution suspected). Running
    to ~14M for one final check (GPU otherwise idle), else killed. Converging verdict: v71c@18M =
    the wave-10 winner (v70 standard parity + 42% latency-turn robustness = weak dominance → deploy
    candidate), with the honest residual: the FULL pre-swipe habit likely needs OBS support (a
    dist-to-junction / predicted-turn-landing style scalar for the approach cell — the predicted_jump
    pattern applied to turns) as the next lever, OR accept the partial and pivot to the flame
    slide-under model (the 10,545 + 11,560 record-enders).**
    **v71d@14M: 1103 = v71c's 14M mark EXACTLY — all arms plateau ~1100-1200 (+40%); the turn-dense
    exposure hypothesis is SETTLED (no acceleration; profile dilution ~neutral).**
    **★★ WAVE-10 DEVICE VERDICT (cooled 36.0°C same-session A/B): v71c@18M DEVICE-REGRESSED — batch
    611/597/3618/592, 3/4 deaths = EARLY CREVASSE (pt34/35/36 @~600m, the pre-v59 signature): 20M under
    the turn-pressure mix ERODED the crevasse takeoff timing, and the offline standard-cfg parity
    (2939≈3002) did NOT catch it (crevasse exposure too sparse offline — the v64-era drift lesson
    re-confirmed at consolidation scale). v71c REJECTED. v70 control batch same session:
    2661(pt-1)/4236(pt-1)/1971(tree)/**11731** — median ~3448, v70 performing at level. **v70 REMAINS
    THE CHAMPION.**
    **★★ NEW ALL-TIME RECORD: 11,731 m — v70 on hook 17f (`captures/best_run_11731m_v70_hook17f.jsonl`,
    crown batch ep3), beating the morning's 11,560 (v66). Ender = pt3 turn. NOTE: 3/4 of v70's crown-batch
    enders were TURN-related (2× pt-1 dead-letter + the pt3) — the dead-letter is CONFIRMED as v70's #1
    ender; the wave-10 training lever failed to fix it on device, so the NEXT lever = the obs support
    (approach-cell dist-to-junction / predicted-turn cue) inside a FULL consolidation (all skills incl
    crevasse-field anchored), NOT a pressure-mix variant. The drop/debounce fidelity knobs stay valid
    infrastructure (equivalence-gated, default off).**
- **★★ WAVE 6 (2026-07-08 night): POST-TURN OBSTACLE (v65) + 512 SCALE-UP (v66) + THE THERMAL DISCOVERY.**
  - **★★ THERMAL THROTTLE = the evening device collapse decoded (dumpsys battery/thermal): the phone at
    39-43°C after ~14h play-while-charging → frame-rate throttle → ALL policies collapse 2-3× (same v59:
    4220 morning vs 1434 evening; monotonic decline batch after batch). App restart does NOT fix it; ONLY
    COOLING does (game stopped + screen off: 39→36°C in ~20 min → performance RESTORED). The phone re-heats
    36→38°C in ~4 episodes → OPERATIONAL RULE: cross-model A/Bs must be SHORT ALTERNATING batches (4 ep/arm)
    on a cooled phone; check `dumpsys battery | grep temperature` (≤360 good, ≥380 throttled) before evals.**
  - **v65 = v64 recipe + `turnobstacle_prob` 0.08 (POST-TURN OBSTACLE, device-measured: obstacles on the
    FIRST post-junction cell in ~9% of turns — 72 onsets/6 captures; min_clear_after_turn(2) meant the sim
    NEVER generated it → 7/15 of pt(-1/1/2) deaths followed a turn by <12 frames, ending the 4-7k runs;
    both sims + `turnobstacle` equivalence track, 3380/3380 GREEN). Offline: +9.2% own env, +2.5% v64 cfg.**
  - **v66 = v65@3.5M surgery-widened 256→512 (`surgery_v65_512.py`, 0/500 mism) + 6M cosine 1.2e-4→1e-5 on
    the v65 recipe (the v57 capacity lesson): offline 2140 vs v65 1982 (+8%) = OFFLINE CHAMPION. COOLED-PHONE
    A/B (alternating 4-ep batches): v66 pooled median 3737 (n=8, ZERO crevasse deaths, max 6236) vs v62 3253
    (n=4, max 7368) — v66 leads.** Weights `runs/ppo_v66_512_5500928_steps.zip` (@5.5M = best ckpt).
  - **REMAINING WALL (fresh cool-phone forensic): HOLES = 41% of the day's deaths; the residual root = the
    HELD-JUMP reflex (device held% still ~83): (a) idle hedge-jumps landing ON an unseen hole's lip (grounded
    past-lip → refused → dt1), (b) landing-frame re-jumps into chained holes (the jumpgap scenario — trained,
    improved, not eliminated). Contributing constraint: obs lookahead 2 cells < jump span ~2.8 cells at speed
    → the policy hedges. Levers: next3-obstacle obs (a fork, obs 60→62 + hook), or a next-gap-distance
    scalar obs. **v67 (= v66@5.5M + jumpgap 0.3 + turnobstacle 0.12 pressure-bump) FAILED its own gate
    (1854 vs v66 2021 in v67's OWN cfg) — the v49/v50 lesson re-confirmed: MORE scenario pressure ≠ better;
    0.15/0.08 is the right exposure. DISCARDED.**
  - **★ FINAL NIGHT BENCHMARK (v66@5.5M, coolest window 34°C, 6ep): median 4073, THREE 6k+ runs (6019/6592/
    6706, enders: flame, flame, turn dead-letter). Pooled v66 device n=14: median ~3737, 4 runs >6k, ZERO
    crevasse-graze regression. v66 = THE DEPLOY CHAMPION (`runs/ppo_v66_512_5500928_steps.zip`, 512-net
    obs60, hook 16z). Reliability ladder to 10k median from here: (1) the held-jump/hole reflex (next3 or
    gap-distance obs = the likely key), (2) turn dead-letter at 5-7k (hook-side or gentle-lat retrain),
    (3) flame slide-under at speed (the 10545 ender, still unmodeled).**
  - **★★ USER INSIGHT (live, late night): "the bottleneck is ROBUSTNESS when obstacles STACK — the model
    reaches 10k+ when generation is kind; it struggles to be creative in stacks; may partially come from
    the penalties." CODE-CONFIRMED — the PENALTY-CONFLICT discovery: `slide_jump_penalty` 0.15 docks a JUMP
    whenever must_slide/next_must_slide/is_sliding — which is EXACTLY the legit early jump of a jump→slide
    STACK (the v53 compound: jump the gap early while the slide looms next_must_slide=True). The death-model
    says jump-now-or-die; the shaping says −0.15 for jumping → CONFLICTING GRADIENTS precisely on stacked
    obstacles. Plus train_fast's IMPLICIT `--idle-action-penalty` 0.05 (default, in every recipe) taxes
    pre-jumps beyond the flag horizon. These penalties were pre-death-model scaffolding; the sim now prices
    real mistakes with real deaths. TWO-ARM RESPONSE + VERDICTS (both equivalence 3380/3380 GREEN):
    - **v68 = next3 obs fork (obs 60→62, hook 16z→17a; surgery `surgery_v66_next3.py` 0/400): NO offline
      gain (best ckpt 2078 vs the surgery base 2140) — the boolean at cell+3 didn't pay within 4M (likely
      partially redundant with next2_path_level / not the binding constraint). SHELVED as a policy; the
      obs62/17a INFRASTRUCTURE stays (committed) for future lookahead work.**
    - **v69 = PENALTY-STRIP (slide_jump 0.15→0 + idle 0.05→0, from the obs62-widened v66): median parity
      (2059 vs 2140) but the TAIL stretched — max 6727 vs 4011, the longest offline run of the night =
      directionally the USER's "creativity → survive stacks → run long" signature, but max-on-n=96 is
      noisy. NOT crowned; the penalty-strip idea stays OPEN (candidate: strip penalties INSIDE the next
      big consolidation train rather than as a finetune).**
    - **⚠️ DEPLOY NOTE (the obs62 fork is now in code): obs60 models (v66 et al) need the surgery-widened
      artifact to run on the current pipeline — `runs/ppo_v66_obs62_next3.zip` = EXACTLY v66's behavior
      (zero-weight new cols, 0/400) on obs62/hook 17a. Deploy THAT for v66.**
- **★★ WAVE 5 (2026-07-08 evening): RHYTHM CHAIN (v63) + the ANTI-DRIFT CONSOLIDATION (v64). USER-driven.**
  - **★ USER INSIGHT (measured, the wave's driver): "mid-to-high still hits the edge" = SEQUENTIAL-FINETUNE
    DRIFT.** Device graze at lvl-3 caps: v59 19% → v62 26-29% → v63 32% (back to v51 baseline); arc
    clearance +3.09 → +0.00. Each isolated 2M finetune optimizes its new scenario while crevasse exposure
    in standard generation is too sparse to defend the timing. **ROOT CAUSE = TRAINING (not sim/reward).
    LESSON: the one-change-per-retrain discipline conflicts with skill RETENTION — once 2+ scenario skills
    matter, they must be trained TOGETHER (anchored mix), not stacked.**
  - **v63 = v62 + `jumpgap_chain_max` 3 (RHYTHM CHAIN: the tail becomes [clear×2, hole×2]×n, n~U(1..3);
    torch = periodic phase emission via `pend_postgap_total`, equivalence 3211/3211 GREEN incl the new
    `jumpgap_chain` track).** Device forensic ground: ab_v62_r1 ep0 @794 / ab_v60_r2 ep4 @1481 = high-ground
    long-holes chained every ~4 cells; the seamless held re-jump arcs into hole #2 (the USER-observed
    "double-jump when the first landing is close to the edge"). Offline: +12.6% chain cfg, parity v62 cfg.
    **DEVICE: median 2180 vs v62 control 2665 = NO win (its crevasse had eroded further). SHELVED as a
    policy; the chain KNOB is kept (it's in v64's base).**
  - **★ v64 = v59@1.5M + the WAVE-3-VALIDATED anchored curriculum + ALL scenario knobs in base (jumpgap
    0.15 + chain 3 + b2b 6), 4M cosine 8e-5→1e-5 (`train_v64_anchored_mix.sh`): profiles 45% standard
    ANCHOR / 15% gauntlet / 12% crevasse-field / 12% rhythm / 10% compound / 6% narrow. OFFLINE = BEST OF
    ALL: 2152 crevasse-cfg (v59 2017, v62 2027) AND best sim graze 23% (v59 24, v62 26) — the mix DEFENDED
    the crevasse skill while adding rhythm.** Best ckpt @3.5M (`ppo_v64_mix_3502080_steps.zip`). DEVICE
    (same-session pair, depressed evening phone): v64 median 2161 / mean 2522 vs v62 2327/2268 = TIE;
    graze n too small to read (13-19 encounters/batch). **STATUS: v64 = the deploy CANDIDATE on offline+sim
    evidence; NOT yet device-crowned. ⚠️ PHONE-STATE: all afternoon/evening medians ran 2-3× below the
    fresh morning (same v59: 4220 morning vs 1434 evening) — cross-session medians are MEANINGLESS.**
  - **IMMEDIATE NEXT ACTION (next session): FRESH-PHONE morning 3-way A/B v64@3.5M vs v62@1.5M vs v59@1.5M
    (8ep each, alternating) + `analyze_crevasse_device` graze on each → crown the champion. Then the
    remaining killers by mix: turn dead-letter (pt-1/1/2 ends most good runs now — the hook-side pre-swipe
    idea from the v52 entry, NO retrain), flame slide-under, turn→immediate-hole (survivability unknown).**
- **★★ WAVE 4 (2026-07-08): v51 → v59 CREVASSE fix = device-validated WIN; then the HIGH-GROUND DOUBLE-JUMP:
  reward-shaping failed twice (v60/v61) → DEATH-MODEL (jumpgap) → v62 = the targeted death mode FIXED on
  device (re-jump deaths 0/46, frequency 17→5.75/ep) + same-session ≥ v59. USER-driven live-watch loop.**
  - **★ v62 = v59@1.5M + `jumpgap_prob` 0.15 (the GAP→GAP death model). TARGET FIXED ON DEVICE.**
    Forensic first (`experiments/rejump_lethality.py`): device = ~17 landing-frame re-jumps/ep, ~2% lethal
    each → **35% of device episodes END within 15 frames of a re-jump**; sim (v59 cfg) = 0.8/ep, **0/47
    lethal** — `min_space_between_obstacles: 6` guarantees a clear landing (a jump spans ~2.2-2.8 cells) so
    the blind re-jump was STRUCTURALLY free in-sim = no learning signal (why shaping kept failing). FIX =
    `jumpgap_prob` compound (both sims, `_pending_post_gap` tail like jumpslide): after a wide-gap, 2 clear
    cells then a 2-cell HoleLong; a blind landing-frame re-jump lands INSIDE gap-2 (landing model → dt4, the
    device signature), land→run→deliberate-jump-at-the-lip clears. Torch: hole payload rides the nlane==0
    generic emit + `o_wide` marked for OBST_HOLE payloads; `queued|jn` one-tail rule. With it ON, v59's sim
    re-jump lethality 0→2.6% ≈ device. **Equivalence 3042/3042 GREEN** (new `jumpgap` injected track ×12
    variants ×14 policies). v62 offline: +4.4% in its training env, NO regression (2027≈2017 crevasse-cfg).
    **DEVICE (8ep, `ab_v62_r1`): re-jump deaths 0/46 events (v59: 7/345), re-jump freq 17→5.75/ep, hi-ground
    jump-deaths 2/8 (v59 3/6). SAME-SESSION control (`ab_v59_r2`, mandatory per the same-session rule):
    v62 median 1845 vs v59 1434 — TIED-leaning-v62 (n=8, overlapping).** NOTE the huge PHONE-STATE effect:
    the SAME v59 scored median 4220 on the fresh morning phone vs 1434 in the depressed afternoon session —
    afternoon medians are NOT comparable to morning ones; only same-session pairs count. **VERDICT: v62 =
    new deploy candidate (`ppo_v62_jumpgap_1503232_steps.zip`) — beats v59 on the targeted death mode,
    ≥ v59 overall. Death mix now: plain low-speed holes (600-1300m) + turn dead-letters + flames = the
    remaining killers.** Scripts: `train_v62_jumpgap.sh`; knobs `jumpgap_prob/_after_distance/_gap_cells/
    _hole_cells` (default 0/off).
  - **v59 = v51 + CREVASSE TAKEOFF-VARIANCE fix (crevasse_graze_worldy + cap_arc_offset 0.4 +
    crevasse_takeoff_std 0.2). DEVICE-VALIDATED.** Device capture (ab_v59_r1): lvl-3 crevasse graze
    26%→19%, arc clears the cap HIGHER (+0.00→+3.09). **USER (live): "the mid to high ground transition is
    almost mastered."** The #1 median killer (crevasse ~610m) largely closed. Offline: v59@1.5M = the sweet
    checkpoint (crevasse +8%, standard +3.2%, NO regression; @2M over-specialized = the drift-past-1M
    lesson). See the CREVASSE entry below for the full mechanism + why v55 failed (fixed offset = sharp
    all-or-nothing; needed the takeoff VARIANCE for a clean partial rate).
  - **★★ v60 = v59 + JUMP-HOLD fix (jump_hold_penalty 0.12). THE DOUBLE-JUMP FIX — TRANSFERRED TO DEVICE
    (where v58's airborne penalty totally failed).** USER-observed (live, watching v59): "on HIGH-GROUND it
    jumps AFTER having jumped an obstacle → jumps into its death. Only on high-ground." DECODED (ab_v59_r1
    ep3/ep4 dt=4 Ledge): the policy HOLDS jump; the instant jump-1 lands, the still-held press fires a
    SEAMLESS jump-2 (device is_jumping never drops between them) whose arc descends into the next step-down
    crevasse / long-hole packed in the high-ground zone → Ledge/Fall. Only high-ground (flat low-ground = the
    stray jump lands on solid ground). WHY v58's airborne penalty MISSED it: in-sim jump-1's landing leaves a
    1-frame is_jumping=False gap so the landing-frame re-jump wasn't docked → the policy learned it's free
    (it is NOT on device where the jumps overlap seamlessly). FIX = jump_hold_penalty: dock a JUMP-after-JUMP
    (SLIDE-after-SLIDE), keyed on `last_action` (the policy's OWN action stream, IDENTICAL sim↔device → it
    TRANSFERS). **PARTIAL/FRAGILE — the double-jump is NOT cleanly fixed (2 reward-shaping attempts failed,
    as the docs warned; deploy champion stays v59):**
    - **v60 (`jump_hold_penalty` 0.12, CONSECUTIVE JUMP-after-JUMP): GAMED + REGRESSED.** Device consecutive-
      held 82.8%→2.7% (looked like a win) BUT the policy EVADED it with 1-frame NoOp gaps (JMP,-,JMP) → still
      double-jumps on high-ground (ab_v60_r2 ep4 dt=4, decoded) AND over-corrected into hole deaths →
      **device median 4220 (v59) → 2285 (v60), whole distribution shifted DOWN.**
    - **v61 (WINDOWED [rejump within jump_hold_window=6 frames, un-gameable by gaps] + GATED on must_jump):
      the GATE BACKFIRED.** Gating "don't dock a jump AT an obstacle" (to protect the legit jump + dropped-
      jump retry) let ~all re-jumps through (obstacle-dense zones = must_jump often True) → v61 rejump(win6)
      84.5% ≈ v59's 86.5% (barely curbed). Distance fine (offline 7764) only because behaviour barely changed.
    - **OPEN LESSON: the ~85% "rejump" is MOSTLY harmless airborne HOLDS (dropped presses); the lethal event
      is specifically the LANDING-frame NEW jump (starts a 2nd airtime). The metric conflates them, and
      shaping keeps either missing the lethal press (gate) or over-correcting (ungated). NEXT = the PROVEN
      death-modeling method: verify the sim does NOT kill v59's high-ground double-jump (survives in-sim = no
      learning signal), then MODEL it — a high-ground obstacle→step-down/hole compound so the 2nd jump lands
      short → Ledge (via the existing wide_gap_landing/apex models) → retrain → device.** Reward tooling
      (jump_hold_penalty windowed+gated, `_jump_recency` in both sims, `measure_jumphold`, `device_jumphold`,
      `decode_highground_death`) stays for reuse. reward.py + both sims' _shaping + train_fast + equivalence
      shaping variant. **DEPLOY CHAMPION = v59 (`ppo_v59_crevasse_1503232_steps.zip`, device-validated). All
    weights gitignored/local; UNCOMMITTED per norm.
- **★ WAVE 3 (2026-07-07 late night): CURRICULUM + SCALE + the jump-spam wall.** New tooling all committed,
  equivalence GREEN throughout. USER-driven.
  - **v55 crevasse geometry fix (`crevasse_cap_arc_offset` + world-Y): DEVICE-DISPROVEN offline** — mild
    regression (dynamics change, sharp threshold, no clean partial rate; needs takeoff-VARIANCE not just an
    offset). Knob committed default-off. Crevasse still the #1 early killer (~600m); UNSOLVED.
  - **★★ 2026-07-08 CREVASSE TAKEOFF-VARIANCE FIX → v59 (the v55 completion). DEVICE-MEASURED + calibrated,
    equivalence GREEN, RETRAIN pending.** Fresh v51 device capture (`captures/v51_crevasse_16z.jsonl`, 6ep,
    hook 16z, `experiments/analyze_crevasse_device.py`): v51 grazes high-ground (level-3, cap ground_height
    15) crevasse caps **26% (n=19), 0% kill**; level-2 caps **0%** — CONFIRMS the documented ~32%. WHY v55
    FAILED (now decoded): with `crevasse_graze_worldy` alone the cap sits near v51's jump APEX in-sim
    (arc_y~19.2 >> cap 15, and the parabola is FLAT there) → clears 100%; a FIXED `cap_arc_offset` moves the
    cap onto the descending arc but v51's in-sim takeoff is near-deterministic → SHARP all-or-nothing (no
    partial rate). FIX = NEW `crevasse_takeoff_std` (both sims): a per-crevasse Gaussian shift of `d`
    (takeoff→cap cells) so the descending-arc clearance SPREADS around the cap top → a FRACTION grazes.
    Gated default-0 (no RNG draw → deterministic-suite stream untouched; follows the jump_exec_lat_jitter
    pattern, drawn from self._rng / torch.randn(gen)). Calibrated in-sim (`experiments/sweep_crevasse_
    variance.py`): **offset 0.4 + std 0.2 → v51 grazes 32% @lvl3 / 0% kill / lvl2 1% — matches device**
    (errs slightly harsh = the safe direction). Equivalence **DETERMINISTIC 2873/2873 GREEN** (incl the
    crevasse_graze variant; the std>0 path is matched-distribution → statistical-tier). v59 = v51 + ONLY
    these two knobs, isolated (no dense profiles, 5e-5, per the v48/v52 one-change lesson);
    `experiments/train_v59_crevasse.sh`. MIND v52/v55: v51 is a SHARP optimum, crevasse retrains regressed
    before → gate HARD on device A/B. **SESSION NOTE: during the v51 capture, ep5 hit 10,389m (v51, hook
    16z) — but run-luck: the 6ep set was 1044/2475/5709/6470/6538/10390, median ~6090 (cool phone). NOT
    reliable 10k; the crevasse/hole early-killers still cap the median (ep0 died @1044 on a hole; the 10k
    run also died on a hole).**
  - **★ 2026-07-08 JUMP-TIMING PROBE = the jump-spam wall is NOT a fidelity gap (see the JUMP-SPAM WALL
    entry below for the full refutation): device airtime 0.767s ≈ sim 0.80s, obs FRESH, airborne presses
    DROPPED (not queued), arc peak 19.17 = sim 19.2. Jump model FAITHFUL; airborne penalty solved a non-
    problem for device. The 6813m "random jump" death = a spurious GROUND jump → airborne into a must-slide
    (the jump→slide death, v53's scenario), a policy habit not a timing gap. No jump_duration recal.**
  - **★ USER INSIGHT (the big one): "modelled correctly but TRAINED incorrectly."** `max_back_to_back_
    obstacles=2` CAPPED chains → the sim literally never made a GAUNTLET; training was sparse (min_space 6).
    FIX = raise b2b to 6 + `episode_profiles` MIX (torch-only, per-reset sample): 45% standard / 15% dense-
    gauntlet / 12% crevasse-field / 12% long-hole-rhythm / 10% compound-dense / 6% narrow-heavy. **v56 =
    v54 + curriculum: best-of-session STANDARD median 2930 (+16% vs v54), NO collapse (45% anchor held).**
    Gauntlet-training generalised to better all-around robustness. CURRICULUM VALIDATED.
  - **LR SCHEDULE added to train_fast.py** (`--lr-schedule cosine|linear --lr-final`; tensor-LR in the
    CUDA graph so it changes without re-capture). Constant-high collapses a warm-start, constant-low
    undertrains; cosine 1.2e-4→1e-5 does both.
  - **512-NET surgery** (`surgery_v56_512.py`, function-preserving top-left-block widen, 0/500 mism).
  - **v57 = 512-net + curriculum + cosine LR + 4M = BEST OF SESSION OFFLINE: standard 3124 / challenge
    1912 (@3.5M, `ppo_v57_512sched_best.zip`).** All 3 user levers (capacity+schedule+curriculum) paid off.
    DEVICE (6ep): median 4111 / max 6813 ≈ v51 (user: close, don't retest v51).
  - **★ JUMP-SPAM WALL (USER-observed on the v57 6813m run + forensic): the policy presses JUMP ~CONTINU-
    OUSLY.** ROOT: the sim IGNORES jump-while-airborne (`_apply_action` `_jump_t<=0` guard) so spam is FREE
    in-sim; the idle penalty checks OBSTACLE context not AIRBORNE STATE. On DEVICE each press queues a
    re-jump on landing → perpetual airborne → mistimed hole takeoff → clip (dt4). (USER nailed the asymmetry:
    the penalty works for STRAFE because random strafe self-punishes via lane/cliff death; jump-spam has no
    in-sim cost.) FIX = `airborne_action_penalty` (dock JUMP-while-jumping / SLIDE-while-sliding; reward.py
    + torch _shaping + train_fast; equivalence GREEN). **v58 = v57 + penalty 0.15: offline jump-spam 80%→
    27% (PROVEN in-sim). BUT DEVICE spam UNCHANGED: v57 97% / v58 97% — the fix did NOT transfer.**
    **★ 2026-07-08 DEVICE JUMP-TIMING PROBE (`scripts/probes/probe_jump_timing.py` single-vs-burst n=41 +
    `experiments/probe_airtime_30hz.py` 30Hz n=50): ALL THREE documented hypotheses REFUTED.** (1) airtime =
    23fr@30Hz = **0.767s** (dist {23:40,24:10}) ≈ the sim's 0.80s (within 1 frame / 4%, inside dr [0.72,0.88]);
    NOT longer. (2) `is_jumping` obs is **FRESH** — liftoff latency 0 frames (single 25/25), NOT TCP-stale.
    (3) NO airborne QUEUEING — burst-of-8/single airborne = **1.00x** median (airborne presses are DROPPED
    exactly like the sim; the bimodal {7:10,14:5} is a post-LANDING press firing a fresh GROUND jump, not a
    queued airborne re-jump). BONUS: device jump PEAK = 19.17 = sim's `crevasse_jump_peak` 19.2 (arc height
    model correct). **CONCLUSION: the sim jump model is FAITHFUL; jump-spam is NOT a fidelity gap. Airborne
    presses are HARMLESS on device (dropped) — the `airborne_action_penalty` solves a non-problem for device,
    so v58 correctly showed no device change. The 6813m "random jump after obstacle" death is really a
    spurious GROUND jump leaving the runner airborne into a must-slide = the JUMP→SLIDE death (v53 scenario),
    a policy-HABIT issue, not a jump-timing gap.** No `jump_duration` recalibration warranted. Probe captures:
    `captures/jump_timing_probe.jsonl`,`captures/airtime_30hz.jsonl`. **NEXT: pivot to the #1 median killer =
    CREVASSE (~610m); the probe validated the arc peak so the graze model's height is right — the gap is the
    cap-arc OFFSET / takeoff-VARIANCE (v55 lesson).** v58 had 0 hole deaths in its 6ep batch but n=6 = run-luck.
    Captures: `ab4_v57_r1`,`ab5_v58`.
  - **DEPLOY: v51 (obs58, hook pre-fork obs.py) remains the safe device champion. v57/v58 (obs60, 512-net,
    hook 16z) are the best offline but NOT device-proven > v51. All weights gitignored/local.**
- **★ EDGE-CASE ACCUMULATION WAVE 2 (2026-07-07 late): the method is now "enumerate + model the edge cases
  that kill early" (USER framing: device variance is RUN-GENERATION luck, not thermal — a run dies when it
  hits an UNMODELLED edge case; the median is capped by how many remain). Edge-case forensic (26 device
  deaths, `experiments/death_edge_cases.py`): CREVASSE strikes EARLIEST (median 610m = #1 median-killer),
  then HOLE (2080m), NARROW (3037m), JUMP->SLIDE (3916m), TURN dead-letter LATEST (5880m, ends good runs).**
  - **JUMP->SLIDE fix -> v53** (`ppo_v53_jumpslide_best.zip`, @1.7M): device forensic — a slide obstacle
    right after a crevasse/long-hole traps the runner AIRBORNE (0.8s jump airtime) -> dt=3/5. Sim never
    generated it (min_space). NEW `jumpslide_prob` compound (both sims, `_pending_post_gap` tail after the
    wide-gap, equivalence GREEN). v53 = v51 + jumpslide (isolated). Offline: +30% on jumpslide cfg (dt3/5
    ELIMINATED), +5.5% standard (NO regression — gap-closing adds don't regress, unlike v52 pressure).
    Device A/B: v53 med 1764 ~= v51 1979 same-session (TIED, run-luck; no regression). CLEAN, kept.
  - **JUMP->STRAFE fix -> v54** (`ppo_v54_jumpnarrow_best.zip`, @1.7M): USER-observed + forensic (narrow
    deaths @5148/3281/2793) — a side-narrow right after a jump-obstacle needs a strafe-into-lane during the
    jump; policy commits too LATE (cp~0.9) -> analog tilt can't reach the lane -> falls off. Sim never
    generated jump->narrow. NEW `jumpnarrow_prob` compound (same `_pending_post_gap`, narrow payload;
    equivalence GREEN 2873/2873; v51 median 2514->1494 with it on = bites). v54 = v53 + jumpnarrow
    (ACCUMULATE). Offline: BEST on jumpnarrow (2623, +31% vs v51), keeps most jumpslide (2659 vs v53 3061 =
    mild interference), ~parity standard (2538). v54 = most WELL-ROUNDED (covers both new edge cases).
  - **NEXT = CREVASSE geometry fix (the #1 killer @610m).** Decoded: step-UP crevasse (lvl2->3) at LOW
    speed (~v166, the FIRST crevasse ~587m), policy takes off cp0.7-0.9, apexes on the lvl-2 gap cell then
    DESCENDS onto the lvl-3 raised cap (vel_y -33..-62 at cap entry) -> clips -> dt=4 Ledge. Sim too lenient
    (lets v51 clear at apex; device cap sits ~0.5 cells LATER in the arc). Fix = shift the sim step-up cap
    LATER in the arc so v51 clips in-sim like device -> retrain. Device cap arc-height dist measured
    (median 17.2, tail 3.1-13 = the clips) but the ground_height REFERENCE FRAME is ambiguous (clip is vs
    the ledge WALL not top) -> needs careful calibration. Config knobs `crevasse_graze_worldy`/`_jump_peak`
    /`_cap_height_hi`/`_graze_band` already in both sims (fix-A foundation) but were a NO-OP (v51 over-
    clears) -> the geometry offset is the missing piece.
- **★★ DEVICE CHAMPION = v51** (`runs/ppo_v51_next2_best.zip`, 256-net **obs60**, hook **16z**): the
  CHAINED-OBSTACLE OBS FIX. Controlled same-session device A/B vs its OWN obs60 baseline (v47 surgery-
  widened, action-identical start — so the ONLY difference is v51's learned use of the 2-ahead obstacle):
  **v51 median 4585 / mean 5214 / max 8763 (n=8) vs v47_obs60 median 2203 (n=6) = +108%, MORE THAN
  DOUBLE.** Non-overlapping (v51 min 3422 > v47 max 2895); thermal-confound RULED OUT (v51's warmest batch
  median 5830 > v47's coolest 2586). Death signature shifted as predicted: v47 died on flames+crevasses,
  **v51 has ZERO flame/crevasse deaths** — now dies 4/8 on the TURN DEAD-LETTER (pt=−1, deliberately left
  unaddressed). Biggest single-fix device gain of the project; offline→device correlation AMPLIFIED
  (+11.8% offline → +108% device: the real track has far more chained scenarios than the offline eval).
  Captures: `captures/ab_v51_r1.jsonl`,`ab_v51_r2.jsonl`,`ab_v47obs60_r1.jsonl`,`ab_v47obs60_r2.jsonl`.
- **v47** (`runs/ppo_v47_land_1703936_steps.zip`, 256-net obs58) = PRIOR champion / obs58 fallback: fresh-
  phone median 3540/6783. Now SUPERSEDED by v51 on device, BUT v47 stays the obs58-lineage deploy (needs
  the PRE-fork obs.py — current code builds obs32/60). `runs/ppo_best.zip` (v36, obs30) = ultimate fallback.
- **★ FIX (B) CHAINED-OBSTACLE OBS = SHIPPED + DEVICE-VALIDATED → v51 (obs60 FORK). 2026-07-07.** Added
  `next2_must_jump`/`next2_must_slide` end-to-end (obs 30→32 / full 58→60): obs.py OBS_BOOL_FIELDS idx
  19/20, both sims' `_snapshot` (nxt2 obst ∈ _MUST_JUMP/_MUST_SLIDE, junction-dropout `not (cell.is_turn
  or nxt.is_turn)`), hook **16z** (`clean-16z`: reads nxt2 PE_Obstacles in the neighbor walk, same NULL-
  through-turn dropout), capture_calib. **Equivalence GREEN 2184/2184.** Surgery v47→obs60
  (`runs/ppo_v47_obs60_next2.zip`, zero cols @19,20, 0/400 action mism). Isolated retrain (ONE change,
  v47's exact sim cfg, no dense profiles) → v51 best=1.70M. Offline: median 2485 vs baseline 2223 (+11.8%),
  ablation-proven (zeroing next2 cols → reverts to 2233). Device: +108% (above). Scripts:
  `experiments/surgery_v47_next2obst.py`, `train_v51_next2obs.sh`. UNCOMMITTED (per norm).
- **TURN FIX v52 = DEVICE-DISPROVEN (2026-07-07).** v51 dies 4/8 on the TURN DEAD-LETTER (pt=−1, at
  3422-8763m). Tried v51 + `turn_send_lat_frames=0.4` ONLY (isolated 2M warm-start, `ppo_v52_turnlat_best`
  = @1.00M; parity GREEN at slat=0.4). Offline it looked +20% under matched latency BUT checkpoints were
  NOISY (bounced 1315-2980, 1.7M collapsed to Tree/Ledge:101). Device (6ep): **median 2274 vs v51 4585 —
  REGRESSED BY HALF.** Turn deaths DID fall (4/8→1/6) but obstacle deaths EXPLODED (5/6: narrows/holes/
  straights), variance 1259-5906. **LESSON: v51 is a SHARP optimum; the turn-latency pressure knocked the
  policy off its obstacle skill (a 5e-5 finetune still moved it too far). USER confirmed live: v52 does
  redundant post-obstacle jumps + grazes mid→high crevasses almost every time.** Captures: `ab2_v52.jsonl`,
  `ab2_v51_r1.jsonl` (2 valid: 5147,1930). **v51 REMAINS CHAMPION.** Turn dead-letter now needs a NON-
  destabilizing fix: gentler lat (0.2), OR — better — a HOOK-SIDE pre-swipe (queue the turn on the approach
  cell in the gadget, like the native turn-buffer, NO retrain → zero risk to v51's obstacle skill).
- **★ NEXT DEATH-FIX (device-forensic 2026-07-07, v51 captures): JUMP→SLIDE AIRBORNE DEATH.** v51's 2
  Slide deaths (dt=3, @4991 pt1 / @3916 pt2) decode to ONE mechanic: a jump-obstacle (crevasse pt35/36 OR
  long-hole pt15/16) is immediately followed by a SLIDE obstacle — the crevasse jump's fixed ~0.8s airtime
  (~2.8 cells @v210) overlaps the slide → the runner is STILL AIRBORNE (`is_jumping=1`) when `must_slide`
  fires → can't slide → dt=3. (The user-observed redundant re-jump near the transition makes it worse.)
  **SIM GAP CONFIRMED: v51's offline death mix is `{Fall,Tree}` — ZERO dt=3/dt=5. `min_space_between_
  obstacles` forces a clear gap after every wide-gap, so the sim NEVER generates jump-obstacle→slide → the
  policy never trains on it → dies on device.** FIX (config-gated, both sims, like the chained-obstacle
  win = add the missing SCENARIO): a `jumpslide_prob` compound that places a TreeSlide/FlameTower ~1-2 clear
  cells after a crevasse/long-hole (bypassing min_space), so the policy must learn to jump the gap EARLY
  (land before the slide) + not re-jump. CPU sim: new `_pending_post_gap` list checked AFTER `_pending_gap`
  (gap cells emit first, then the slide tail); gate the rng on `jumpslide_prob>0` FIRST (else the extra
  random() shifts the whole track stream → scorecard/equivalence drift). Torch sim: harder (batched pending
  sequences) — check `sim_env_torch` gen first. Then isolated retrain from v51 (ONE change) → device A/B.
  NOTE the v52 lesson: gate HARD on device; v51 is a sharp optimum. (v51 next2_must_slide sees 2 ahead but
  the slide sits 3-4 cells from the crevasse-jump DECISION → may also want next3 slide lookahead later.)
- **FIX (A) MID→HIGH CREVASSE GRAZE = IMPLEMENTED but NEEDS GEOMETRY CALIBRATION (deferred 2026-07-07).**
  `crevasse_graze_worldy` knob added to both sims (jump_peak 19.2, cap 15@lvl≥3/0@lvl<3, arc_y<cap→stumble,
  <cap−band→kill), equivalence GREEN 2366/2366 (variant `crevasse_graze`). BUT calibration = NO-OP for v51:
  it clears lvl-3 caps 100% in-sim (arc at full apex 19.2 >> cap 15) while grazing 32% on device (arc at
  cap entry world_y~14-18, DESCENDING). The gap is the sim's crevasse ARC GEOMETRY (cap sits at v51's apex
  in-sim, ~0.3-0.5 cells later on device). To make it bite: measure the exact cap-arc offset from
  `captures/v47_16y_arcs.jsonl` + shift the sim cap so v51 grazes ~32% in-sim, THEN retrain. Deferred —
  the graze is NON-FATAL (chronic slowdown, not death; user: "does not die") so lower-priority than the
  jump→slide DEATH. Calib script: `experiments/calib_crevasse_graze.py`.
- **(superseded) FIX (A) MID→HIGH CREVASSE GRAZE — original spec (now see the IMPLEMENTED entry above):** "Transition mid→
  high is complicated, the policy does NOT die but takes a hit almost every time" = the chronic high-ground
  cap Stumble (×0.85) I decoded via hook 16y (jump_peak 19.2, cap 15@lvl3, graze iff arc_y<cap_top). v51/
  v52 share it (neither touched crevasse). Config-gated/safe (default off, equivalence stays green). Since
  closing a REAL device gap = fuel (chained-obs proved it), this is the highest-value next line — but MIND
  the v52 lesson: any retrain risks knocking v51 off its optimum, so keep it ONE isolated change + gate hard
  on device. Also seen: redundant post-obstacle jump-spam (candidate: redundant_action_penalty, but reward-
  shaping is destabilising — treat carefully).
- **DEVICE-OPS (this session, 2 recoveries): (1) the injector `gadget_inject.py` self-exits after 3600s
  (1h) → hook UNLOADS → captures go "device degraded". For long sessions re-inject before it lapses.
  (2) `adb forward` rules keep getting DROPPED (scrcpy/Coolmuster adb-server thrash) → re-forward with
  scrcpy's adb + re-inject (see [[device-adb-scrcpy-thrash]] memory). Always verify `_hv` on 9999 with a
  `settimeout` socket after any device op.**
- **DEVICE-BOX NOTE (this PC, 2026-07-07): phone 3f80a3a IS on this machine** (frida 17.14.0, app installed
  + running). adb = **scrcpy's** `D:\téléchargements\scrcpy-win64-v3.3.4\adb.exe` (v1.0.41) — USE ONLY THIS
  ONE. A stray Coolmuster adb v1.0.36 (`C:\Users\bartb\AppData\Roaming\Coolmuster\adb\1.0.36\adb.exe`)
  causes a SERVER-VERSION THRASH: whoever starts a server kills the other's + DROPS ALL FORWARDS → severs
  the frida session (injector process stays alive but hook stops streaming). RULE: keep scrcpy running,
  don't restart it mid-capture; if forwards drop, re-forward with scrcpy's adb + re-inject (kill old
  injector PID first). No repo `.venv` — use `python` (= G:/python312, has frida + torch cu126).
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
the narrow↔crevasse interference but drifts — needs a stable from-scratch train, see `experiments/
train_gpu_256.sh`), (4) stumble→rejump recovery.

## Setup
- **Engine** Unity 2022.3 IL2CPP, arm64 only → needs a REAL arm64 phone (no emulator: Frida can't hook
  the ARM lib under x86 translation). Device **3f80a3a** (Redmi, Android 15, NOT rooted).
- **Frida Gadget** (not frida-server — no root): `libgadget.so` injected via LIEF `DT_NEEDED` on
  `libmain.so`; listens 127.0.0.1:**27042**. Build: `tools/build_gadget_apk.py` → `TempleRun-gadget.apk`.
- **Reverse eng.**: Il2CppDumper → `il2cpp_work/output/dump.cs` (signatures + field offsets + RVAs;
  bodies are native, not readable). All offsets/RVAs below are from it (re-dump if the APK changes).

## Run it
`python scripts/device/setup_device.py` forwards 27042 + state **9999** + action **9998**, then injects.
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
  `docs/archive/CLAUDE_FULL_HISTORY.md` §5 / `docs/OFFSETS.md`.

## Sim + training
- `src/templerun_rl/sim_env.py` = fast headless model (`TempleRunSim-v0`); obs `obs.py` (**OBS_LEN 30**,
  bools + scalars + cell_progress/lane/narrow_target_lane/**predicted_landing**/next_ground_change_higher
  + **predicted_jump** (velocity-invariant step-up takeoff cue, host-derived) + **next_path_level/
  next2_path_level** (upcoming-altitude direction, hook 16r) + K=4 action history → 58-wide; power-ups
  REMOVED 2026-07-04); reward `reward.py`; PPO `scripts/train/train_sim.py` (net 128,128, linger 3).
  Deploy = same obs on the phone via the hook.
- Sim turn-buffer config: `hook_turn_buffer=true` models the device swipe-queue (turn near a junction lands
  CENTERED, timing-free, `from_buffer` offset 0) + **late gate `turn_buffer_late_cp=0.56`** (swipe past it =
  silent MISS; device-measured CanTurn window end, velocity-INDEPENDENT — v14 turned 46/46 up to v207).
  Warm-start base **`runs/ppo_v14_obs30_leveldir.zip`** (EXACT surgical equivalent of v14, new obs cols
  zero-init; `experiments/surgery_v24_leveldir.py`).
- **JUNCTION LEVEL DROPOUT (2026-07-05 night, device-measured):** the hook's PE neighbor walk BREAKS through
  turns → on-junction frames stream `next_path_level=next2_path_level=-1` (178/202), approach frames
  `(lvl,-1)`. Sim `_snapshot` reproduces it (commit f8d7eff). Obs30 finetunes trained pre-fix saw a
  junction-frame input spike on device they never met in training. v14 immune (zero-init cols).
- **Capture-log alignment (capture_calib):** logged `action(i)` = policy's response to `state(i-1)` (predict
  runs BEFORE env.step; the record holds the POST-step state). And a FIRED turn resets cell_progress → the
  swipe record's cp is POST-turn — don't read "swipe cp" stats off fired turns (artifact; burned us twice).
- **Jump exec latency (sim, 2026-07-05):** `jump_exec_lat_frames`/`jump_exec_lat_jitter` buffer the JUMP
  firing. DEVICE-MEASURED (`scripts/probes/probe_jump_exec.py`, 25/25): **exactly 1 frame @10Hz, zero jitter**
  (uniform with the turn-side 1-frame exec). Constant latency is ABSORBED by policies (window shifts);
  only JITTER creates training pressure (v14 hiMed 2246→130 under 2±1; a predicted_jump pre-aim oracle
  survives). Train at **1±1** (measured mean + TCP obs-staleness variance); harness/replay at 0.
- **Warm-start recipe that does NOT collapse: `--learning-rate 5e-5`** (new knob; default 3e-4 ALWAYS
  collapses a v14 warm-start regardless of ent). Keep the BEST checkpoint — drift resumes past ~500k-1M.
- **THE FEEDBACK LOOPS (use before any retrain/deploy):**
  - **`scripts/validate/gate_battery.py <candidate.zip>` (WAVE 10 lesson, MANDATORY before ANY device A/B):**
    4-config offline battery vs the champion (standard / crevasse-dense / stack-heavy / lat-stress)
    with per-death-type breakdowns; thresholds 0.90x (1.00x lat). VALIDATED: retro-rejects v71c
    (crevasse 0.77 + stacks 0.74 FAIL) exactly matching its device regression — a standard-cfg-only
    gate MISSES skill erosion (crevasses/stacks too sparse there). RED = do not spend phone time.
  - `scripts/validate/fidelity_scorecard.py --strict` — ~13 phone-free checks vs `captures/device_reference.json`.
    GREEN required before retrain. (Blind for un-checked mechanics — green ≠ complete.)
  - `scripts/validate/replay_diff.py <capture>` — policy-free ONE-STEP sim-vs-device transition diff on a real
    capture. Great for turns/obstacles; CANNOT test multi-frame mechanics (the crevasse jump arc).
  - Device eval: `scripts/device/capture_calib.py <model> --episodes 8 --turn-cooldown 0`.
- Checkpoints are noisy + late-collapse → **keep the BEST checkpoint, not `_final`**. Registry `runs/MODELS.md`.

## GPU sim + training (2026-07-06, GPU box — torch port of the sim)
- **`src/templerun_rl/sim_env_torch.py` = batched-tensor port of sim_env.py** (`TorchTempleRunVecEnv`,
  B envs as [B] tensors, cuda/cpu). The CPU sim stays the REFERENCE — if they disagree, fix the port.
  **Equivalence gates (run after ANY change to either sim): `scripts/validate/sim_equivalence.py`** (1344
  deterministic step-exact scenario×config×policy diffs + statistical tier), `scripts/validate/scorecard_torch.py
  --strict` (the unmodified scorecard via a registry alias; 12/13 PASS — turn.gates oracle-survival is
  threshold-MARGINAL on BOTH sims: true rate 73%cpu/72%torch vs the 70% bar, n=80 seed-batch coin flip),
  + v36 plays both sims to median ratio 1.002 (n=160). Sim steps ~40k/s at B=4096 (65x the SubprocVecEnv).
- **Training**: `scripts/train/train_sim_gpu.py` (SB3 over the batched env; recipe-faithful geometry =
  B=256 n_steps=16 batch=64 → ~900 steps/s end-to-end — the PPO optimizer loop is the wall, NOT the env;
  batch=512 gives 5k steps/s but BROKE a from-scratch run — dynamics change, use only with curve
  validation). `scripts/train/train_fast.py` = pure-torch GPU-resident PPO (SB3-exact losses, saves normal SB3
  zips) — **VALIDATED 2026-07-06: 1.29x SB3 at recipe geometry + reproduces from-scratch turns-only
  learning exactly (96/96, median 6238 = SB3's). Use it for new runs.** Quick verdict tool:
  `scripts/validate/eval_ckpt_torch.py <ckpts...>` (batched device-true-cfg eval).
- **FROM-SCRATCH WALL (measured 2026-07-06): the full task is NOT from-scratch trainable** — flat ~100m
  (the FIRST junction) for 3.4M steps (reference recipe, CPU trainer) AND 6M (torch). Cause: since the
  2026-07-02 fatal-turn remodel, turn-on-straight=death makes turn exploration a needle; every good
  policy is a warm-start descendant; train_gpu_256.sh was never actually validated. **FIX (user-designed,
  verified): curriculum.** (a) turns-only phase A (obstacle_prob 0, turn dense) → from-scratch 256 masters
  turns in <100k steps (96/96 to cap), then full-task phase B warm-start (climbing, 1173@1.3M); (b) BETTER:
  **per-episode `episode_profiles` mix** (torch sim only: cfg list of {weight, overrides} sampled each
  reset — e.g. 50% standard / 15% turns-only / 12% narrows-from-0 / 12% crevasse-dense / 6% hole-rhythm /
  5% compound) = guaranteed rare-mechanic exposure with zero phase-atrophy. See runs/ppo_mix256 lineage.
- **256-net path that works: v41 = v36 surgery-widened (experiments/surgery_v39_256.py, exact-equivalent
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
  correlate again). Forensics (`experiments/jump_arc.py`, 1600 jumps / 8 captures): 2-cell HoleLongs
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
  device crevasse HIT-RATE (slowdown metric, `experiments/crevasse_clips.py`) 19%→15% (v36=16%) — the fix
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
- **`scripts/validate/trajectory_diff.py` (NEW, multi-frame comparator — the cheap sim-vs-game tool replay_diff
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
  scripts/device/setup_device.py).
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
  healthy. Judge checkpoints with `experiments/eval_swipe_deaths.py` (device-true cfg, deaths by mechanic).
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
