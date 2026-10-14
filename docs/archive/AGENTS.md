# Temple Run RL — project memory (minimal)

Goal: an RL policy that reaches **10,000 m** on real **Temple Run** (`com.imangi.templerun` v1.39.0),
trained in a headless sim, deployed on a physical Android phone via **Frida** (read game state + inject
actions over TCP — no screen-scraping, no synthetic input). Full history: `docs/archive/CLAUDE_FULL_HISTORY.md`.

## ⚡ CURRENT STATE (2026-07-14) — read this first
- **★★★ DEPLOY CHAMPION = v78 (`runs/ppo_v78_guided_18001920_steps.zip`, obs64, 512-net, hook 17f). The
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
  working dir has UNCOMMITTED same-file edits (clip band, min_obst, new scripts, this docs/archive/AGENTS.md) —
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
  (Doc-bug found: docs/archive/AGENTS.md's old "JumpVelocity 0x80/GroundHeight 0x88 (CharacterPlayer)" are the ENEMY
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
