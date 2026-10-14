# Turn dead-letter fix — v71 twin-arm design (2026-07-09)

## Problem

The turn dead-letter (pt-1 death) ends good runs: v70 @4733 (decoded frame-by-frame in
`experiments/turn_death_forensic.py` on `captures/ab_v70_r2b.jsonl` ep0), 4/8 of v51's deaths,
one of v66's. Mechanism (device-measured):

- At v209 one 10 Hz control frame covers ~0.47 cell-progress (cp).
- The policy only decides a turn ON the junction cell. Junction first-sighting phase is a
  sampling coin flip: seen at cp 0.1 → survives; seen at cp 0.5 → the TURN byte lands at
  cp ~0.9, the write reaches the game's swipe buffer (`gi_should_turn_r=ON` in the capture)
  but the junction ends before it is consumed → runs off → dt2.
- The latency-immune play exists and is free: swipe on the APPROACH cell — the game's native
  HandleDelayedTurning buffer holds it and fires at the corner. The sim models it
  (`turn_approach_buffer`). v70 never uses it: across all 4 junctions in the death tail it
  acted only on the junction cell.

Root cause (fidelity): the sim gates a swipe on the DECISION cp (≤ `turn_buffer_late_cp`
0.56) with NO send latency, while the device effectively gates DECISION cp + ~1 control
frame. Late junction-cell swipes are free in-sim → no gradient toward the pre-swipe.

## Fix

`turn_send_lat_frames` (existing, equivalence-green knob; parity 16/16): gates on
proc_cp = decision_cp + velocity·lat·dt/cell. Makes the late junction-cell swipe MISS
in-sim exactly as on device; the pre-swipe (approach-cell, from_buffer) is immune.

History: v52 tried 0.4 as an ISOLATED 2M finetune on v51 (a sharp optimum) → regressed by
half. The failure mode was the fragile-finetune regime, not necessarily the value: v70's
anchored-mix consolidation held 5 scenario skills over 20M without drift.

## Arms (one knob each vs v70 — clean attribution)

**REVISED after adversarial review + pre-flight sweep (the original 0.2/0.4 doses were INERT):**
a critic subagent verified against sim code that in-sim a first-frame junction reaction decides
at cp < v·dt/cell ≈ 0.35 @v210, so lat ≤ 0.4 never pushes proc_cp past the 0.56 gate. The sweep
(v70@20M, n=96/dose) confirmed empirically: lat 0.4 median 3002 = IDENTICAL to lat 0 (inert);
lat 0.7 → 846 (binds, 86/96 turn deaths); lat 1.0 → 272 (96/96). The v52-era "0.4 = pressure"
note was mis-calibrated device arithmetic.

- **v71a** = v70 recipe + `turn_send_lat_frames: 0.7` (binding pressure, partial survival = gradient)
- **v71b** = v70 recipe + `turn_send_lat_frames: 1.0` (device-plausible full dose; pre-swipe or die)

Both: init `runs/ppo_v66_obs63_n2nl.zip` (v70's own init), 20M steps, cosine 8e-5→5e-6,
identical profiles/knobs otherwise. Trained CONCURRENTLY (GPU: v70 alone used 803 MiB / 59%;
two runs contend → expect ~1.5-2× wall each, ~2 h total).

## Gates

1. Offline (n=96, `eval_ckpt_torch`), each arm's best ckpt vs v70@20M:
   - standard device-true cfg: parity or better required (no obstacle-skill regression);
   - **lat-1.0 stress cfg**: the validated pre-swipe metric — a junction-cell reactor scores
     ~272 (v70's measured floor); a learned pre-swiper is unaffected by the latency. Require
     the arm's lat-1.0 distance ≈ its lat-0 distance.
2. Device A/B (cooled-phone alternating 4-ep batches vs v70): turn pt-1 deaths ↓,
   pooled median ≥ v70's 3130.

## Rejected alternatives

- Reward bonus for approach swipes: reward shaping failed twice (v60 gamed by 1-frame gaps,
  v61 gate let the killer through); latency pressure prices the real mistake instead.
- Hook-side auto-turn: violates the NN-decides-all-actions rule; the hook only translates.
- Finetune from v70@20M: cheaper but reproduces the v52 fragile-finetune failure mode;
  consolidation-from-base is the proven regime for adding pressure.
- 20 Hz control: measured WORSE historically (retrain confound documented).
