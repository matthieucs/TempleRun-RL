# Overnight autonomous run — toward 10,000 m

Started 2026-07-03 (night). Autonomous per user: work until a device-confirmed 10,000 m or told to stop.

**Strategy:** velocity caps at 210 (~4200 m), difficulty saturates (~6000 m) → beyond that nothing is
harder. So 10k = robustness at v210. Each iteration attacks a v210 death mode (high-speed turn overshoot,
deep crevasse, narrow dither), validated offline (scorecard/replay_diff) before device eval. NN decides
all actions; no hook rules. No lateral-reward changes (v15 proved `off_center_penalty` device-toxic).

## Baseline entering the night
- **Deploy floor / best: v14** — `runs/ppo_v14_best.zip`, device **median 1652, best 2908** (8 eps, hook 16i).
  Death modes: high-speed turn overshoot (off-center approach + T-junction), deep crevasse @2908, 1 narrow.
- Rejected: **v15** (off_center_penalty) — device median 457, lateral-off-edge. `off_center_penalty` toxic.

## Iterations

### v16 — high-speed (v210) curriculum ALONE (warm-start v14)
- Change: cmax 8000, deep_start_frac 0.55, dist_power 0.9 (33% of episodes start ≥v205). No reward change.
- Scorecard: GREEN. Offline (checkpoint-select):
  - 2.0M: offctr-turn **14%** (best; v14 16%), nlane 4.5, fast crev 0, slow crev 5/turn 11/other 6.
  - 2.5M: offctr 16%, fast crev 0, slow crev 2/turn 12/other 4.
- Selected **2.0M** for device eval (best turn-centering + nlane, fast crev 0).
- Device eval hit a "degraded" phone (stuck on the v15 dead-run). Recovered via the recipe.
- **DEVICE: median 629, best 1350 — REGRESSION vs v14 (1652).** Deaths LATERAL-OFF-EDGE (lane 11-22)
  even WITHOUT off_center_penalty. **CONCLUSION: the high-speed deep-start curriculum ITSELF induces
  over-strafe OOD on device.** Both v15 + v16 failed this way → **abandon high-speed curriculum retrains.**
  v14's tamed curriculum was right.

### PIVOT → fidelity fixes (the moves that actually worked here: predicted_landing, 2-cell crevasse)
replay_diff on v14 capture: **turns are the least-faithful family (junction TL/TR 36% dead-agree** vs
90%+ elsewhere; turn_land_err ~3). Measured the device turn-landing curve from the v14 capture:
- **Overshoot is driven by SWIPE TIMING (cell_progress), NOT approach lane** (corr just 0.17):
  cp ≥ 0.62 → landing 10-13 → DIES (even near-center); cp ≤ 0.58 → landing <7.3 → LIVES (even lane +11).
- Root: the turn rotates the frame 90°, so pre-turn lane barely maps to landing. But the sim carried
  approach-lane at **3×** (`turn_lane_carry`), wrongly killing off-center-approach turns the device survives.

### v17 — turn FIDELITY fix (turn_lane_carry 3.0 → 0.25)
- Carry-sweep vs device junction dead-agree: **3.0=46%, 1.0=94%, 0.5=97%, 0.25=99%, 0.0=99%** (land_err
  4.6→1.7). Set **0.25**. Mechanism hypothesis: carry=3 forced center-then-swipe → delayed the swipe into
  the fatal-cp zone; removing it lets the policy swipe on-time regardless of lane → fewer device overshoots.
- Validated: scorecard `--strict` **GREEN** (turn.gates still pass — off-center/lean/pre-fire die via
  post-turn held-tilt drift, not carry); replay_diff junction 36→58%, turn_land_err 3.0→2.4. Committed b9e1b7c.
- v17 = warm-start v14 + **v14 TAMED recipe** (cmax5000, NO high-speed skew) + carry fix. Training. Device: _pending_.
- ALSO found: a separate LATERAL-EDGE gap (sim kills lane 12.7, device tolerates 12+) — candidate next fix.
- Human v210 crevasse takeoff cp≈0.58 (above) still pending as a crevasse-apex calibration.

## Op notes (autonomous gotchas)
- **NEVER `pkill -f <pattern>` where the pattern also appears in the invoking command** — `pkill -f`
  matches the shell's own command line and SIGTERMs it (exit 144). Kill by numeric PID (`kill <pid>`)
  or `kill -0 <pid>` to test liveness. Same for `while pgrep -f capture_calib` waiters: the waiter
  matches itself → infinite loop. Use `while kill -0 <pid>`.
- Device recovery recipe works: `am force-stop` → `am start ...ImangiUnityActivity` → wait 15s →
  re-`adb forward` 27042/9999/9998 → one `setsid` injector → verify `_hv` on :9999.

## Candidate pipeline (attack v210 death modes; validate offline first)
1. v16 — high-speed curriculum alone (running).
2. If turns still overshoot: **turn-isolation curriculum at v210** (the narrows trick) — more high-speed
   junction exposure; the sim already kills off-center turns, so the gradient exists, it's undertrained.
3. **Deep-crevasse apex calibration** — read `GroundHeight`/`JumpVelocity` live (hook PERCEPTION change,
   allowed) to nail `stepup_min_height`; scorecard-guard. Attacks the @2908 Ledge death.
4. Possible: sharpen junction decisions (lower ent_coef in a fine-tune phase) for the T-junction flip.
- Rule reaffirmed: no lateral-reward changes (v15). Training/curriculum/sim/perception only.

## Human v210 crevasse takeoff (offline calibration, from human_calib_v2.jsonl — the 14,956 m run)
- 42 crevasse-related jump takeoffs. Takeoff **cell_progress** on the approach cell (crevasse = next cell):
  - **v185+ (med 210): median 0.58, p25 0.37, p75 0.71.**
  - mid (v175): median 0.40.
- So at top speed the human commits the crevasse jump ~mid-approach-cell (0.58), not super-late. Use this
  to calibrate the sim's crevasse apex (`stepup_min_height` / takeoff window) instead of guessing — the
  deep-crevasse @2908 death is the target. TODO: verify the sim lets a cp-0.58 approach-takeoff clear at
  v210; adjust if it demands a later takeoff.
