# Model registry — which weights = what

Trace of the key `runs/*.zip` checkpoints: the sim they were trained on, the recipe, and the result.
Obs width: **57** = obs-29 base + 4×7 action-history (pre-narrows). **58** = obs-30 base + 4×7
(narrows obs added, commit 4c018d7). All NN-decides (K=4 action history, frame_stack=1, MLP [128,128]).

| model (`runs/…`) | obs | sim version | recipe delta | in-sim median | device | notes |
|---|---|---|---|---|---|---|
| `ppo_nn_best.zip` | 57 | pre-audit, 1-cell crevasse, 3-bucket lane | turn4b @1.5M, ent0.04 | — | **median 1210 / best 2182** | the CLAUDE.md-documented deploy model. **STALE on sim-calibration** (obs is 58 now). Run via capture_calib obs-adapter. Dies at crevasse (~616) + turns. |
| `ppo_best.zip` | 57 | older (lingerclean) | — | — | best ~1610 | superseded by ppo_nn_best |
| `ppo_calib_best.zip` | 58 | **PRE-density-audit** (2-cell crevasse added, but jump still 0.6, old densities/3-bucket lane) | ent0.04 @1.9M | 1712 | not tested | first crevasse retrain; ent0.04 → slow/noisy |
| `ppo_faithful2_best.zip` | 58 | **faithful** (jump0.8, densities audited) | ent0.05, NO strafe-dock, 3-bucket lane | 1634 @1.0M | strafe-off-edge @297-590m | ent0.05 fixed turn-collapse |
| `ppo_faithful3_best.zip` | 58 | faithful | ent0.05 + **strafe_penalty 0.30**, 3-bucket lane | 1536 @400k | strafe-off-edge @94m (worse, brittle ckpt) | strafe dock alone didn't fix it |
| `ppo_faithful4_best.zip` | 58 | faithful + **CONTINUOUS LANE** | ent0.05 + strafe0.30 + continuous lane | 1823 @1.2M | strafe-off REDUCED not gone (600k: 2/6 off + 3/6 crevasse@~600; 1.2M: 5/6 off) | continuous-lane obs; remaining gap = INSTANT vs GRADUAL device strafe |
| `ppo_nn_turn5c_*`, `ppo_nn_turn6_*` | 58 | post-narrows | various | weak (~260) | — | abandoned |
| ~~`ppo_turnfix_best.zip`~~ (=1400k) | 58 | TURN re-model **with BUFFERING** (window 0.45, early-swipe latch); linger 0 | ent0.04 @1.5M | 1980 (99.8% turns) | **DEVICE FAILED: median 226 (< 736!)** | **REGRESSED.** The early-swipe BUFFER was a SIM EXPLOIT: policy pre-fired turns on next_is_turn (safe in-sim) → on device rotates off-path into the WATER (15/21 turns fired on straights). 99.8% in-sim was gamed. DO NOT DEPLOY. |
| **`ppo_turnfix2_best.zip`** (retraining) | 58 | TURN re-model **NO buffer** (2026-07-01 fix): turn ONLY on is_turn flash; off-junction swipe = stumble (2 close = Fall); linger 0 | same recipe | pending | pending device | buffer removed (`_attempt_turn`); `scripts/turn_gate.py` now PASSES (PRE-FIRE dies 0/80 @101m = the device failure reproduced). See [[turn-investigation-roadmap]] |
| ~~`ppo_tilt_v1_best.zip`~~ (1.8M) | 58 | first SCORECARD-GREEN sim, but **`--turn-linger 0`** | recipe §12, linger 0 | ~1617 | **DEVICE: DOUBLE-TURNED** (12 bursts, 8/8 turn deaths, dist 180-540) | linger 0 = sim consumes is_turn instantly → policy never learned to resist the device's 2-frame is_turn. Superseded by v2. See CLAUDE.md §16 |
| **`ppo_tilt_v2_best.zip`** (2.3M, 2026-07-02) | 58 | scorecard-green sim + **`--turn-linger 3`** (double-turn temptation modeled; turn_gate.double_turn gate dies 0/80) | recipe §12, linger 3, ent 0.04 | 1562 @40ep, 11 deaths, **5 double-turns (v1 had far more)** | pending — deploy under hook **lateral-accel-16b** (accel tilt + Region2D band, RAW is_turn / latch off), `capture_calib --turn-cooldown 0` | THE double-turn fix; keep BEST ckpt (2.3M = fewest deaths + double-turns) |

**Recipe baseline (all faithfulN):** `--net-arch 128,128 --action-history-len 4 --frame-stack 1
--turn-linger 3 --ent-coef <E> --action-cost 0.06 --idle-action-penalty 0.12 --turn-spam-penalty 0.14
--redundant-action-penalty 0.12 [--strafe-penalty 0.30]` (domain_rand on).

**Checkpoint policy:** runs are NOISY + late-collapse → KEEP THE BEST CHECKPOINT (rigorous 50-ep
`_rollout`), not `_final`. `<name>_best.zip` + `.json` sidecar (frame_stack/K/arch) = the picked one.

**Eval:** `scripts/capture_calib.py <model> --episodes N --turn-cooldown 0` (drives policy on device,
logs frames; has an obs-adapter for the obs-57 ppo_nn_best). Classify deaths: last-frame `lane`
(|lane|>7 → strafe-off-edge) / `path_type` (33-36 crevasse, 17-30 narrow, 3/7 turn).

See memory `sim-calibration-capture-2026-06-26.md` for the full fidelity story + the gradual-strafe fix.
