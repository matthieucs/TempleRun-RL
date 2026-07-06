# Opening prompt for the GPU-box Claude Code session

Copy-paste everything below the line into the first message.

---

You're on a headless GPU machine (no phone attached). Start by reading `CLAUDE.md` in full — it's the
handoff doc and it's current. Context in one breath: this is an RL agent that plays real Temple Run;
the 10,000 m goal was REACHED yesterday (device run 10,545 m, policy v36). We are now in the
**RELIABILITY phase — the job is a 10,000 m MEDIAN, not just a best** (yesterday's device runs ranged
589 → 10,545 m; high variance).

Your machine's job is **training only** — this box has the GPU but no phone, so you cannot device-eval.
You train candidates in the headless sim and gate them offline; device verification happens on the
other machine (the one with the phone). Hand back a model file + an offline scorecard, don't claim a
device result you can't measure.

**The specific task: a stable FROM-SCRATCH 256-net.** Yesterday's finding (see CLAUDE.md STATUS + the
v37–v40 campaign): a 128-net cannot hold reach-center-narrows AND centered-crevasses at once — they
interfere, and warm-start retrains just trade one skill for another. A 256-net *widened* from v36 broke
the interference offline (narrow 0/50 + crevasse 3/40) but drifts unstably because it inherited v36's
128-net optimum. The clean shot is a 256-net trained **from scratch** (no warm-start) on the full
scenario stack so it learns every skill together. `scratchpad/train_gpu_256.sh` is the starting recipe
(SB3 auto-uses CUDA; raise `--num-envs` to your core count — the bottleneck is the CPU-bound Python sim,
not the MLP). Expect it may need >6M steps to reach v36's quality from scratch.

**Fold in the uncovered edge cases** (the variance sources, listed in CLAUDE.md's GOAL-REACHED block) as
you go — model them in the sim, keep the fidelity_scorecard `--strict` GREEN, don't break existing
mechanics:
1. Flame-tower slide-under at speed (the 10,545 m death — died Burnt, pt31/32).
2. Multi-hole jump rhythm (4-cell must-jump chains, jump-spam fails — needs staggered land-then-rejump).
3. Narrows-at-speed strict [3,7] band (already modeled via `strafe_exec_lat_frames` + narrow band; the
   256-net is the capacity to hold it alongside crevasses).
4. Stumble → rejump recovery.

**Hard constraints (unchanged, from CLAUDE.md):** the NN decides ALL actions — no hook/host action rules;
power-ups stay off; `fidelity_scorecard.py --strict` must be GREEN before you consider a model shippable;
keep the BEST checkpoint (they drift + late-collapse), not `_final`; offline-green ≠ device-safe for
un-modeled mechanics, so be honest about what the scorecard can and can't see.

First actions: (1) read CLAUDE.md, (2) confirm `torch.cuda.is_available()` is True, (3) run the
scorecard to confirm a GREEN baseline on this checkout, (4) launch `scratchpad/train_gpu_256.sh` (tune
`--num-envs` first). Then iterate: gate each checkpoint offline on narrows@v210 AND crevasse AND compound
AND the new edge cases — a winner beats v36 on ALL axes, not one. Report the best checkpoint + its
offline scorecard so it can be carried to the phone machine for the real median test.
