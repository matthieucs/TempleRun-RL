"""RNG draw-order lockstep with type_reject_model ON.

cpp_sim/test_rng_order.py is left UNMODIFIED so its 60/60 stays comparable to the frozen
baseline. This re-runs the SAME harness with the new sampler enabled, which is the only check
that actually exercises the new draw site in generation (the equivalence deterministic tier
injects fixed tracks and cannot see the generator at all).

    python experiments/_tr_rng_order.py            # type_reject ON,  redraws 1
    TR_REDRAWS=3 python experiments/_tr_rng_order.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
from cpp_sim import test_rng_order as T

# A deliberately dense matrix: every rejectable `cur` (1 Hole / 2 Narrow / 6 FlameTower) against
# several `prev` types, doses spanning 0.2..0.95, so both a "no roll" and a "reject then re-draw"
# outcome are hit constantly.
T.CFG = dict(T.CFG)
T.CFG.update({
    "type_reject_model": True,
    "type_reject_redraws": int(os.environ.get("TR_REDRAWS", "1")),
    "type_reject": {"1,1": 0.80, "1,2": 0.35, "1,6": 0.20,
                    "3,1": 0.25, "4,1": 0.55, "4,2": 0.40,
                    "5,1": 0.95, "5,2": 0.70, "5,6": 0.30,
                    "6,1": 0.90, "6,2": 0.60,
                    "7,1": 0.45, "7,2": 0.15},
    # legacy knobs ON at the same time: with the new model selected they must be inert on BOTH
    # ports, which is itself a draw-order assertion (a stray legacy roll would desync instantly).
    "pair_same_type_prob": 0.6,
    "pair_anti_repeat_prob": 0.41,
})

fails = 0
total_steps = 0
n_ep = int(os.environ.get("RNG_ORDER_EPS", "60"))
for ep in range(n_ep):
    rng = np.random.default_rng(1000 + ep)
    vals = rng.random(200000)
    K, FS = (4, 1) if ep % 3 == 0 else ((0, 2) if ep % 3 == 1 else (0, 1))
    err, n = T.run_episode(vals, 400, K, FS)
    total_steps += n
    if err:
        fails += 1
        print(f"FAIL ep{ep} (K={K},FS={FS}): {err}", flush=True)
print(f"\nRNG DRAW-ORDER [type_reject_model ON, redraws={T.CFG['type_reject_redraws']}]: "
      f"{n_ep - fails}/{n_ep} episodes lockstep ({total_steps} shared-stream steps)")
print("GREEN" if fails == 0 else f"RED: {fails} failure(s)")
sys.exit(1 if fails else 0)
