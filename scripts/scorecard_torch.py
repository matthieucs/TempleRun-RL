#!/usr/bin/env python3
"""Run the UNMODIFIED fidelity scorecard against the TORCH sim.

Swaps the ``TempleRunSim-v0`` gym registry entry to the torch adapter
(``sim_env_torch.TorchSimGymAdapter``) and then invokes ``fidelity_scorecard.main()``
verbatim — every check the CPU sim must pass, the torch sim must pass too.

    python scripts/scorecard_torch.py --strict
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import fidelity_scorecard  # noqa: E402  (imports templerun_rl -> registers the CPU env)
import templerun_rl.sim_env_torch  # noqa: E402  (registers TempleRunSimTorch-v0)
from gymnasium.envs.registration import EnvSpec, registry  # noqa: E402

registry["TempleRunSim-v0"] = EnvSpec(
    id="TempleRunSim-v0",
    entry_point="templerun_rl.sim_env_torch:TorchSimGymAdapter",
    max_episode_steps=None,
    disable_env_checker=True,
)
print(">>> TempleRunSim-v0 now aliased to the TORCH sim (sim_env_torch) <<<", flush=True)

if __name__ == "__main__":
    fidelity_scorecard.main()
