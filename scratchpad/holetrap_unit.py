#!/usr/bin/env python3
"""WAVE 14 unit check: on a HAND-BUILT airborne-trap track (crevasse -> N clear -> single 1-cell
hole), sweep the takeoff cell/cp and report, for hole_landing_model OFF vs ON, whether the runner
survives. Answers 'does the new check actually fire, and at what d?' without any policy noise."""
from __future__ import annotations
import sys
from pathlib import Path
_R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_R / "src"))
sys.path.insert(0, str(_R / "scripts"))
from sim_equivalence import make_cpu, cell, S, DET_BASE  # noqa: E402

CLEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 0     # clear cells between crevasse and hole
TRACK = (S(8) + [cell(type=35, obstacles=7, wide_gap=True, level=3),
                 cell(type=35, obstacles=7, wide_gap=True, stepup=True, level=3)]
         + S(CLEAR) + [cell(type=1, obstacles=1)] + S(12))
HOLE_CELL = 1 + 8 + 2 + CLEAR      # +1 for the leading cell make_cpu prepends


def trial(jump_step, model_on, vel=200.0):
    cfg = dict(DET_BASE)
    cfg.update({"wide_gap_landing_model": True, "wide_gap_land_grace": 0.35,
                "crevasse_graze_worldy": True, "crevasse_cap_arc_offset": 0.4,
                "crevasse_takeoff_std": 0.0, "start_velocity": vel, "velocity_curve": None,
                "jump_exec_lat_frames": 0, "jump_exec_lat_jitter": 0})
    if model_on:
        cfg.update({"hole_landing_model": True, "hole_trap_apex_frac": 0.5, "hole_trap_kill_prob": 1.0})
    env, obs, st = make_cpu(TRACK, cfg)
    env._velocity = vel
    took_off = None
    for t in range(60):
        act = 2 if t == jump_step else 6
        env.step(act)
        if took_off is None and env._jump_t > 0.0:
            took_off = env._jump_takeoff_pos
        if env._is_dead:
            return took_off, env._cell_index, env._death_type
        if env._cell_index > HOLE_CELL:
            return took_off, env._cell_index, 0
    return took_off, env._cell_index, 0


print(f"airborne-trap track: crevasse(2) + {CLEAR} clear + single hole @cell {HOLE_CELL}")
print(" jump_step  takeoff_pos     d      OFF: died@cell/dt      ON: died@cell/dt")
for js in range(6, 40):
    tk_off, c_off, d_off = trial(js, False)
    tk_on, c_on, d_on = trial(js, True)
    if tk_off is None:
        continue
    d = HOLE_CELL - tk_off
    print(f"   {js:4}      {tk_off:8.2f}   {d:5.2f}    "
          f"{'ALIVE' if not d_off else f'dead c{c_off} dt{d_off}':>20}  "
          f"{'ALIVE' if not d_on else f'dead c{c_on} dt{d_on}':>20}")
