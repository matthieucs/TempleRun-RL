#!/usr/bin/env python3
"""OFFLINE GATE BATTERY (wave-10 lesson, 2026-07-09): run a candidate through the multi-config
eval battery vs the champion BEFORE any device time. v71c passed a standard-only gate then
device-regressed on crevasses (3/4 deaths @~600m) — the crevasse-dense config would have caught
it offline in minutes. No candidate goes to the phone without ALL PASS here.

Usage: python scripts/gate_battery.py <candidate.zip> [--champion runs/ppo_v70_consolidated_20000768_steps.zip] [--episodes 96]

Configs (all on top of eval_ckpt_torch's device-true base):
  standard    : {}                                             gate: median >= 0.90x champion
  crevasse    : zone_change_every_cells 9, obstacle_prob .55   gate: median >= 0.90x champion
  stacks      : slidegap .4 jumpnarrow .3 jumpgap .3
                jumpslide .2 turnobstacle .12                  gate: median >= 0.90x champion
  latstress   : turn_send_lat_frames 0.7 + drop/debounce       gate: median >= 1.00x champion (the
                                                               candidate must not be WORSE here)
Also prints per-death-type counts for each config — a skill regression shows up as a death-type
shift (e.g. dt4 Ledge spike = crevasse timing eroded) even when the median holds.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, re

# DEVICE-TRUE EXTRAS: measured fidelity knobs that belong in EVERY eval config (they describe the
# real device, not a training trick). post_land_gap_rejump: 46% landing-frame gap re-jump fire rate
# (WAVE 12, 41/90 device). Without this, a candidate trained on the corrected sim is scored in a sim
# where its (real!) recovery affordance never fires — a false FAIL (bit v74's first battery run).
# WAVE 14 additions (device-measured 2026-07-25): the AIRBORNE TRAP. crevasse_single_cell_prob =
# 67.5% of real crevasse structures are ONE cell (n=409); the 2-cell-only sim pushed every following
# obstacle outside the jump arc so the trap could not form. hole_landing_model = a 1-cell hole
# entered PAST THE ARC APEX kills ~30% on device (frac>=0.51 on every death, <=0.41 on all 858 free
# crossings) while the sim's only gate was binary airborne. Both belong in EVERY eval config for the
# same reason post_land_gap_rejump does: they describe the real device, not a training trick.
DEVICE_TRUE = {"post_land_gap_rejump_prob": 0.46, "crevasse_single_cell_prob": 0.675,
               "hole_landing_model": True, "hole_trap_apex_frac": 0.5, "hole_trap_kill_prob": 0.35}
CONFIGS = {
    "standard": ({**DEVICE_TRUE}, 0.90),
    "crevasse": ({**DEVICE_TRUE, "zone_change_every_cells": 9, "obstacle_prob": 0.55}, 0.90),
    "stacks": ({**DEVICE_TRUE, "slidegap_prob": 0.4, "jumpnarrow_prob": 0.3, "jumpgap_prob": 0.3,
                "jumpslide_prob": 0.2, "turnobstacle_prob": 0.12}, 0.90),
    "latstress": ({**DEVICE_TRUE, "turn_send_lat_frames": 0.7, "turn_offjunction_drop": True,
                   "turn_drop_debounce_s": 0.3}, 1.00),
}
LINE = re.compile(r"(\S+\.zip)\s+median=\s*(\d+)\s+mean=\s*(\d+)\s+max=\s*(\d+)\s+deaths=(\{.*\})")


def run_eval(ckpts, cfg, episodes, max_steps=4000):
    cmd = [sys.executable, "scripts/eval_ckpt_torch.py", *ckpts, "--episodes", str(episodes),
           "--max-episode-steps", str(max_steps)]
    if cfg:
        cmd += ["--sim-config", json.dumps(cfg)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    res = {}
    for m in LINE.finditer(out):
        res[m.group(1).split("\\")[-1].split("/")[-1]] = (int(m.group(2)), m.group(5))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidate")
    # reigning DEPLOY champion (WAVE 13 device A/B: median 5455, 0 turn dead-letters, record 12,779m)
    p.add_argument("--champion", default="runs/ppo_v78_guided_18001920_steps.zip")
    p.add_argument("--episodes", type=int, default=96)
    # WAVE 14: the corrected sim is less punishing on crevasses (67.5% are the easy
    # 1-cell form), so a strong candidate now saturates the old 4000-step cap and the
    # median is CENSORED (v78 scored median 8110 / max 8454 = the cap). Raise it so the
    # gate can actually discriminate.
    p.add_argument("--max-episode-steps", type=int, default=8000)
    a = p.parse_args()
    cand_name = a.candidate.replace("\\", "/").split("/")[-1]
    champ_name = a.champion.replace("\\", "/").split("/")[-1]
    fails = 0
    print(f"GATE BATTERY: {cand_name} vs {champ_name} (n={a.episodes}/config)\n")
    for name, (cfg, thresh) in CONFIGS.items():
        res = run_eval([a.candidate, a.champion], cfg, a.episodes, a.max_episode_steps)
        if cand_name not in res or champ_name not in res:
            print(f"  {name:10} ** EVAL BROKEN — missing result for "
                  f"{[n for n in (cand_name, champ_name) if n not in res]} (obs-width mismatch?) **")
            fails += 1
            continue
        c_med, c_deaths = res[cand_name]
        h_med, h_deaths = res[champ_name]
        ratio = c_med / max(1, h_med)
        ok = ratio >= thresh
        fails += 0 if ok else 1
        print(f"  {name:10} cand {c_med:5} vs champ {h_med:5}  ratio {ratio:.2f} "
              f"(need >={thresh:.2f})  {'PASS' if ok else '** FAIL **'}")
        print(f"  {'':10} deaths cand {c_deaths}  champ {h_deaths}")
    print(f"\n{'GREEN: all gates pass — device A/B is justified' if fails == 0 else f'RED: {fails} gate(s) FAILED — do NOT spend phone time'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
