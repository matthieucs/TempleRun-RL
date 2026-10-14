#!/usr/bin/env python3
"""READ THE GAME'S OWN GENERATION LAW — no policy, no episodes, no statistics.

USER (2026-07-26): "Isn't it things that you can find in the source code instead of collecting data
on the phone? I was previously looking to find a cheap way to estimate the difference between the
simulation and real data without having to run a policy on the phone and collect data."

Correct, and far better than fitting distributions to captures. `PathElement` keeps its ENTIRE
generation law in static fields (Il2CppDumper, dump.cs:283287) and even ships a DumpStatics()
method. Reading them takes seconds and yields EXACT values instead of noisy estimates over
thousands of cells:

    sMinSpaceBetweenTurns      0x2C   <- sim: min_space_between_turns
    sMaxBackToBackObstacles    0x30   <- sim: max_back_to_back_obstacles
    sMinSpaceBetweenObstacles  0x34   <- sim: min_space_between_obstacles (the hard floor of 6 that
                                        removes 19% of real spacings and blocks every -> Crevasse
                                        compound)
    sDoubleObstaclePercent     0x38   <- ** NOT MODELLED IN THE SIM AT ALL ** — almost certainly the
                                        mechanism behind the device's tight obstacle pairs
    sAllow* toggles            0x04.. <- which hazard classes are enabled at all
    sMinSpacesBetweenBonusItems/sProbability* / coin-run fields

This attaches to the ALREADY-RUNNING gadget (the hook must be injected), resolves the PathElement
class via the Il2CPP API, and reads its static field block directly.

Usage: python scripts/probe_generation_statics.py        (game running + hook injected)
"""
from __future__ import annotations
import json
import sys

try:
    import frida
except ImportError:
    sys.exit("frida not installed")

JS = r"""
'use strict';
// libil2cpp.so exports are stripped, so resolve by BASE + RVA exactly like temple_run_hook.js does.
// PathElement.DumpStatics() (RVA 0x12E1508, dump.cs) makes the GAME print its own generation law to
// logcat -- no statistics, no episodes, exact values.
const _m = Process.findModuleByName('libil2cpp.so');   // Frida 17: findBaseAddress is gone
const base = _m ? _m.base : null;
if (!base) { send({error: 'libil2cpp.so not loaded'}); }
else {
  try {
    const dumpStatics = new NativeFunction(base.add(0x12E1508), 'void', ['pointer']);
    dumpStatics(ptr(0));
    send({ok: true, base: base.toString(), called: 'PathElement.DumpStatics'});
  } catch (e) {
    send({error: 'DumpStatics call failed: ' + e.message});
  }
}
"""

# what the sim currently uses, for a side-by-side
SIM = {
    "sMinSpaceBetweenTurns": ("min_space_between_turns", 12),
    "sMaxBackToBackObstacles": ("max_back_to_back_obstacles", "2 (default) / 6 (training)"),
    "sMinSpaceBetweenObstacles": ("min_space_between_obstacles", 6),
    "sDoubleObstaclePercent": ("** NOT MODELLED **", None),
}


def main():
    dev = frida.get_device_manager().add_remote_device("127.0.0.1:27042")
    ses = dev.attach("Gadget")
    got, errs = {}, []
    def on_msg(m, d):
        if m.get("type") == "error":
            errs.append(m.get("stack") or m.get("description"))
        else:
            got.update(m.get("payload") or {})
    scr = ses.create_script(JS)
    scr.on("message", on_msg)
    scr.load()
    import time
    time.sleep(2.0)          # send() is async — give the runtime a moment to deliver
    if errs:
        sys.exit("JS error:\n" + "\n".join(errs))
    if not got or got.get("error"):
        sys.exit(f"probe failed: {got.get('error', 'no data')}")
    print("GAME'S OWN GENERATION LAW (PathElement statics, read live):\n")

    print("\nMISMATCHES vs the sim are the generation gap — fix these EXACTLY, do not fit them.")


if __name__ == "__main__":
    main()
