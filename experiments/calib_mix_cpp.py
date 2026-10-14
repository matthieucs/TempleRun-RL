#!/usr/bin/env python3
"""Fit obstacle_type_weights so the C++ TRAINER's EMITTED structure mix matches the device.

    python experiments/calib_mix_cpp.py

Measures the trainer directly via sim_probe_cells (no policy, no Python sim). The emitted share is
NOT the weight table -- holes also arrive via wide-gap/compound paths that bypass the type draw --
so this iterates weight *= target/emitted until it converges.

DEVICE TARGET, measured 2026-07-28 over 15 captures / 3316 structures, obstacle types 1-6, STRUCTURE
units (adjacent same-type cells collapsed) -- the identical statistic computed the identical way on
both sides. State the basis with the number, always.
"""
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import ctypes, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp_sim"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cpp_vec_env as CV

NAME = {1: "Hole", 2: "Narrow", 3: "Stumble", 4: "TreeJump", 5: "TreeSlide", 6: "Flame"}
DEVICE = {1: 33.1, 2: 14.7, 3: 11.0, 4: 11.4, 5: 12.0, 6: 17.7}


def emitted(cfg, n_seeds=24, n_cells=6000, start_dist=3000.0):
    L = CV.lib()
    L.sim_probe_cells.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_double,
                                  ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int)]
    conf = CV._Config(cfg, "procedural", 0.1, 1000, 0.1, "distance", 1.0, 4, 1, {})
    conf.install()
    buf = ctypes.create_string_buffer(L.sim_state_size())
    ob = (ctypes.c_int * n_cells)()
    counts = {}
    for s in range(n_seeds):
        L.sim_probe_cells(buf, s, start_dist, n_cells, ob, None)
        prev = 0
        for i in range(n_cells):
            o = ob[i]
            if o and o != prev and o in NAME:
                counts[o] = counts.get(o, 0) + 1
            prev = o
    tot = sum(counts.values()) or 1
    return {k: 100.0 * counts.get(k, 0) / tot for k in NAME}


def main():
    base = json.load(open("experiments/_v108_cfg.json"))
    w = {int(k): float(v) for k, v in base["obstacle_type_weights"].items()}
    print("target (device):  " + "  ".join("%s %.1f%%" % (NAME[k], DEVICE[k]) for k in NAME))
    for it in range(6):
        cfg = dict(base); cfg["obstacle_type_weights"] = {str(k): v for k, v in w.items()}
        em = emitted(cfg)
        err = max(abs(em[k] - DEVICE[k]) for k in NAME)
        print("iter %d  max-err %5.2f pts  |  " % (it, err)
              + "  ".join("%s %.1f" % (NAME[k], em[k]) for k in NAME))
        if err < 1.0:
            break
        for k in NAME:                       # weights are not the emitted share -> iterate
            if em[k] > 0.05:
                w[k] *= DEVICE[k] / em[k]
        tot = sum(w.values())
        w = {k: v / tot for k, v in w.items()}
    print("\nfitted obstacle_type_weights:")
    print(json.dumps({str(k): round(v, 4) for k, v in w.items()}, indent=1))
    out = Path("experiments/_v114_cfg.json")
    base["obstacle_type_weights"] = {str(k): round(v, 4) for k, v in w.items()}
    out.write_text(json.dumps(base, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
