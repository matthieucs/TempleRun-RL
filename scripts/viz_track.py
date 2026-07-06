#!/usr/bin/env python3
"""Visual track comparison: REAL Temple Run track vs the SIM's generated track.

Renders both as 2D top-down maps so you can eyeball whether the simulator's track *looks* like
the real game — corridor lengths between turns, turn frequency, obstacle density + type mix, the
overall maze shape. The real side is the path the player actually traversed (reconstructed from a
recorded trace); the sim side is the procedural generator's output walked the same way.

Both are laid out identically: start heading North, advance one tile per cell in the current
heading, bend at turn cells. Markers: ^ jump (red), v slide (blue), square turn/junction (green),
faint dot plain. So a faithful sim produces visually similar corridor lengths + obstacle spacing.

Usage:
    python scripts/viz_track.py --trace traces/human.jsonl --out track_compare.png
    python scripts/viz_track.py --trace traces/topaz.jsonl --ep 0 --sim-cells 220 --seed 3
Open the PNG to compare.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# PathDirection (N=1,E=2,S=3,W=4) -> unit step in 2D.
DIR = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
_LEFT = {1: 4, 4: 3, 3: 2, 2: 1}
_RIGHT = {1: 2, 2: 3, 3: 4, 4: 1}


def _kind(c: dict) -> str:
    if c.get("must_jump"):
        return "jump"
    if c.get("must_slide"):
        return "slide"
    if c.get("bend") or c.get("is_turn"):
        return "turn"
    return "plain"


def mark_bends(cells: list) -> list:
    """Flag each cell where the heading changes from the previous cell — the actual path bend
    (a *taken* turn). More reliable than the momentary is_turn junction flag for marking turns."""
    prev = None
    for c in cells:
        c["bend"] = prev is not None and c["facing"] != prev
        prev = c["facing"]
    return cells


# --------------------------------------------------------------------------- real trace -> cells
def cells_from_trace(path: str, ep_index: int) -> list:
    """Walk a recorded trace; return the ordered cells the player traversed (one per crossing)."""
    eps, cur = [], None
    for line in open(path, encoding="utf-8"):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o["t"] == "ep_start":
            cur = []
        elif o["t"] == "step":
            if cur is not None:
                cur.append(o)
        elif o["t"] == "ep_end":
            if cur:
                eps.append(cur)
            cur = None
    if not eps:
        return []
    ep = eps[ep_index] if ep_index < len(eps) else max(eps, key=len)

    def sig(tr):
        return [(t.get("type"), t.get("obstacles"), int(bool(t.get("is_turn")))) for t in tr]

    cells, prev = [], None
    for s in ep:
        st = s["state"]
        tr = sig(s.get("track") or [])
        crossed = False
        if prev is not None and len(tr) >= 3 and len(prev) >= 4:
            k = min(len(tr), len(prev) - 1)
            if tr[:k] == prev[1:1 + k] and tr[:k] != prev[:k]:
                crossed = True
        if not cells or crossed:
            cells.append({"must_jump": bool(st["must_jump"]), "must_slide": bool(st["must_slide"]),
                          "is_turn": bool(st["is_turn"]), "facing": int(st["facing"]) if st["facing"] > 0 else 1})
        prev = tr
    return cells


# --------------------------------------------------------------------------- sim -> cells
def cells_from_sim(n: int, seed: int, config: dict | None = None) -> list:
    """Generate ``n`` procedural cells from the sim and assign a heading by turning at junctions."""
    import gymnasium as gym
    import templerun_rl  # noqa: F401  registers TempleRunSim-v0
    e = gym.make("TempleRunSim-v0", config=config or {}, seed=seed).unwrapped
    e.reset(seed=seed)
    # Walk the player cell-by-cell: advance distance (so the ramp engages) + extend the track
    # ahead, collecting the cell the player is on. This is the real traversed sequence — it includes
    # the straight intro that reset() pre-generates (first turn only once the counter reaches the
    # ramped min spacing), unlike calling _next_cell_procedural() cold.
    cell_len = float(e.cfg["cell_length_real"])
    start = float(e.cfg["start_distance"])
    raw, idx = [], 0
    while len(raw) < n:
        e._distance_real = start + idx * cell_len
        e._cell_index = idx
        e._extend_track()
        if idx < len(e._track):
            raw.append(e._track[idx])
        idx += 1
    cells, facing = [], 1
    for c in raw:
        obst = int(c.get("obstacles", 0))
        cells.append({"must_jump": obst in (1, 4), "must_slide": obst in (5, 6),
                      "is_turn": bool(c.get("is_turn")), "facing": facing})
        if c.get("is_turn"):  # take the turn -> bend the path for following cells
            if c.get("can_turn_left"):
                facing = _LEFT[facing]
            elif c.get("can_turn_right"):
                facing = _RIGHT[facing]
    return cells


# --------------------------------------------------------------------------- layout + render
def layout(cells: list):
    """2D polyline: advance one tile per cell in its heading."""
    xs, ys, x, y = [], [], 0.0, 0.0
    for c in cells:
        xs.append(x)
        ys.append(y)
        dx, dy = DIR.get(c["facing"], (0, 1))
        x += dx
        y += dy
    return xs, ys


def _stats(cells: list) -> str:
    n = len(cells)
    turns = sum(c.get("bend") or c.get("is_turn") for c in cells)
    jumps = sum(c["must_jump"] for c in cells)
    slides = sum(c["must_slide"] for c in cells)
    obst = jumps + slides
    mtt = (n / turns) if turns else float("inf")
    mto = (n / obst) if obst else float("inf")
    return (f"cells={n}  turns={turns} (~1/{mtt:.0f})  "
            f"jumps={jumps} slides={slides} (obst ~1/{mto:.0f})")


def _draw(ax, cells: list, title: str):
    xs, ys = layout(cells)
    ax.plot(xs, ys, "-", color="0.75", lw=1.0, zorder=1)
    styles = {"jump": ("^", "#d62728"), "slide": ("v", "#1f77b4"), "turn": ("s", "#2ca02c")}
    for kind, (m, col) in styles.items():
        px = [xs[i] for i, c in enumerate(cells) if _kind(c) == kind]
        py = [ys[i] for i, c in enumerate(cells) if _kind(c) == kind]
        if px:
            ax.scatter(px, py, marker=m, c=col, s=28, zorder=3, label=f"{kind} ({len(px)})")
    if xs:
        ax.scatter([xs[0]], [ys[0]], marker="o", c="black", s=40, zorder=4, label="start")
    ax.set_title(title + "\n" + _stats(cells), fontsize=9)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", default="traces/human.jsonl", help="recorded trace for the REAL track")
    p.add_argument("--ep", type=int, default=-1, help="episode index in the trace (-1 = longest)")
    p.add_argument("--sim-cells", type=int, default=220, help="how many procedural cells to generate")
    p.add_argument("--seed", type=int, default=3, help="sim generation seed")
    p.add_argument("--out", default="track_compare.png")
    args = p.parse_args()

    real = mark_bends(cells_from_trace(args.trace, args.ep if args.ep >= 0 else 1 << 30))
    sim = mark_bends(cells_from_sim(args.sim_cells, args.seed))
    print(f"REAL ({args.trace}): {_stats(real)}")
    print(f"SIM  (seed={args.seed}): {_stats(sim)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))
    _draw(ax1, real, f"REAL — {Path(args.trace).name}")
    _draw(ax2, sim, "SIM — procedural (device-read statics)")
    fig.suptitle("Temple Run track: real (player path) vs simulator (generated)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
