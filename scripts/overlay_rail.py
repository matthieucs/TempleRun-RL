#!/usr/bin/env python3
"""The action rail: six control icons drawn with Pillow, one ring lit.

Geometry is expressed as FRACTIONS of the frame, never pixels, so the same code renders correctly
for any device resolution or capture orientation (measured 2026-07-28: the phone is 1080x2340 =
9:19.5, NOT the 9:16 one might assume, and the game paints full-bleed with no letterbox bars).

Icon proportions are read off the user's reference image (1834px wide, bar 1708x305):
  bar height / bar width      305 / 1708 = 0.179
  icon centre pitch           276 / 1708 = 0.162, first centre 163 / 1708 = 0.095
  arrow height / bar height   115 / 305  = 0.377
  ring diameter / bar height  215 / 305  = 0.705
  stroke width / bar height    17 / 305  = 0.056
Arrowheads are OPEN CHEVRONS with round caps at the same weight as the shaft -- not filled
triangles. That is the single most visible characteristic of the reference set.

Everything is drawn at SS x supersample and downscaled with Lanczos, which antialiases the ring,
the chevrons and the rotated phone without any special-casing. Only seven images are ever produced
(one per lit cell, plus nothing lit), so the cost is irrelevant.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------- layout (fractions)
RAIL_W_FRAC       = 0.78     # rail width as a fraction of frame width
RAIL_H_OF_W       = 0.179    # rail height as a fraction of rail width
RAIL_BOTTOM_FRAC  = 0.034    # gap below the rail, fraction of frame height
CORNER_OF_RAIL_H  = 0.16     # corner radius, fraction of rail height

CELL_FIRST_FRAC   = 0.095    # first icon centre, fraction of rail width
CELL_PITCH_FRAC   = 0.162    # spacing between icon centres, fraction of rail width

ICON_H_OF_RAIL    = 0.377    # nominal icon height, fraction of rail height
STROKE_OF_RAIL    = 0.056
RING_D_OF_RAIL    = 0.705
RING_W_OF_RAIL    = 0.028
GLOW_BLUR_OF_RAIL = 0.095
PAD_OF_RAIL       = 0.34     # strip padding around the rail so the glow is not clipped

# ---------------------------------------------------------------- colour
PANEL_FILL = (18, 18, 18, 232)   # opaque enough to stay legible over the bright death card
PANEL_EDGE = (255, 255, 255, 64)
ICON_COL   = (255, 255, 255, 255)
RING_COL   = (255, 150, 26, 255)
GLOW_COL   = (255, 132, 0, 205)

SS = 4  # supersample factor

# rail order, left to right, and the action byte each cell corresponds to
RAIL_ORDER = ("TL", "SL", "JP", "SD", "SR", "TR")
BYTE_TO_KEY = {4: "TL", 0: "SL", 2: "JP", 3: "SD", 1: "SR", 5: "TR"}  # 6 = NoOp -> nothing lit
KEY_LABEL = {
    "TL": "turn left", "SL": "tilt left", "JP": "jump",
    "SD": "slide", "SR": "tilt right", "TR": "turn right",
}


# ---------------------------------------------------------------- stroke helpers
def _cap(d: ImageDraw.ImageDraw, p: Sequence[float], w: float, col) -> None:
    r = w / 2.0
    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=col)


def _stroke(d: ImageDraw.ImageDraw, pts: Sequence[Sequence[float]], w: float, col,
            caps: bool = True) -> None:
    """Polyline with rounded joints and rounded caps (Pillow rounds joints, not caps)."""
    pts = [(float(x), float(y)) for x, y in pts]
    if len(pts) >= 2:
        d.line(pts, fill=col, width=max(1, int(round(w))), joint="curve")
    if caps:
        _cap(d, pts[0], w, col)
        _cap(d, pts[-1], w, col)


def _bezier(p0, p1, p2, n: int = 16) -> list[tuple[float, float]]:
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        out.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                    u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))
    return out


# ---------------------------------------------------------------- icons
# Each icon is drawn into a box of side `size` centred on (cx, cy). Coordinates below are in
# units of `size` with the origin at the centre, which keeps them readable and resolution-free.

def _pt(cx, cy, size, x, y):
    return (cx + x * size, cy + y * size)


def icon_arrow(d, cx, cy, size, w, col, down: bool = False) -> None:
    """Straight arrow with an open chevron head. Jump (up) and slide (down)."""
    s = -1.0 if down else 1.0          # s = +1 -> tip at the top
    tip = _pt(cx, cy, size, 0.0, -0.50 * s)
    tail = _pt(cx, cy, size, 0.0, 0.50 * s)
    barb_l = _pt(cx, cy, size, -0.26, -0.20 * s)
    barb_r = _pt(cx, cy, size, 0.26, -0.20 * s)
    _stroke(d, [tail, tip], w, col)
    _stroke(d, [barb_l, tip, barb_r], w, col)


def icon_turn(d, cx, cy, size, w, col, right: bool = False) -> None:
    """Bent arrow: tail rises on one side, rounded elbow, head points outward."""
    m = -1.0 if right else 1.0         # m = +1 -> head points left

    tail_bottom = _pt(cx, cy, size, 0.30 * m, 0.50)
    elbow_in    = _pt(cx, cy, size, 0.30 * m, -0.06)
    elbow_ctl   = _pt(cx, cy, size, 0.30 * m, -0.30)
    elbow_out   = _pt(cx, cy, size, 0.06 * m, -0.30)
    tip         = _pt(cx, cy, size, -0.34 * m, -0.30)

    shaft = [tail_bottom, elbow_in]
    shaft += _bezier(elbow_in, elbow_ctl, elbow_out)
    shaft += [tip]
    _stroke(d, shaft, w, col)

    barb_a = _pt(cx, cy, size, -0.12 * m, -0.52)
    barb_b = _pt(cx, cy, size, -0.12 * m, -0.08)
    _stroke(d, [barb_a, tip, barb_b], w, col)


# phone proportions, all as fractions of the nominal icon height, measured off the reference
# (phone 62 x 95 px, outline 14 px, against an arrow 115 px tall):
PH_W_OF_ICON      = 0.54     # 62 / 115
PH_H_OF_ICON      = 0.83     # 95 / 115
PH_STROKE_OF_ICON = 0.105    # 14 / 115  -- thinner than the arrow stroke, or the body fills in
PH_RADIUS_OF_W    = 0.16
PH_TILT_DEG       = 20.0
ARC_STROKE_OF_ICON = 0.062
ARC_R1_OF_ICON     = 0.55
ARC_R2_OF_ICON     = 0.68


def icon_tilt(layer: Image.Image, cx, cy, size, w, col, right: bool = False) -> None:
    """A phone leaning over, with motion arcs -- the analog accelerometer hold.

    The body is drawn into its own layer and rotated, because Pillow cannot rotate a rounded
    rectangle in place. The motion arcs are circular arcs about the icon centre, which is exactly
    what a rotation indicator should be.

    The phone's own stroke is deliberately thinner than the arrows' (0.12 vs 0.148 of icon height).
    At the arrow weight the two side strokes very nearly meet across a body this narrow and the
    icon reads as a filled blob; the reference keeps ~55 % of the body width as interior.
    """
    ph_w, ph_h = PH_W_OF_ICON * size, PH_H_OF_ICON * size
    ph_stroke = max(1.0, PH_STROKE_OF_ICON * size)
    box = int(math.ceil(size * 1.5))
    sub = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sub)
    x0, y0 = (box - ph_w) / 2.0, (box - ph_h) / 2.0
    # inset by half the stroke so the outline's outer edge lands on the nominal body rectangle
    h = ph_stroke / 2.0
    sd.rounded_rectangle([x0 + h, y0 + h, x0 + ph_w - h, y0 + ph_h - h],
                         radius=PH_RADIUS_OF_W * ph_w,
                         outline=col, width=int(round(ph_stroke)))
    # the dot must clear the inner edge of the bottom stroke, else it merges into it and vanishes
    dot_r = 0.032 * size
    dot_cy = y0 + ph_h - (PH_STROKE_OF_ICON + dot_r / size + 0.045) * size
    sd.ellipse([box / 2.0 - dot_r, dot_cy - dot_r, box / 2.0 + dot_r, dot_cy + dot_r], fill=col)

    angle = PH_TILT_DEG if not right else -PH_TILT_DEG   # PIL rotates CCW: +20 leans the top left
    sub = sub.rotate(angle, resample=Image.BICUBIC, expand=False)
    layer.alpha_composite(sub, (int(round(cx - box / 2.0)), int(round(cy - box / 2.0))))

    # motion arcs. PIL angles: 0 deg = 3 o'clock, increasing clockwise because y points down.
    # Two arcs on the leading corner, one on the trailing corner, as in the reference.
    d = ImageDraw.Draw(layer)
    aw = max(1, int(round(ARC_STROKE_OF_ICON * size)))
    if not right:
        arcs = [(ARC_R1_OF_ICON, 200, 232), (ARC_R2_OF_ICON, 204, 228), (ARC_R1_OF_ICON, 20, 52)]
    else:
        arcs = [(ARC_R1_OF_ICON, -52, -20), (ARC_R2_OF_ICON, -48, -24), (ARC_R1_OF_ICON, 128, 160)]
    for rad_f, a0, a1 in arcs:
        r = rad_f * size
        d.arc([cx - r, cy - r, cx + r, cy + r], a0, a1, fill=col, width=aw)


def _draw_icon(layer: Image.Image, key: str, cx: float, cy: float, size: float, w: float, col):
    d = ImageDraw.Draw(layer)
    if key == "JP":
        icon_arrow(d, cx, cy, size, w, col, down=False)
    elif key == "SD":
        icon_arrow(d, cx, cy, size, w, col, down=True)
    elif key == "TL":
        icon_turn(d, cx, cy, size, w, col, right=False)
    elif key == "TR":
        icon_turn(d, cx, cy, size, w, col, right=True)
    elif key == "SL":
        icon_tilt(layer, cx, cy, size, w, col, right=False)
    elif key == "SR":
        icon_tilt(layer, cx, cy, size, w, col, right=True)
    else:
        raise ValueError(f"unknown icon key {key!r}")


# ---------------------------------------------------------------- the rail
class Rail:
    """Pre-renders the seven possible rail images for a given frame size.

    The rail only ever looks seven ways, so per-frame work during compositing collapses to
    selecting a buffer and writing it to a pipe.
    """

    def __init__(self, frame_w: int, frame_h: int, *,
                 rail_w_frac: float = RAIL_W_FRAC,
                 rail_h_of_w: float = RAIL_H_OF_W,
                 bottom_frac: float = RAIL_BOTTOM_FRAC) -> None:
        self.frame_w, self.frame_h = int(frame_w), int(frame_h)
        self.rail_w = self.frame_w * rail_w_frac
        self.rail_h = self.rail_w * rail_h_of_w
        pad = self.rail_h * PAD_OF_RAIL

        self.strip_h = int(round(self.rail_h + 2 * pad))
        self.strip_y = int(round(self.frame_h - self.frame_h * bottom_frac
                                 - self.rail_h - pad))
        self.strip_y = max(0, min(self.strip_y, self.frame_h - self.strip_h))

        self._pad = pad
        self.states: dict[str | None, Image.Image] = {}
        for key in (None, *RAIL_ORDER):
            self.states[key] = self._render(key)

    # -- internals -------------------------------------------------
    def _render(self, lit: str | None) -> Image.Image:
        W, H = self.frame_w * SS, self.strip_h * SS
        rail_w, rail_h = self.rail_w * SS, self.rail_h * SS
        pad = self._pad * SS
        x0 = (W - rail_w) / 2.0          # centred, as requested
        y0 = pad
        stroke = self.rail_h * SS * STROKE_OF_RAIL
        icon_sz = self.rail_h * SS * ICON_H_OF_RAIL
        ring_r = self.rail_h * SS * RING_D_OF_RAIL / 2.0
        ring_w = self.rail_h * SS * RING_W_OF_RAIL
        cy = y0 + rail_h / 2.0

        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([x0, y0, x0 + rail_w, y0 + rail_h],
                            radius=rail_h * CORNER_OF_RAIL_H,
                            fill=PANEL_FILL, outline=PANEL_EDGE,
                            width=max(1, int(round(2 * SS * self.frame_w / 1080))))

        centres = [x0 + rail_w * (CELL_FIRST_FRAC + i * CELL_PITCH_FRAC) for i in range(6)]

        if lit is not None:
            i = RAIL_ORDER.index(lit)
            cx = centres[i]
            glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow)
            gd.ellipse([cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
                       outline=GLOW_COL, width=max(1, int(round(ring_w * 2.4))))
            glow = glow.filter(ImageFilter.GaussianBlur(self.rail_h * SS * GLOW_BLUR_OF_RAIL))
            # composited twice: one pass reads as a faint smudge, the reference has a real bloom
            img.alpha_composite(glow)
            img.alpha_composite(glow)
            d = ImageDraw.Draw(img)
            d.ellipse([cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
                      outline=RING_COL, width=max(1, int(round(ring_w))))

        for i, key in enumerate(RAIL_ORDER):
            _draw_icon(img, key, centres[i], cy, icon_sz, stroke, ICON_COL)

        return img.resize((self.frame_w, self.strip_h), Image.LANCZOS)

    # -- public ----------------------------------------------------
    def frame(self, lit: str | None) -> Image.Image:
        return self.states[lit]

    def sheet(self, background: tuple[int, int, int] = (26, 30, 28)) -> Image.Image:
        """All seven states stacked, for eyeballing the icons without rendering a video."""
        one = self.states[None]
        w, h = one.size
        out = Image.new("RGB", (w, h * 7), background)
        for i, key in enumerate((None, *RAIL_ORDER)):
            out.paste(self.states[key], (0, i * h), self.states[key])
        return out
