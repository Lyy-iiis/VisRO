"""Base generator of logic_matrix, in the style of BabyVision's "Logic Patterns" and "Overlay Patterns" tasks.

Abstract-reasoning matrix puzzles: an attribute matrix with a "?" cell, or an
"Overlay Patterns" two-row puzzle in which the 3rd cell is the boolean overlay of
the first two. A matrix of cells follows a hidden rule; the last cell is
replaced by "?"; three or four option panels are drawn INTO the image and exactly
one satisfies the rule. Pick its label.

Representation level: **high** (abstract rule induction + application). The text
never states the rule; you must *induce* it from the visible cells and *apply* it
to predict the missing cell, then match against near-miss distractors. The answer
is not derivable from the question text — it is a property of the pixels.

Two rule families
-----------------
(A) ATTRIBUTE PROGRESSION  (a 2x3 or 3x3 grid of glyphs)
    Each cell is a glyph with attributes (shape, color, count of pips, rotation,
    size). One or more attributes change by a fixed rule along the rows (and the
    rule is consistent across rows so it is uniquely inferable):
      * count   +k per column
      * rotate  +step° per column
      * color   cycle through a fixed color order per column
      * shape   progress through a fixed shape order per column
      * size    grow one step per column
    Difficulty = how many attributes change simultaneously (1 -> 3).

(B) BOOLEAN OVERLAY        ("Overlay Patterns")
    Two rows, 3 cells each. Cells are binary part-patterns over a shared tessellation
    (a grid of triangles / squares / hexagons), each cell some subset "on" (colored).
    The 3rd cell of a row = op(cell1, cell2) where op in {XOR, OR(union),
    AND(intersection), DIFF(set difference)}. The same op holds for both rows, so
    it is inferable from row 1; apply it to row 2 to get the "?" cell.
    Difficulty = op subtlety + how minimally distractors deviate.

How correctness is guaranteed
-----------------------------
Every cell is a structured DESCRIPTOR (attribute dict, or a binary mask). The
correct answer cell is COMPUTED from the rule applied to the givens. Distractors
are the correct cell with a single *named* rule-violation (off-by-one count,
wrong color, wrong rotation step, wrong overlay op, one flipped tile). We then
re-run the rule-checker on EVERY option and assert exactly one passes -> the
answer label is the slot of the unique valid option, never hand-typed.

DIFFICULTY KNOB (rule depth / distractor subtlety)
--------------------------------------------------
  * easy   : 1 changing attribute (A) or OR/AND overlay (B); blatant distractors.
  * medium : 2 attributes (A) or XOR overlay (B); distractors off by one step.
  * hard   : 3 attributes (A) or DIFF/XOR with a dense grid (B); distractors differ
             by a single tile / a single attribute step (a minimal violation).

The problem builders draw from the rng the caller passes (logic_low.build).
"""
from __future__ import annotations
from visro.paths import ROOT as _ROOT

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = (_ROOT / 'assets/fonts')


def _font(size: int, bold: bool = True):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


BG = (255, 255, 255)
PALETTE = [
    (231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15),
    (155, 89, 182), (230, 126, 34), (26, 188, 156), (236, 64, 122),
]
EDGE = (30, 30, 30)
OFF_COLOR = (255, 255, 255)
GRID_LINE = (120, 120, 120)

SHAPES = ["circle", "triangle", "square", "pentagon", "hexagon", "star"]
LABELS = ["A", "B", "C", "D"]


# =========================================================================== A
# ---- attribute-progression matrix
def draw_glyph(d, cx, cy, R, attr):
    """Draw a glyph with attributes: shape, color, count (pips), rot (deg), size."""
    R = R * attr.get("size", 1.0)
    shape = attr["shape"]
    col = attr["color"]
    rot = math.radians(attr.get("rot", 0))
    if shape == "circle":
        d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=col, outline=EDGE, width=3)
    else:
        n = {"triangle": 3, "square": 4, "pentagon": 5, "hexagon": 6, "star": 5}[shape]
        if shape == "star":
            pts = []
            for k in range(2 * n):
                rr = R if k % 2 == 0 else R * 0.45
                a = rot - math.pi / 2 + math.pi * k / n
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
            d.polygon(pts, fill=col, outline=EDGE)
        else:
            pts = []
            for k in range(n):
                a = rot - math.pi / 2 + 2 * math.pi * k / n
                pts.append((cx + R * math.cos(a), cy + R * math.sin(a)))
            d.polygon(pts, fill=col, outline=EDGE)
    # pips: 'count' small dots on an inner ring. They ROTATE WITH the shape (the
    # ring is offset by `rot`) so a rotation change is always visible even on a
    # rotationally-symmetric outline; an extra phase offset keeps a single pip
    # off the symmetry axis.
    cnt = attr.get("count", 0)
    if cnt > 0:
        pr = R * 0.42
        dot = max(3, int(R * 0.13))
        for k in range(cnt):
            a = rot - math.pi / 2 + 0.6 + 2 * math.pi * k / max(cnt, 1)
            px, py = cx + pr * math.cos(a), cy + pr * math.sin(a)
            d.ellipse([px - dot, py - dot, px + dot, py + dot],
                      fill=(255, 255, 255), outline=EDGE, width=2)


def rasterize_glyph(attr, px=96):
    """Render a single glyph to a small RGB array — used to verify that two
    option DESCRIPTORS actually look different (catches rotational-symmetry
    aliasing, e.g. a hexagon rotated by 60° looking identical)."""
    im = Image.new("RGB", (px, px), BG)
    d = ImageDraw.Draw(im)
    draw_glyph(d, px / 2, px / 2, px * 0.36, attr)
    return np.asarray(im, dtype=np.int16)


def glyphs_distinct(a, b, px=96, tol=0.004):
    """True if glyphs a,b render to visibly different bitmaps."""
    A = rasterize_glyph(a, px)
    B = rasterize_glyph(b, px)
    diff = np.abs(A - B).sum()
    return diff > (tol * A.size * 255)


def attr_progression_problem(rng, diff):
    """Build a rows x cols attribute matrix with a per-column rule on N attributes."""
    rows = 2 if diff == "easy" else 3
    cols = {"xhard":4,"xxhard":5}.get(diff,3)
    n_attr = {"easy":3,"medium":4,"hard":5,"xhard":5,"xxhard":5}[diff]

    # pools
    color_cycle = rng.sample(PALETTE, 3)
    shape_cycle = rng.sample(SHAPES, 3)
    rot_step = rng.choice([30, 45, 60, 90])
    base_count = rng.randint(1, 6-cols)
    cnt_step = 1

    # choose which attributes vary across columns
    changeable = ["count", "rot", "color", "shape", "size"]
    rng.shuffle(changeable)
    varying = changeable[:n_attr]

    # fixed (non-varying) defaults, randomized per matrix
    fixed = {
        "shape": rng.choice(SHAPES),
        "color": rng.choice(PALETTE),
        "count": rng.randint(1, 3),
        "rot": rng.choice([0, 30, 45]),
        "size": 1.0,
    }

    row_phase = {a: rng.choice([1,2]) for a in ("color","shape","size")}

    def cell_attr(r, c):
        a = dict(fixed)
        # each row can have a different starting offset, but the per-column
        # *delta* rule is shared, which makes the rule inferable from any row.
        if "count" in varying:
            a["count"] = base_count + r + c * cnt_step
        if "rot" in varying:
            a["rot"] = (fixed["rot"] + 40 * r + sum(rot_step*(1+k%2 if diff=="xxhard" else 1) for k in range(c))) % 360
        if "color" in varying:
            a["color"] = color_cycle[(c + row_phase["color"] * r) % 3]
        if "shape" in varying:
            a["shape"] = shape_cycle[(c + row_phase["shape"] * r) % 3]
        if "size" in varying:
            a["size"] = 0.85 + 0.15 * ((c + row_phase["size"] * r) % 3)
        return a

    grid = [[cell_attr(r, c) for c in range(cols)] for r in range(rows)]
    answer_cell = grid[rows - 1][cols - 1]  # the hidden "?"
    rule_desc = describe_attr_rule(varying, dict(
        rot_step=rot_step, color_cycle=color_cycle, shape_cycle=shape_cycle,
        cnt_step=cnt_step))

    if diff=="xxhard":
        rule_desc=rule_desc.replace(f"the shape rotates {rot_step}° each column",f"rotation increments alternate {rot_step}° then {2*rot_step}°")

    # distractors: minimally violate ONE varying attribute
    options = build_attr_options(answer_cell, varying, rng, diff, dict(
        rot_step=rot_step, color_cycle=color_cycle, shape_cycle=shape_cycle))
    return dict(family="attribute", rows=rows, cols=cols, grid=grid,
                answer_cell=answer_cell, varying=varying, rule_desc=rule_desc,
                options=options)


def describe_attr_rule(varying, params):
    parts = []
    if "count" in varying:
        parts.append("the number of pips increases by 1 each column")
    if "rot" in varying:
        parts.append(f"the shape rotates {params['rot_step']}° each column")
    if "color" in varying:
        parts.append("the color cycles through a fixed 3-color order")
    if "shape" in varying:
        parts.append("the shape progresses through a fixed 3-shape order")
    if "size" in varying:
        parts.append("the size advances through a fixed three-size cycle, wrapping after the largest")
    return "; ".join(parts)


def _attr_equal(a, b):
    keys = ["shape", "color", "count", "rot", "size"]
    for k in keys:
        av, bv = a.get(k), b.get(k)
        if k == "size":
            if abs(av - bv) > 1e-6:
                return False
        elif k == "rot":
            if (av % 360) != (bv % 360):
                return False
        else:
            if av != bv:
                return False
    return True


def build_attr_options(correct, varying, rng, diff, params):
    # Binary codes balance every changed attribute across all four candidates.
    # No candidate is uniquely closest to a consensus of the others.
    def alternate(attr):
        if attr=="count":return max(1,min(7,correct[attr]+rng.choice([-1,1])))
        if attr=="rot":return (correct[attr]+rng.choice([30,45,60]))%360
        if attr=="color":return rng.choice([c for c in PALETTE if c!=correct[attr]])
        if attr=="shape":return rng.choice([c for c in SHAPES if c!=correct[attr]])
        return rng.choice([v for v in (0.85,1.0,1.15) if abs(v-correct[attr])>0.01])
    for _ in range(100):
        attrs=list(varying);rng.shuffle(attrs)
        codes=([1,2,3]+[rng.choice([1,2,3]) for _ in attrs[3:]]) if len(attrs)>=3 else [1,2]
        alts={a:alternate(a) for a in attrs}
        if any(alts[a]==correct[a] for a in attrs):continue
        opts=[]
        for bits in range(4):
            cand=dict(correct)
            for attr,code in zip(attrs,codes):
                if (bits&code).bit_count()%2:cand[attr]=alts[attr]
            opts.append((cand,bits==0,"satisfies the rule" if bits==0 else "balanced attribute violation"))
        if all(glyphs_distinct(opts[i][0],opts[j][0],px=72) for i in range(4) for j in range(i)):
            rng.shuffle(opts);return opts
    raise RuntimeError("No visually distinct balanced attribute choices")


# =========================================================================== B
# ---- boolean overlay matrix (the "Overlay Patterns" family)
def overlay_cells(grid_kind,diff="medium"):
    rank=("easy","medium","hard","xhard","xxhard").index(diff)
    if grid_kind=="square":return 4+rank,4+rank
    if grid_kind=="triangle":return (4,5,6,7,8)[rank],(6,8,10,12,14)[rank]
    return (3,4,5,6,7)[rank],(4,5,6,7,8)[rank]


def tile_polys(grid_kind, R, C, x0, y0, cell):
    """Return list of (idx, polygon-points) for the tessellation."""
    polys = []
    if grid_kind == "square":
        for r in range(R):
            for c in range(C):
                x, y = x0 + c * cell, y0 + r * cell
                polys.append([(x, y), (x + cell, y), (x + cell, y + cell), (x, y + cell)])
    elif grid_kind == "triangle":
        h = cell * math.sqrt(3) / 2
        for r in range(R):
            for c in range(C):
                x = x0 + c * (cell / 2)
                y = y0 + r * h
                up = (c % 2 == 0)
                if up:
                    polys.append([(x, y + h), (x + cell / 2, y), (x + cell, y + h)])
                else:
                    polys.append([(x, y), (x + cell, y), (x + cell / 2, y + h)])
    else:  # hexagon flat-top grid
        w = cell
        h = cell * math.sqrt(3) / 2
        for r in range(R):
            for c in range(C):
                cx = x0 + c * (w * 0.75)
                cy = y0 + r * h + (h / 2 if c % 2 else 0)
                pts = []
                for k in range(6):
                    a = math.pi / 180 * (60 * k)
                    pts.append((cx + (w / 2) * math.cos(a), cy + (w / 2) * math.sin(a)))
                polys.append(pts)
    return polys


def overlay_problem(rng, diff):
    grid_kind = rng.choice(["square", "triangle", "hexagon"])
    R, C = overlay_cells(grid_kind, diff)
    n = R * C

    op = rng.choice(["OR","AND","XOR","DIFF"])

    def rand_mask(density):
        return np.array([1 if rng.random() < density else 0 for _ in range(n)], dtype=int)

    def apply_op(a, b):
        if op == "OR":
            return (a | b)
        if op == "AND":
            return (a & b)
        if op == "XOR":
            return (a ^ b)
        if op == "DIFF":
            return (a & (1 - b))

    # build two rows so the op is non-degenerate (the 3rd cell isn't trivially
    # equal to a given cell, and has some on-tiles).
    def make_row():
        for _ in range(200):
            a = rand_mask(0.42)
            b = rand_mask(0.42)
            c = apply_op(a, b)
            if 2 <= c.sum() <= n - 2 and not np.array_equal(c, a) and not np.array_equal(c, b):
                return a, b, c
        return a, b, apply_op(a, b)

    for _ in range(200):
        r1=make_row(); a,b,_=r1
        demonstrated=[a|b,a&b,a^b,a&(1-b)]
        if len({m.tobytes() for m in demonstrated})==4:break
    else:raise RuntimeError("Overlay demonstration does not identify the operator")
    r2 = make_row()
    answer_mask = r2[2]

    # distractors: minimal violations of the correct overlay mask
    options = build_overlay_options(answer_mask, r2[0], r2[1], op, rng, diff, n)
    on_color = rng.choice([(231, 64, 96), (52, 120, 219), (46, 180, 100), (150, 80, 190)])
    return dict(family="overlay", grid_kind=grid_kind, R=R, C=C, n=n, op=op,
                rows=[r1, r2], answer_mask=answer_mask, options=options,
                on_color=on_color)


def _mask_eq(a, b):
    return np.array_equal(a, b)


def build_overlay_options(correct,a,b,op,rng,diff,n):
    on=list(np.flatnonzero(correct));off=list(np.flatnonzero(1-correct))
    if len(on)<2 or len(off)<2:raise RuntimeError("Overlay needs two on/off pairs")
    rng.shuffle(on);rng.shuffle(off)
    opts=[]
    for bits in range(4):
        cand=correct.copy()
        for k in (0,1):
            if bits&(1<<k):cand[on[k]],cand[off[k]]=0,1
        opts.append((cand,bits==0,"correct overlay" if bits==0 else "equal-area tile swaps"))
    rng.shuffle(opts);return opts


# =========================================================================== render
def render_attr_cell(img, d, attr, x0, y0, box):
    cx, cy = x0 + box / 2, y0 + box / 2
    draw_glyph(d, cx, cy, box * 0.34, attr)


def render_overlay_cell(img,attr_mask,grid_kind,R,C,x0,y0,box,on_color):
    polys=tile_polys(grid_kind,R,C,0,0,1)
    xs=[x for poly in polys for x,y in poly];ys=[y for poly in polys for x,y in poly]
    xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys)
    scale=(box-4)/max(xmax-xmin,ymax-ymin)
    if scale<7:raise RuntimeError("Overlay detail below minimum rendering scale")
    ox=x0+(box-(xmax-xmin)*scale)/2;oy=y0+(box-(ymax-ymin)*scale)/2
    d=ImageDraw.Draw(img)
    for i,poly in enumerate(polys):
        points=[(ox+(x-xmin)*scale,oy+(y-ymin)*scale) for x,y in poly]
        d.polygon(points,fill=on_color if attr_mask[i] else OFF_COLOR,outline=GRID_LINE)


def render_problem_image(prob, size):
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)

    if prob["family"] == "attribute":
        rows, cols = prob["rows"], prob["cols"]
        # matrix occupies top ~62%
        m_top = int(size * 0.05)
        m_h = int(size * 0.55)
        box = min(int(size * 0.86) // cols, m_h // rows)
        gw = box * cols
        gx0 = (size - gw) // 2
        for r in range(rows):
            for c in range(cols):
                x, y = gx0 + c * box, m_top + r * box
                d.rectangle([x, y, x + box, y + box], outline=(190, 190, 190), width=2)
                if r == rows - 1 and c == cols - 1:
                    bq = _font(int(box * 0.45))
                    tb = d.textbbox((0, 0), "?", font=bq)
                    d.text((x + box // 2 - (tb[2] - tb[0]) // 2,
                            y + box // 2 - (tb[3] - tb[1]) // 2 - tb[1]),
                           "?", font=bq, fill=(150, 150, 160))
                else:
                    render_attr_cell(img, d, prob["grid"][r][c], x, y, box)
        opt_top = m_top + rows * box + int(size * 0.03)
        _render_options(img, prob, opt_top, size, kind="attribute")

    else:  # overlay: two rows of 3 cells, "?" at (1,2)
        gk, R, C = prob["grid_kind"], prob["R"], prob["C"]
        m_top = int(size * 0.04)
        box = int(size * 0.26)
        gap = int(size * 0.02)
        gw = box * 3 + gap * 2
        gx0 = (size - gw) // 2
        for r in range(2):
            for c in range(3):
                x = gx0 + c * (box + gap)
                y = m_top + r * (box + gap)
                d.rectangle([x, y, x + box, y + box], outline=(190, 190, 190), width=2)
                if r == 1 and c == 2:
                    bq = _font(int(box * 0.42))
                    tb = d.textbbox((0, 0), "?", font=bq)
                    d.text((x + box // 2 - (tb[2] - tb[0]) // 2,
                            y + box // 2 - (tb[3] - tb[1]) // 2 - tb[1]),
                           "?", font=bq, fill=(150, 150, 160))
                else:
                    render_overlay_cell(img, prob["rows"][r][c], gk, R, C, x, y, box,
                                        prob["on_color"])
        opt_top = m_top + 2 * (box + gap) + int(size * 0.03)
        _render_options(img, prob, opt_top, size, kind="overlay")
    return img


def _render_options(img, prob, top, size, kind):
    d = ImageDraw.Draw(img)
    opts = prob["options"]
    n = len(opts)
    pad = int(size * 0.025)
    panel_w = (size - 2 * pad - (n - 1) * pad) // n
    panel_h = size - top - pad - int(size * 0.04)
    labf = _font(max(14, int(size * 0.036)))
    for i, (cell, _is_c, _desc) in enumerate(opts):
        px = pad + i * (panel_w + pad)
        d.rectangle([px, top, px + panel_w, top + panel_h],
                    fill=(248, 248, 250), outline=(205, 205, 205), width=2)
        d.text((px + 5, top + 3), LABELS[i], font=labf, fill=(20, 20, 20))
        inner = top + int(size * 0.05)
        cbox = min(panel_w - 8, panel_h - int(size * 0.06))
        cx0 = px + (panel_w - cbox) // 2
        if kind == "attribute":
            render_attr_cell(img, d, cell, cx0, inner, cbox)
        else:
            render_overlay_cell(img, cell, prob["grid_kind"], prob["R"], prob["C"],
                                cx0, inner, cbox, prob["on_color"])


def verify_unique(prob):
    """Re-run the rule checker on every option; exactly one must pass.

    Returns (answer_label, is_unique). For attribute: option must equal the
    computed answer_cell. For overlay: option mask must equal op(cell1, cell2).
    """
    opts = prob["options"]
    if prob["family"] == "attribute":
        target = prob["answer_cell"]
        valid = [_attr_equal(c, target) for (c, _i, _d) in opts]
        # PIXEL guard: each non-target option must render visibly different from
        # the target glyph (catches rotational-symmetry aliasing, e.g. a hexagon
        # whose rot differs by a multiple of 60° but looks identical).
        for (c, _i, _d), v in zip(opts, valid):
            if not v and not glyphs_distinct(c, target):
                return None, False
    else:
        a, b = prob["rows"][1][0], prob["rows"][1][1]
        op = prob["op"]
        if op == "OR":
            target = a | b
        elif op == "AND":
            target = a & b
        elif op == "XOR":
            target = a ^ b
        else:
            target = a & (1 - b)
        valid = [_mask_eq(c, target) for (c, _i, _d) in opts]
    n_valid = sum(valid)
    label = LABELS[valid.index(True)] if n_valid == 1 else None
    return label, (n_valid == 1)


# =========================================================================== main
