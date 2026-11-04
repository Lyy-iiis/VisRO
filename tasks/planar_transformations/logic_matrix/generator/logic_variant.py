"""Visual variants for the boolean grids of logic_matrix hard/xhard.

WHY
---
hard, xhard and xxhard all render as the same filled-square lattice, so the
three tiers look interchangeable even though the logic under them differs
(2 vs 3 composition stages, 1 vs 2 rotations).  A single visual vocabulary also
lets a model overfit one layout instead of reading the rule.

WHERE THIS LIVES
----------------
`boolean_grids.py` stays authoritative for geometry and truth values; this
module only REPAINTS what it already published in `rec['render_regions']` --
each entry carries `bounds`, `cell` and the `values` matrix, which is exactly
enough to redraw a grid in a different style without recomputing any logic.

TWO INVARIANTS THAT MUST NOT BREAK
----------------------------------
1.  `adapter.check()` samples the CENTRE PIXEL of every cell and asserts it
    equals ON or OFF exactly:

        assert im.getpixel((x+c*cell+cell//2, y+r*cell+cell//2)) == (ON if v else OFF)

    That assertion is what proves the image really encodes the truth values.
    Every style below therefore paints the centre exactly ON (for 1) or exactly
    OFF (for 0) -- work around the palette, never around the check.

2.  The variant must never reach the dedup key.  `adapter.core_key()`
    whitelists ('n','operators','rotations','inputs') for logic_matrix, so the
    style is derived from the generation index and recorded in
    `meta['render_style']` for traceability only.  If it ever entered the core
    key, one logical problem drawn four ways would count as four distinct
    problems and the capacity of the task would be overstated.

The question text says "Filled cells represent 1 and light cells 0", so every
style keeps 1 dark-and-filled and 0 light -- only the mark changes.
"""
from PIL import Image, ImageDraw

# The first four are plain marks, the last three are more elaborate: hard, xhard
# and xxhard are all boolean grids, so a single rendering would leave them one
# visual vocabulary.
STYLES = ('square', 'disc', 'tile', 'dotmatrix', 'bevel', 'concentric', 'hatch')
ON = (40, 65, 100)
OFF = (249, 249, 246)
EDGE = (120, 125, 132)
FAINT = (206, 211, 217)
DEEP = (24, 42, 68)

# Palettes, taken from COLORS in shared/puzzles/drawing.py -- the set the
# easy/medium attribute questions use, so the boolean grids look like the same
# task family instead of carrying a palette of their own.  The yellow
# (231,182,37) is left out on purpose: against the near-white background it is
# too light for the "filled = dark" convention to read.
#
# adapter.check verifies the centre pixel against meta['render_palette'] and
# also asserts that the two colours are >= 120 apart in Manhattan distance, so
# a degenerate palette cannot turn the check into a no-op.
PALETTES = (
    ('ink',    (40, 65, 100),  (249, 249, 246)),
    ('red',    (199, 49, 58),  (249, 249, 246)),
    ('blue',   (42, 103, 193), (249, 249, 246)),
    ('green',  (40, 151, 99),  (249, 249, 246)),
    ('violet', (142, 78, 172), (249, 249, 246)),
)


def palette_for(rec):
    """The second variation axis: the style rotates with the index, the
    palette with index // len(STYLES)."""
    return PALETTES[(rec['generation']['index'] // len(STYLES)) % len(PALETTES)]


def _shade(rgb, factor):
    """Scale a colour lighter (>1) or darker (<1); used for bevel highlights and hatching."""
    return tuple(max(0, min(255, int(round(v * factor)))) for v in rgb)


def _stamp_centre(draw, x0, y0, x1, y1, value, on, off):
    """Stamp a small centre patch with the exact ON/OFF colour.

    adapter.check() asserts the single pixel (x0+cell//2, y0+cell//2).
    Decorative texture (hatching, rings, highlight edges) can land on that
    point, so the elaborate styles draw their decoration first and call this
    last to restore the exact centre value -- the texture works around the
    check, not the other way round.
    """
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    pad = 1 if (x1 - x0) < 20 else 2
    draw.rectangle((cx - pad, cy - pad, cx + pad, cy + pad), fill=on if value else off)


def style_for(rec):
    """Deterministic from the generation index, so a record redraws identically."""
    return STYLES[rec['generation']['index'] % len(STYLES)]


def paint(draw, x0, y0, x1, y1, value, style, on=ON, off=OFF):
    """Draw one cell.  In every style the centre pixel is exactly `on` or `off`."""
    lit_on, dim_on = _shade(on, 1.55), _shade(on, .55)
    lit_off, dim_off = (255, 255, 253), _shade(off, .83)
    deep = _shade(on, .6)
    if style == 'square':
        draw.rectangle((x0, y0, x1, y1), fill=on if value else off, outline=EDGE, width=1)
    elif style == 'disc':
        draw.rectangle((x0, y0, x1, y1), fill=off, outline=EDGE, width=1)
        if value:
            pad = max(1, (x1 - x0) // 6)
            draw.ellipse((x0 + pad, y0 + pad, x1 - pad, y1 - pad), fill=on)
    elif style == 'tile':
        # Separated mosaic: no shared borders, so the lattice reads as objects.
        pad = max(1, (x1 - x0) // 8)
        draw.rectangle((x0, y0, x1, y1), fill='white')
        draw.rectangle((x0 + pad, y0 + pad, x1 - pad, y1 - pad),
                       fill=on if value else off,
                       outline=on if value else EDGE, width=1)
    elif style == 'bevel':
        # Bevelled tile: highlight top/left, shadow bottom/right; 1 is raised,
        # 0 is sunken.  The style with the most depth.
        base = on if value else off
        lit = lit_on if value else lit_off
        dim = dim_on if value else dim_off
        draw.rectangle((x0, y0, x1, y1), fill=base)
        w = max(1, (x1 - x0) // 7)
        for i in range(w):
            draw.line((x0 + i, y0 + i, x1 - i, y0 + i), fill=lit)
            draw.line((x0 + i, y0 + i, x0 + i, y1 - i), fill=lit)
            draw.line((x0 + i, y1 - i, x1 - i, y1 - i), fill=dim)
            draw.line((x1 - i, y0 + i, x1 - i, y1 - i), fill=dim)
        _stamp_centre(draw, x0, y0, x1, y1, value, on, off)
    elif style == 'concentric':
        # Concentric squares: 1 is a solid cell with a light inner ring, 0 is a
        # light cell with a thin inner frame.
        draw.rectangle((x0, y0, x1, y1), fill=off, outline=EDGE, width=1)
        step = max(2, (x1 - x0) // 5)
        if value:
            draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), fill=on)
            draw.rectangle((x0 + step, y0 + step, x1 - step, y1 - step), outline=lit_on, width=1)
        else:
            draw.rectangle((x0 + step, y0 + step, x1 - step, y1 - step), outline=FAINT, width=1)
        _stamp_centre(draw, x0, y0, x1, y1, value, on, off)
    elif style == 'hatch':
        # Hatching: 1 is a dark cell with darker diagonals, 0 a light cell with
        # very faint ones.  The texture density separates the two states.
        #
        # The diagonals are drawn on a cell-sized tile and pasted back.  Drawn
        # directly on the canvas, their ends would cross the cell border and
        # overwrite the centre pixel already stamped on the left neighbour.
        w, h = x1 - x0, y1 - y0
        tile = Image.new('RGB', (w, h), on if value else off)
        td = ImageDraw.Draw(tile)
        gap = max(3, w // 4)
        line = deep if value else FAINT
        for k in range(-h, w + h, gap):
            td.line((k, 0, k + h, h), fill=line)
        td.rectangle((0, 0, w - 1, h - 1), outline=EDGE, width=1)
        draw._image.paste(tile, (x0, y0))
        _stamp_centre(draw, x0, y0, x1, y1, value, on, off)
    else:  # dotmatrix -- faint lattice, a mark only where the value is 1
        draw.rectangle((x0, y0, x1, y1), fill=OFF, outline=FAINT, width=1)
        if value:
            pad = max(1, (x1 - x0) // 4)
            draw.ellipse((x0 + pad, y0 + pad, x1 - pad, y1 - pad), fill=on)


def _repaint(im, regions, style, on, off):
    draw = ImageDraw.Draw(im)
    for region in regions:
        x0, y0 = region['bounds'][0], region['bounds'][1]
        cell = region['cell']
        for r, row in enumerate(region['values']):
            for c, value in enumerate(row):
                paint(draw, x0 + c * cell, y0 + r * cell,
                      x0 + (c + 1) * cell, y0 + (r + 1) * cell, value, style, on, off)
    return im


def restyle(rec, im):
    """Repaint a parent-rendered image in this record's style.  'square' is the
    parent's own look, so it is returned untouched rather than redrawn."""
    style = style_for(rec)
    name, on, off = palette_for(rec)
    rec['meta']['render_style'] = style
    rec['meta']['render_palette'] = [list(on), list(off)]
    rec['meta']['render_palette_name'] = name
    if style == 'square' and name == 'ink':
        # The generator already draws square + ink; nothing to change.
        return im
    regions = rec.get('render_regions')
    if not regions:
        # Nothing published to repaint from: keep the generator's image rather
        # than guessing at geometry.  A missing region list is a generator
        # change, not something to paper over.
        return im
    return _repaint(im.copy(), regions, style, on, off)


# ---------------------------------------------------------------------------
# xxhard is drawn by shared/puzzles/symbolic.py, whose records carry no
# render_regions, so the repaint-from-published-geometry path above does not
# apply.  Here the geometry is recomputed from meta and checked against the
# original image.
#
# Recomputing is safe because the layout of shared/puzzles/symbolic.py:logic()
# is a pure function of n, the number of stages and the canvas width, with no
# randomness.  It is still checked because that module is shared: the check is
# the evidence that the recomputed geometry matches what was drawn.  On any
# mismatch the image is returned unchanged.
#
# oracles.check_render verifies only size and labels for composed_boolean, so
# this pixel check adds coverage rather than replacing any.
SYMBOLIC_LOGIC_FAMILY = 'composed_boolean'


def symbolic_logic_regions(rec, im):
    """Recompute the grid geometry of a symbolic.logic item from meta and check
    it cell by cell against the image; None on any mismatch."""
    m = rec.get('meta') or {}
    if m.get('family') != SYMBOLIC_LOGIC_FAMILY:
        return None
    try:
        n = m['n']
        stages = len(m['operators'])
        rank = ('hard', 'xhard', 'xxhard').index(rec['difficulty'])
        cell = (12, 12, 11)[rank]
        tile = n * cell
        width = im.width
        large = width > 448
        gap = 18 if large else 14
        rowh = tile + (42 if large else 32)
        regions = []
        for j, panel in enumerate(m['demonstrations']):
            y = 24 + (j // 2) * rowh
            left = 10 + (j % 2) * (width // 2)
            for k, grid in enumerate(panel):
                x = left + k * (tile + gap)
                regions.append(dict(name=f'demo-{j}-{k}', bounds=(x, y, x + tile, y + tile),
                                    cell=cell, values=grid))
        top = 24 + stages * rowh + 36
        stride = width // 4
        for k, grid in enumerate(m['inputs']):
            x = k * stride + (stride - tile) // 2
            regions.append(dict(name=f'input-{k}', bounds=(x, top, x + tile, top + tile),
                                cell=cell, values=grid))
        for k, grid in enumerate(m.get('options') or []):
            x = k * stride + (stride - tile) // 2
            y = top + tile + 58
            regions.append(dict(name=f'option-{k}', bounds=(x, y, x + tile, y + tile),
                                cell=cell, values=grid))
    except Exception:  # noqa: BLE001 - geometry that cannot be computed counts as a mismatch
        return None
    # Every cell's centre pixel must be one of the two colours the parent
    # drew, and must agree with the truth value.
    for region in regions:
        x0, y0, x1, y1 = region['bounds']
        if not (0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height):
            return None
        for r, row in enumerate(region['values']):
            for c, value in enumerate(row):
                got = im.getpixel((x0 + c * cell + cell // 2, y0 + r * cell + cell // 2))
                if got != (ON if value else OFF):
                    return None
    return regions


def restyle_symbolic_logic(rec, im):
    """Apply this record's style and palette to a symbolic.logic item.  If the
    geometry check fails the image is returned unchanged."""
    style = style_for(rec)
    name, on, off = palette_for(rec)
    if style == 'square' and name == 'ink':
        return im
    regions = symbolic_logic_regions(rec, im)
    if not regions:
        return im
    rec['meta']['render_style'] = style
    rec['meta']['render_palette'] = [list(on), list(off)]
    rec['meta']['render_palette_name'] = name
    rec['render_regions'] = regions
    return _repaint(im.copy(), regions, style, on, off)
