"""Option-board renderer ("top4" layout): the target on top, options A-D in one
row below, on a W x H canvas whose top `target_frac` holds the target.
"""
from PIL import Image, ImageDraw

from shared.drawing.fonts import load_bundled_font

BG = (255, 255, 255)
TARGET_FILL = (96, 110, 128)
PIECE = [(244, 150, 90), (150, 185, 235), (170, 210, 160), (230, 180, 210),
         (240, 210, 130), (180, 170, 225), (140, 205, 205), (235, 165, 160)]


def bwh(c):
    return (max(x for x, _ in c) + 1, max(y for _, y in c) + 1)


def cells(d, cs, px, ox, oy, fill, ow=2):
    for (x, y) in sorted(cs):
        d.rectangle([ox + x * px, oy + y * px, ox + (x + 1) * px, oy + (y + 1) * px],
                    fill=fill, outline=(38, 38, 42), width=ow)


def layout_rows(dims, px, W, gap):
    """greedy row-wrap; -> rows of (index,w,h), packed width, packed height"""
    rows = [[]]
    rw = 0
    for i, (w, h) in enumerate(dims):
        pw = w * px
        if rows[-1] and rw + gap + pw > W:
            rows.append([])
            rw = 0
        rows[-1].append((i, w, h))
        rw += (gap if len(rows[-1]) > 1 else 0) + pw
    tot_h = sum(max(h for _, _, h in r) * px for r in rows) + gap * (len(rows) - 1)
    max_w = max(sum(w * px for _, w, _ in r) + gap * (len(r) - 1) for r in rows)
    return rows, max_w, tot_h


def _geom(W, H, target_frac):
    """-> (target_box, [4 panel boxes], separator y)   boxes are (x0,y0,x1,y1)"""
    zone = int(H * target_frac)
    tbox = (8, 6, W - 8, zone - 6)
    sw = (W - 8) // 4
    pans = []
    for k in range(4):
        x0 = 8 + k * sw
        pans.append((x0, zone + 6, x0 + sw - 8, H - 8))
    return tbox, pans, zone


def panel_px(display, W, H, target_frac, cap=64):
    """The per-cell pixel size the option panels will actually get."""
    _, pans, _ = _geom(W, H, target_frac)
    x0, y0, x1, y1 = pans[0]
    lab = max(14, int(min(W, H) * 0.05))
    gap = max(4, min(W, H) // 150)
    best = cap
    for parts in display:
        dims = [bwh(p) for p in parts]
        Wi, Hi = (x1 - x0) - 22, (y1 - y0) - 6 - lab
        px = 5
        for t in range(cap, 4, -1):
            _, mw, mh = layout_rows(dims, t, Wi, gap)
            if mw <= Wi and mh <= Hi:
                px = t
                break
        best = min(best, px)
    return best


def compose(target, display, W, H, target_frac):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    tbox, pans, zone = _geom(W, H, target_frac)

    tw, th = bwh(target)
    tx0, ty0, tx1, ty1 = tbox
    tp = max(8, min((tx1 - tx0) // tw, (ty1 - ty0) // th, 64))
    ox = tx0 + ((tx1 - tx0) - tw * tp) // 2
    oy = ty0 + ((ty1 - ty0) - th * tp) // 2
    cells(d, target, tp, ox, oy, TARGET_FILL, ow=max(2, tp // 16))
    d.line([(10, zone), (W - 10, zone)], fill=(220, 220, 228), width=2)

    lab = max(14, int(min(W, H) * 0.05))
    f = load_bundled_font(max(15, int(lab * 0.72)), "sans-bold")
    gap = max(4, min(W, H) // 150)
    px = panel_px(display, W, H, target_frac)

    slots = []
    for k, parts in enumerate(display):
        x0, y0, x1, y1 = pans[k]
        slots.append((x0, y0, x1, y1))
        d.rounded_rectangle((x0, y0, x1, y1), radius=10,
                            outline=(214, 214, 224), width=2)
        d.text((x0 + 10, y0 + 4), "ABCD"[k], fill=(28, 28, 32), font=f)
        Wi, Hi = (x1 - x0) - 22, (y1 - y0) - 6 - lab
        dims = [bwh(p) for p in parts]
        rows, mw, mh = layout_rows(dims, px, Wi, gap)
        py = y0 + lab + (Hi - mh) // 2
        ci = 0
        for row in rows:
            rw = sum(w * px for _, w, _ in row) + gap * (len(row) - 1)
            rh = max(h for _, _, h in row) * px
            pxx = x0 + 11 + (Wi - rw) // 2
            for (i, w, h) in row:
                cells(d, parts[i], px, pxx, py + (rh - h * px) // 2,
                      PIECE[ci % len(PIECE)], ow=max(1, px // 12))
                pxx += w * px + gap
                ci += 1
            py += rh + gap
    return img, slots, px, tp

