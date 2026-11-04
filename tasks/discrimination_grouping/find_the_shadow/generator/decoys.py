"""Synthesis of near-clone decoy shadows.

An unmatched decoy shadow needs no coloured counterpart, so it need not be a real
icon: it can be MANUFACTURED from a real silhouette by one LOCAL edit.  That
turns the rival-similarity axis from a supply problem ("how similar a pair of
real icons can you find?", which topped out at IoU 0.94) into a design problem
("how similar do you want it?", unbounded up to the refutability floor).

The edit is always ONE feature on the OUTER CONTOUR -- a rounded notch bitten out
of the boundary, or a rounded lobe filled into a concavity -- never an interior
knockout, because an interior hole in a silhouette is spotted instantly (the
outer-contour gate `out_max` forbids exactly that).  The
feature's size is tuned by bisection so that

    IoU(original, decoy)            == the tier's target, and
    inscribed disc inside the XOR   >= rmin px AT THE DELIVERY TILE SIZE,

i.e. the whole difference is concentrated into ONE blob a reader can put a finger
on.  Concentrated = "you have to hunt for it" (a good hard problem); smeared out
along the whole contour = "you can only guess" (a bad one).
"""
import numpy as np
import cv2

# ------------------------------------------------------------------ helpers
def norm_mask(m, tile):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return np.zeros((tile, tile), bool)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = (m[y0:y1, x0:x1].astype(np.uint8)) * 255
    h, w = crop.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return cv2.resize(sq, (tile, tile), interpolation=cv2.INTER_AREA) > 96


def iou(a, b):
    return float(np.logical_and(a, b).sum()) / max(int(np.logical_or(a, b).sum()), 1)


def disc_radius(a, b):
    x = np.logical_xor(a, b).astype(np.uint8)
    if x.sum() == 0:
        return 0.0
    return float(cv2.distanceTransform(x, cv2.DIST_L2, 5).max())


def outline_fill(m):
    """The silhouette with every interior hole filled -- the 'outer body'."""
    u = m.astype(np.uint8)
    cnts, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(u)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out > 0


def n_components(m):
    return int(cv2.connectedComponents(m.astype(np.uint8), 8)[0]) - 1


def bbox(m):
    ys, xs = np.where(m)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


# ------------------------------------------------------------------ the edit
def contour_sites(m, probe_r, step=6):
    """Boundary points with a smoothed inward normal, the local half-thickness,
    and the LOCAL SOLID FRACTION -- the share of a disc of radius `probe_r`
    around the point that is already inside the silhouette.

    That last number is what decides how the edit READS.  sf > 0.5 means the
    point sits in a valley (a concavity); sf < 0.5 means it sits on a tip or a
    lobe.  Filling a valley or biting off a tip makes the decoy SMOOTHER than
    the real shadow -- the real one is then the one carrying the extra feature.
    Filling a flat stretch or biting into one makes the decoy carry the extra
    feature instead.  Mixing the two inside one image is what stops
    "whichever shadow has the odd little bump is the fake" from being a
    complete solution."""
    u = m.astype(np.uint8)
    cnts, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return []
    c = max(cnts, key=len).reshape(-1, 2)          # (x, y)
    n = len(c)
    if n < 40:
        return []
    K = max(6, n // 60)
    k = max(3, int(round(2 * probe_r)) | 1)
    solid = cv2.blur(u.astype(np.float32), (k, k))
    sites = []
    for i in range(0, n, step):
        p = c[i]
        t = c[(i + K) % n] - c[(i - K) % n]
        nl = np.hypot(*t)
        if nl < 1e-6:
            continue
        nv = np.array([-t[1], t[0]], np.float64) / nl
        q = p + nv * 4.0
        qi = (int(round(q[1])), int(round(q[0])))
        if not (0 <= qi[0] < m.shape[0] and 0 <= qi[1] < m.shape[1]):
            continue
        if not m[qi]:
            nv = -nv
            q = p + nv * 4.0
            qi = (int(round(q[1])), int(round(q[0])))
            if not (0 <= qi[0] < m.shape[0] and 0 <= qi[1] < m.shape[1]) or not m[qi]:
                continue
        d = 0.0
        for sstep in np.arange(2.0, 90.0, 3.0):
            r = p + nv * sstep
            ri = (int(round(r[1])), int(round(r[0])))
            if not (0 <= ri[0] < m.shape[0] and 0 <= ri[1] < m.shape[1]) or not m[ri]:
                break
            d = sstep
        sites.append((p.astype(np.float64), nv, float(d),
                      float(solid[int(p[1]), int(p[0])])))
    return sites


def _blob(shape, c, r, asp, ang):
    z = np.zeros(shape, np.uint8)
    ax = (max(1, int(round(r * asp))), max(1, int(round(r))))
    cv2.ellipse(z, (int(round(c[0])), int(round(c[1]))), ax, ang, 0, 360, 1, -1)
    return z > 0


def _apply(m, kind, c, r, asp, ang, bb):
    """One edit + a LOCAL fillet, so the junction has no right-angle burr."""
    b = _blob(m.shape, c, r, asp, ang)
    m2 = (m & ~b) if kind == "bite" else (m | b)
    if kind == "fill":                                  # never grow the bbox
        y0, y1, x0, x1 = bb
        keep = np.zeros_like(m2)
        keep[y0:y1 + 1, x0:x1 + 1] = True
        m2 &= keep
    sig = max(1.8, 0.13 * r)
    sm = cv2.GaussianBlur(m2.astype(np.float32), (0, 0), sig) > 0.5
    win = _blob(m.shape, c, r * 2.4, 1.0, 0)
    m2 = np.where(win, sm, m2)
    if kind == "fill":
        y0, y1, x0, x1 = bb
        keep = np.zeros_like(m2)
        keep[y0:y1 + 1, x0:x1 + 1] = True
        m2 &= keep
    return m2


def make_decoy(m, target_iou, tile, rmin, rmax, rng, polarity="extra",
               tol=0.005, tries=60, why=None, avoid=(), avoid_r=0.0):
    """Return (decoy_mask, info) or None.

    `polarity` is what the decoy should LOOK like next to the real shadow:
      "smoother" -- the decoy is the plainer of the two (an existing lobe bitten
                    off, or an existing notch filled in), so the REAL shadow is
                    the one with the extra feature;
      "extra"    -- the decoy carries the extra feature.
    Both are the same operation; only the site's local convexity differs.

    IoU and the inscribed disc are measured on tile-resolution canonical masks --
    exactly the quantities the shipped PNG is audited on.  The proposal search
    runs at half resolution and the winner is rebuilt and re-bisected at full
    resolution, so every number returned is a full-resolution number."""
    def _w(k):
        if why is not None:
            why[k] = why.get(k, 0) + 1

    SF = m.shape[0] // (tile * 2)
    ms = (cv2.resize(m.astype(np.uint8), (tile * 2, tile * 2),
                     interpolation=cv2.INTER_AREA) > 0) if SF > 1 else m
    base_s = norm_mask(ms, tile)
    bb_s, bb_f = bbox(ms), bbox(m)
    holes_f = n_components(~m)
    # nominal blob radius for this target, used to probe local convexity at the
    # scale the edit will actually have
    r0 = float(np.sqrt(2.0 * ms.sum() * (1.0 - target_iou) / np.pi))
    sites = contour_sites(ms, max(4.0, 1.3 * r0))
    if not sites:
        return None
    base_f = norm_mask(m, tile)
    SSs = ms.shape[0] / float(tile)

    def bisect(mask, bb, p, nv, kind, ar, asp, ang, lo_r, hi_r, iters, scale):
        sgn = 1.0 if kind == "bite" else -1.0
        best = None
        nb0 = norm_mask(mask, tile)
        for _ in range(iters):
            r = 0.5 * (lo_r + hi_r)
            c = p + nv * (sgn * ar * r)
            m2 = _apply(mask, kind, (c[0], c[1]), r, asp, ang, bb)
            v = iou(nb0, norm_mask(m2, tile))
            best = (r, m2, v)
            if abs(v - target_iou) <= tol:
                return best, True
            if v > target_iou:
                lo_r = r
            else:
                hi_r = r
            if hi_r - lo_r < 0.02 * scale:
                break
        return best, False

    # order the proposals so the requested polarity is tried first
    props = []
    for si, (p, nv, depth, sf) in enumerate(sites):
        if avoid and min(float(np.hypot(p[0] * SF - ax, p[1] * SF - ay))
                         for ax, ay in avoid) < avoid_r:
            continue                    # keep a second clone's edit far from the first
        for kind in ("bite", "fill"):
            if kind == "bite" and depth < max(6.0, 1.15 * r0):
                continue
            smoother = (sf > 0.52) if kind == "fill" else (sf < 0.48)
            pol = "smoother" if smoother else "extra"
            if pol != polarity:
                continue
            # the more pronounced the local convexity, the more natural and the
            # more area-efficient the edit is
            score = abs(sf - 0.5)
            props.append((score + rng.random() * 0.06, si, kind, sf))
    if not props:
        _w("no_props"); return None
    props.sort(reverse=True)
    props = props[:tries]

    for _sc, si, kind, sf in props:
        p, nv, depth, _ = sites[si]
        asp = rng.uniform(1.0, 1.22)
        ang = rng.uniform(0, 180)
        # An edit that ERASES an existing feature is bounded by the shape that
        # already surrounds it, so pushing the blob's centre further out only
        # makes it chunkier -- no undercut is created.  An edit that ADDS a
        # feature to a flat stretch would turn into a keyhole if pushed, so its
        # ladder stops early.
        ladder = ((0.0, 0.25, 0.5, 0.75) if polarity == "smoother"
                  else (0.0, 0.15, 0.30))
        for ar in ladder:
            _w("props")
            got, hit = bisect(ms, bb_s, p, nv, kind, ar, asp, ang,
                              2.5 * SSs, min(0.45 * ms.shape[0], 26.0 * SSs),
                              14, SSs)
            if got is None or not hit:
                _w("no_bisect"); break
            r, m2, v = got
            d = disc_radius(base_s, norm_mask(m2, tile))
            if d < rmin + 0.5:
                _w("disc_small"); continue              # try a leaner shape
            if d > rmax:
                _w("disc_big"); break
            if SF > 1:
                got2, hit2 = bisect(m, bb_f, p * SF, nv, kind, ar, asp, ang,
                                    r * SF * 0.72, r * SF * 1.35, 11, SSs * SF)
                if got2 is None or not hit2:
                    _w("no_bisect_full"); break
                rf, mf, vf = got2
            else:
                rf, mf, vf = r, m2, v
            if n_components(mf) != 1 or n_components(~mf) != holes_f:
                _w("topology"); break
            if bbox(mf) != bb_f:
                _w("bbox"); break
            nb = norm_mask(mf, tile)
            d = disc_radius(base_f, nb)
            if not (rmin <= d <= rmax):
                _w("disc_full"); break
            xo = np.logical_xor(outline_fill(base_f), outline_fill(nb)).sum()
            xa = np.logical_xor(base_f, nb).sum()
            if xa == 0 or xo / xa < 0.60:
                _w("outline"); break
            return mf, dict(edit_kind=kind, polarity=polarity,
                            local_solid_frac=round(sf, 3),
                            iou=round(vf, 4), disc_px=round(d, 2),
                            r_px=round(rf / (m.shape[0] / tile), 2),
                            undercut=ar, aspect=round(asp, 2),
                            xor_px=int(xa), outline_frac=round(float(xo) / xa, 3),
                            site=[int(p[0]) * SF, int(p[1]) * SF])
    return None
