"""Output-preserving speed-ups for find_the_shadow's generator (installed by the adapter, all tiers).

Nothing here changes a pixel, a random draw or a decision: every replacement computes exactly what the
function it replaces computes, it only stops recomputing work (1.7-2.0x less CPU per item on every tier).

  * norm_mask / bbox: the bounding box from row/column projections (m.any) instead of
    np.where over every ink pixel -- same integers.
  * _apply: the edit, its fillet blur and the window are computed on a crop around the window, padded
    by the Gaussian kernel radius + 3 px (clipped to the image, where cv2's BORDER_REFLECT_101 then sees the
    same image edge).  Outside the window the result is the input mask, exactly as in the original (the
    blob lies inside the window and a fill is clipped to the icon's own bbox, which the mask already lies in).
  * make_decoy: the per-base-mask work (half-resolution mask, its normalised masks, bboxes, hole
    count, contour sites) is computed once per mask content and reused across the c clones of a family,
    their retries and later attempts (keyed by a blake2b digest of the mask bytes); the bisection's
    reference mask (recomputed on every call in the original) comes from that cache.  The random stream
    (rng.random per proposal, rng.uniform x2 per tried proposal) is consumed in the original order.
  * puzzle.base_tile: rendered tiles are cached per (icon, orientation, size, tint) in a bounded LRU
    (PIL images; callers never mutate them).
install() replaces decoys.norm_mask, decoys.make_decoy (built on the fast norm_mask / bbox / _apply) and
puzzle.base_tile.
"""
import collections, hashlib
import numpy as np
import cv2

from . import decoys, puzzle

_LRU_TILES = 512
_LRU_BASE = 256


class _LRU(collections.OrderedDict):
    def __init__(self, cap):
        super().__init__(); self.cap = cap

    def get_or(self, key, fn):
        v = super().get(key)
        if v is not None:
            self.move_to_end(key)
            return v
        v = fn()
        self[key] = v
        if len(self) > self.cap:
            self.popitem(last=False)
        return v


def _bbox_any(m):
    r = np.flatnonzero(m.any(1))
    if len(r) == 0:
        return None
    c = np.flatnonzero(m.any(0))
    return int(r[0]), int(r[-1]), int(c[0]), int(c[-1])


def norm_mask_fast(m, tile, thr=96):
    bb = _bbox_any(m)
    if bb is None:
        return np.zeros((tile, tile), bool)
    y0, y1, x0, x1 = bb[0], bb[1] + 1, bb[2], bb[3] + 1
    crop = (m[y0:y1, x0:x1].astype(np.uint8)) * 255
    h, w = crop.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return cv2.resize(sq, (tile, tile), interpolation=cv2.INTER_AREA) > thr


def bbox_fast(m):
    return _bbox_any(m)


def _ksize_radius(sig):
    # cv2 createGaussianKernels for a CV_32F source: ksize = cvRound(sigma * 4 * 2 + 1) | 1
    return (int(np.floor(sig * 8 + 1 + 0.5)) | 1) // 2


def apply_fast(m, kind, c, r, asp, ang, bb):
    H, W = m.shape
    cx, cy = int(round(c[0])), int(round(c[1]))
    sig = max(1.8, 0.13 * r)
    A = max(1, int(round(r * 2.4)))
    Ab = max(max(1, int(round(r * asp))), max(1, int(round(r))))
    A = max(A, Ab)
    pad = _ksize_radius(sig) + 3
    # window crop (output region) and blur crop (input region)
    oy0, oy1 = max(0, cy - A - 2), min(H, cy + A + 3)
    ox0, ox1 = max(0, cx - A - 2), min(W, cx + A + 3)
    if oy0 >= oy1 or ox0 >= ox1:
        return m.copy()
    iy0, iy1 = max(0, oy0 - pad), min(H, oy1 + pad)
    ix0, ix1 = max(0, ox0 - pad), min(W, ox1 + pad)
    sub = m[iy0:iy1, ix0:ix1]
    z = np.zeros(sub.shape, np.uint8)
    cv2.ellipse(z, (cx - ix0, cy - iy0), (max(1, int(round(r * asp))), max(1, int(round(r)))), ang, 0, 360, 1, -1)
    b = z > 0
    m2 = (sub & ~b) if kind == "bite" else (sub | b)
    if kind == "fill":
        y0, y1, x0, x1 = bb
        keep = np.zeros_like(m2)
        ky0, ky1 = max(0, y0 - iy0), max(0, min(iy1, y1 + 1) - iy0)
        kx0, kx1 = max(0, x0 - ix0), max(0, min(ix1, x1 + 1) - ix0)
        keep[ky0:ky1, kx0:kx1] = True
        m2 &= keep
    sm = cv2.GaussianBlur(m2.astype(np.float32), (0, 0), sig) > 0.5
    zw = np.zeros(sub.shape, np.uint8)
    cv2.ellipse(zw, (cx - ix0, cy - iy0), (max(1, int(round(r * 2.4 * 1.0))), max(1, int(round(r * 2.4)))), 0, 0, 360,
                1, -1)
    win = zw > 0
    m2 = np.where(win, sm, m2)
    if kind == "fill":
        m2 &= keep
    out = m.copy()
    # only the output crop can differ from m; inside the blur margin the original keeps m as well
    # (np.where(win, ...) takes m2 == the edited-but-unblurred mask outside win, and the edit lies in win)
    out[oy0:oy1, ox0:ox1] = m2[oy0 - iy0:oy1 - iy0, ox0 - ix0:ox1 - ix0]
    return out


def install():
    if decoys.norm_mask is norm_mask_fast:
        return
    decoys.norm_mask = norm_mask_fast
    base_cache = _LRU(_LRU_BASE)
    tiles = _LRU(_LRU_TILES)
    render_base_tile = puzzle.base_tile

    def base_tile(lib, ents, ref, S, tint):
        key = (int(ref[0]), int(ref[1]), int(S), tuple(int(x) for x in tint))
        return tiles.get_or(key, lambda: render_base_tile(lib, ents, ref, S, tint))
    puzzle.base_tile = base_tile

    norm_mask, iou, disc_radius, outline_fill, n_components, bbox = (
        norm_mask_fast, decoys.iou, decoys.disc_radius, decoys.outline_fill, decoys.n_components, bbox_fast)
    contour_sites, _apply = decoys.contour_sites, apply_fast

    def _base_info(m, tile):
        key = (hashlib.blake2b(np.packbits(m).tobytes(), digest_size=16).digest(), m.shape, tile)

        def mk():
            SF = m.shape[0] // (tile * 2)
            ms = (cv2.resize(m.astype(np.uint8), (tile * 2, tile * 2),
                             interpolation=cv2.INTER_AREA) > 0) if SF > 1 else m
            return dict(SF=SF, ms=ms, base_s=norm_mask(ms, tile), bb_s=bbox(ms), bb_f=bbox(m),
                        holes_f=n_components(~m), ms_sum=float(ms.sum()), base_f=norm_mask(m, tile), sites={})
        return base_cache.get_or(key, mk)

    def make_decoy(m, target_iou, tile, rmin, rmax, rng, polarity="extra",
                   tol=0.005, tries=60, why=None, avoid=(), avoid_r=0.0):
        def _w(k):
            if why is not None:
                why[k] = why.get(k, 0) + 1

        info = _base_info(m, tile)
        SF, ms, base_s, bb_s, bb_f, holes_f = (info["SF"], info["ms"], info["base_s"], info["bb_s"],
                                               info["bb_f"], info["holes_f"])
        r0 = float(np.sqrt(2.0 * info["ms_sum"] * (1.0 - target_iou) / np.pi))
        pr = max(4.0, 1.3 * r0)
        if pr not in info["sites"]:
            info["sites"][pr] = contour_sites(ms, pr)
        sites = info["sites"][pr]
        if not sites:
            return None
        base_f = info["base_f"]
        SSs = ms.shape[0] / float(tile)

        def bisect(mask, nb0, bb, p, nv, kind, ar, asp, ang, lo_r, hi_r, iters, scale):
            sgn = 1.0 if kind == "bite" else -1.0
            best = None
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

        props = []
        for si, (p, nv, depth, sf) in enumerate(sites):
            if avoid and min(float(np.hypot(p[0] * SF - ax, p[1] * SF - ay))
                             for ax, ay in avoid) < avoid_r:
                continue
            for kind in ("bite", "fill"):
                if kind == "bite" and depth < max(6.0, 1.15 * r0):
                    continue
                smoother = (sf > 0.52) if kind == "fill" else (sf < 0.48)
                pol = "smoother" if smoother else "extra"
                if pol != polarity:
                    continue
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
            ladder = ((0.0, 0.25, 0.5, 0.75) if polarity == "smoother"
                      else (0.0, 0.15, 0.30))
            for ar in ladder:
                _w("props")
                got, hit = bisect(ms, base_s, bb_s, p, nv, kind, ar, asp, ang,
                                  2.5 * SSs, min(0.45 * ms.shape[0], 26.0 * SSs),
                                  14, SSs)
                if got is None or not hit:
                    _w("no_bisect"); break
                r, m2, v = got
                d = disc_radius(base_s, norm_mask(m2, tile))
                if d < rmin + 0.5:
                    _w("disc_small"); continue
                if d > rmax:
                    _w("disc_big"); break
                if SF > 1:
                    got2, hit2 = bisect(m, base_f, bb_f, p * SF, nv, kind, ar, asp, ang,
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

    decoys.make_decoy = make_decoy
