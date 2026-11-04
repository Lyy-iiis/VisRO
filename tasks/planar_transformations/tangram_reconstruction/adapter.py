"""tangram_reconstruction: cover a gray triangle-grid silhouette with numbered pieces.

Generator: generator/tangram.py, all five tiers in one module (easy/medium:
the constructive build; hard/xhard/xxhard: the sampling loop; the polytan
helpers are generator/polytan.py).

  * the question says k is the smallest quarter-turn count giving the placed
    piece, and build()
    rejects every candidate whose exact-cover count under the stated turns is
    not 1 (congruent, i.e. interchangeable, pieces always give a second answer
    and are rejected with their own reason);
  * easy: the rule forbids rotation outright, uniqueness is enforced under k=0,
    and additionally under k=0..3 so no rotated alternative exists either;
  * candidates with a piece made only of whole diamonds are rejected: for those
    shapes turns k and k+2 can give the same physical shape with a different
    triangle set, and they can sit half a diagonal off the lattice.  For every
    other piece "same placed shape" and "same triangles" coincide and no
    off-lattice fit exists;
  * the answer is one [x,y,k] per piece, with no worked example; the question
    says to count grid squares, not pixels;
  * fonts come from the repository's assets/fonts (sha-pinned);
  * meta carries the render layout so the verifier can decode the image.
"""
from __future__ import annotations

import random

from visro import api
from visro.seeding import canonical, digest

from tasks.planar_transformations.tangram_reconstruction.generator import tangram

TASK = "tangram_reconstruction"
VERSION = f"{TASK}@{api.RELEASE_VERSION}"

_COMMON = ("triangle-grid silhouette (each unit square = 4 triangles) split by random multi-source BFS into "
           "connected parts of >=4 triangles; no free piece made only of whole diamonds; answer = JSON [x,y,k] "
           "per piece; exactly one valid answer under the stated turns (congruent pieces rejected), k minimal")
TIER_DEFINITIONS = {
    "easy": "target 18 triangles, bbox <=6x6 squares, 3 parts = 1 blue fixed hint + 2 free pieces, pieces drawn as "
            "used and rotation forbidden (k=0; also unique if rotations were allowed), target cell min(60,200//span) px, "
            "pieces in one row at a shared cell >=12 px; " + _COMMON,
    "medium": "target 24 triangles, bbox <=8x8, 4 parts = 1 blue fixed hint + 3 free pieces, each drawn turned by a "
              "random k in 0..3 and rotatable by 0/90/180/270 deg clockwise (no reflection), target cell "
              "min(60,200//span) px, pieces in one row at a shared cell >=12 px; " + _COMMON,
    "hard": "target 28 triangles, bbox <=11x11, 4 parts = 1 blue fixed hint + 3 free pieces, random display turn, "
            "rotations 0..3 allowed, target cell min(32,184//span) px, pieces on a 4-column grid at "
            "min(22,88//w,66//h) px per square (candidate rejected below 10, so >=11); " + _COMMON,
    "xhard": "target 44 triangles, bbox <=11x11, 6 free pieces and no fixed hint, random display turn, rotations "
             "0..3 allowed, target cell min(32,184//span) px, pieces on a 4-column grid (2 rows) at "
             "min(22,88//w,66//h) px per square (candidate rejected below 10, so >=11); " + _COMMON,
    "xxhard": "target 64 triangles, bbox <=11x11, 8 free pieces and no fixed hint, random display turn, rotations "
              "0..3 allowed, target cell min(32,184//span) px, pieces on a 4-column grid (2 rows) at "
              "min(22,88//w,66//h) px per square (candidate rejected below 10, so >=11); " + _COMMON,
}

# --------------------------------------------------------------------------- #
# Content key: the puzzle up to the symmetries the task is invariant to
# --------------------------------------------------------------------------- #
_REFLECT_T = (0, 3, 2, 1)  # mirror x -> -x swaps the E and W triangles


def _transform(cells, g):
    """g in 0..7: optional mirror (g >= 4) followed by g % 4 clockwise quarter turns."""
    out = [tuple(c) for c in cells]
    if g >= 4:
        out = [(-x - 1, y, _REFLECT_T[t]) for x, y, t in out]
    for _ in range(g % 4):
        out = [(-y - 1, x, (t + 1) % 4) for x, y, t in out]
    return out


def _norm(cells, ref=None):
    ref = ref if ref is not None else cells
    mx = min(c[0] for c in ref)
    my = min(c[1] for c in ref)
    return sorted((x - mx, y - my, t) for x, y, t in cells)


def core_key(target, fixed, pieces, turns) -> str:
    """Min over the 8 board symmetries of (target, fixed, sorted pieces).

    Rotating or mirroring the whole scene (board and every piece together) gives the same
    puzzle; piece numbering is irrelevant; where rotations are allowed a piece's display
    turn is irrelevant too (each piece is reduced to its rotation-canonical form).
    """
    best = None
    for g in range(8):
        t = _transform(target, g)
        body = dict(target=_norm(t), fixed=_norm(_transform(fixed, g), t) if fixed else [])
        ps = []
        for p in pieces:
            q = _transform(p, g)
            if len(turns) > 1:
                q = min(_norm(_transform(q, r)) for r in range(4))
            else:
                q = _norm(q)
            ps.append(q)
        body["pieces"] = sorted(ps)
        s = canonical(body)
        if best is None or s < best:
            best = s
    return digest(dict(task=TASK, puzzle=best))


def _congruent(pieces, turns) -> bool:
    keys = [min(tuple(sorted(tangram.rotated(p, k))) for k in turns) for p in pieces]
    return len(set(keys)) < len(keys)


def build(tier: str, seed: int, *, split: str, index: int) -> api.Item:
    return make_item(tier, seed, gate=True)


def make_item(tier: str, seed: int, *, gate: bool = True) -> api.Item:
    """gate=False skips the uniqueness gate (for building known-ambiguous controls)."""
    cfg = tangram.CONFIG[tier]
    turns = list(cfg["turns"])
    rng = random.Random(int(seed))
    target, fixed, shown, poses, attempts = tangram.sample(tier, rng)

    # Uniqueness gate: exactly one answer string under the stated rule.
    if gate:
        if _congruent(shown, turns):
            raise api.Reject("congruent pieces: swapping them gives a second answer")
        n = tangram.count_assemblies(target, fixed, shown, turns, cap=2)
        if n != 1:
            raise api.Reject(f"{'second tiling' if n > 1 else 'no tiling'} under the stated turns")
        if tier == "easy" and tangram.count_assemblies(target, fixed, shown, tangram.ALL_TURNS, cap=2) != 1:
            raise api.Reject("easy: another assembly exists if pieces were rotated")
        if any(tangram.diamond_only(p) for p in shown):
            raise api.Reject("diamond-only piece: turns k and k+2 / half-diagonal placements alias")

    im, layout = tangram.render(tier, target, fixed, shown)
    answer = tangram.answer_string(poses)
    params = {k: cfg[k] for k in ("parts", "area", "min_size", "span", "hint", "turns")}
    meta = dict(
        family="constructive_tangram",
        tier_params=params,
        free_pieces=len(shown),
        target=sorted(map(list, target)),
        fixed=sorted(map(list, fixed)),
        pieces=[sorted(map(list, p)) for p in shown],
        planted_poses=poses,
        layout=layout,
        attempts=attempts,
        triangle_encoding="(x,y,t): unit square (x,y), y down; t 0=N 1=E 2=S 3=W quarter; "
                          "clockwise turn (x,y,t)->(-y-1,x,t+1)",
        generator="generator/tangram.py:sample",
    )
    return api.Item(
        question=tangram.question(tier),
        answer=answer,
        image=im,
        meta=meta,
        core_key=core_key(target, fixed, shown, turns),
        answer_type="text",
    )
