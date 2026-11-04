# -*- coding: utf-8 -*-
"""The QUESTION PLAN for the five-tier ladder - ONE wording family, five tiers.

It covers absolute, none and chained questions, with the number of candidate
slots as a parameter so the two low tiers can run a 6-slot bank while the three
top tiers keep their 12-slot bank.  The text is compact (see "wording" below):
the conventions are each stated once, so that every question of every tier is
at most 240 tokens, and no worked format example is printed.

ONE STEM, TWO BODIES
--------------------
Every tier prints the SAME stem (`_STEM` below) with only {S} and {K} filled
in, and the low tiers ask the SAME question the hard tier asks (the plain
body):

    Which figure is the base turned clockwise by each angle: ...?

easy, medium and hard all take that plain body, with n and the angle list
moving and nothing else.

xhard and xxhard cannot: a question whose honest answer is "none", and a chained
question, need the enumerated body, one angle per line, with the none rule and
(xxhard) the "figure k+X" lines.  So the ladder carries exactly two bodies, and
the switch between them is mechanical: a plan that is nothing but present
absolute questions gets the plain body, anything else gets the enumerated one.

gates
  G12 the questions of an item aim at n DIFFERENT target shifts
  G13 a none question is verified to match ZERO of the candidates by the same
      exhaustive match that verifies the others match exactly one, and its
      target is never 0 deg or 180 deg (which the stem excludes -> free none)
  G15 every INTERMEDIATE of a chain is one of the numbered figures
  G16 a chain never revisits a figure (partial sums distinct mod 16)
  G17 every chain root is a literal figure number in the stem; roots are distinct
  G18 at most slots-2 questions per item answer with a figure, so the answers
      are never a permutation of 1..slots and elimination gives nothing
"""
from __future__ import annotations

from .word import K


def max_present(slots):
    return slots - 2                     # G18


def legal_shifts():
    """S2: the stem excludes 0 deg and 180 deg, so 14 shifts are legal."""
    return [s for s in range(1, K) if s != K // 2]


def fmt_angle(s):
    return f"{360.0 * s / K:g}"


def path_shifts(spec, shifts):
    """The shift values a question walks through.  For ABS that is just the one
    target; for a chain it is [shift(root), after link 1, ..., after link d]."""
    if spec["kind"] == "abs":
        return [spec["step"] % K]
    s0 = shifts[spec["ref"] - 1]
    out, acc = [s0], 0
    for a in spec["steps"]:
        acc += a
        out.append((s0 + acc) % K)
    return out


def target_of(spec, shifts):
    return path_shifts(spec, shifts)[-1]


def solve_plan(w, shifts, plan):
    """G3: every question matched exhaustively against ALL candidates as cyclic
    words.  A present target must hit exactly once, an absent target exactly
    zero times.  G15/G16 are enforced on the same walk that produces the answer."""
    words = [tuple(w[(j - s) % K] for j in range(K)) for s in shifts]
    assert len(set(words)) == len(words), "two candidates collide"
    out = []
    for spec in plan:
        ps = path_shifts(spec, shifts)
        if spec["kind"] == "chain":
            assert len(set(ps)) == len(ps), "G16: chain revisits a figure"
            for t in ps[:-1]:
                assert t in shifts, "G15: chain walks through a missing figure"
        t = ps[-1]
        assert t not in (0, K // 2), "G13: target excluded by the stem"
        tw = tuple(w[(j - t) % K] for j in range(K))
        hit = [i + 1 for i, cw in enumerate(words) if cw == tw]
        assert len(hit) <= 1, f"target {t}: {len(hit)} hits"
        assert len(hit) == (0 if spec["absent"] else 1), \
            f"target {t}: absent={spec['absent']} but {len(hit)} hits"
        out.append(str(hit[0]) if hit else "none")
    return out, words


def make_plan(rng, shifts, n_abs, n_none, chains, slots):
    """n_abs present absolute questions, n_none absolute questions aimed at an
    angle the bank does not cover, and one entry per chained question.

    TWO BRANCHES with their own fixed rng call sequences:
      chains == []  (easy, medium, hard, xhard)
      chains != []  (xxhard)
    Changing either sequence would change which words and items the tiers draw."""
    cand = legal_shifts()
    absent = [s for s in cand if s not in shifts]
    assert len(absent) == len(cand) - len(shifts) == len(cand) - slots, absent
    assert n_none <= len(absent), f"only {len(absent)} missing angles exist"

    if not chains:                                   # ---- no chains
        assert n_abs <= max_present(slots), f"G18: {n_abs} present answers"
        plan = []
        for s in rng.sample(shifts, n_abs):                    # ABS, present
            plan.append(dict(kind="abs", step=s, absent=False))
        for s in rng.sample(absent, n_none):                   # ABS, none
            plan.append(dict(kind="abs", step=s, absent=True))
        rng.shuffle(plan)
        return plan

    # ---------------------------------------------------- with chains
    n_present = n_abs + sum(1 for _, a in chains if not a)
    n_tot_none = n_none + sum(1 for _, a in chains if a)
    assert n_present <= max_present(slots), f"G18: {n_present} present answers"
    assert n_tot_none <= len(absent), f"only {len(absent)} missing angles exist"

    plan, used = [], set()
    for s in rng.sample(shifts, n_abs):                        # ABS, present
        plan.append(dict(kind="abs", step=s, absent=False))
        used.add(s)
    free_absent = list(absent)
    rng.shuffle(free_absent)
    for _ in range(n_none):                                    # ABS, none
        s = free_absent.pop()
        plan.append(dict(kind="abs", step=s, absent=True))
        used.add(s)

    roots = set()
    for d, want_absent in chains:
        for tries in range(1, 60001):
            assert tries < 60000, "chain search exhausted"
            ref = rng.randrange(1, slots + 1)
            if ref in roots:
                continue
            steps = [rng.randrange(1, K) for _ in range(d)]
            s0 = shifts[ref - 1]
            acc, ps = 0, [s0]
            for a in steps:
                acc += a
                ps.append((s0 + acc) % K)
            if len(set(ps)) != len(ps):                        # G16 acyclic
                continue
            if any(t not in shifts for t in ps[1:-1]):         # G15 intermediates
                continue
            t = ps[-1]
            if t in used:                                      # G12
                continue
            if want_absent:
                if t not in free_absent:
                    continue
                free_absent.remove(t)
            elif t not in shifts:
                continue
            used.add(t)
            roots.add(ref)
            plan.append(dict(kind="chain", ref=ref, steps=steps,
                             absent=want_absent))
            break
    rng.shuffle(plan)
    return plan


# ------------------------------------------------------------------- wording
# rotation_patterns: the compact wording.  Every convention is stated once, and
# the worst case over every plan any tier can draw is at most 240 tokens under
# both the LLaVA (vicuna sentencepiece, + BOS) and PaliGemma tokenizers.  The
# enumerated body lists one angle per line under a single question.
#
#   stem   _STEM below, with S = the number of candidates and K = 16.
#   plain  Which figure is the base turned clockwise by each angle: A, B, C?
#          Answer the figure numbers in order, comma-separated.
#   enum   Which figure is the base[ (or the named figure)] turned clockwise
#          by each angle below? Write none if no figure is. Answer in order,
#          comma-separated.
#          then one line per question: "A" (absolute) or "figure k+A" (a
#          depth-1 chain: figure k rotated a further A).
_STEM = (
    "Figures 1-{S} are the bold-framed base (far left) turned clockwise by different "
    "multiples of 22.5 degrees; the bead lengths on the {K} spokes are the code. ")


def _entry(s):
    """One question as printed: the angle, or 'figure k+angle' for a chain.
    No tier draws a chain deeper than 1 (TIERS in ladder.py: xxhard's chains
    are all (1, False)); the compact wording only defines depth-1 chains, so a
    deeper one is refused rather than printed ambiguously."""
    if s["kind"] == "abs":
        return fmt_angle(s["step"])
    if len(s["steps"]) != 1:
        raise ValueError("the wording states depth-1 chains only")
    return f"figure {s['ref']}+{fmt_angle(s['steps'][0])}"


def question_text(L, plan, shifts, slots):
    plain = all(s["kind"] == "abs" and not s["absent"] for s in plan)
    items = [_entry(s) for s in plan]
    stem = _STEM.format(S=slots, K=K)
    if plain:
        body = (f"Which figure is the base turned clockwise by each angle: {', '.join(items)}? "
                "Answer the figure numbers in order, comma-separated.")
        return stem + body
    who = "the base (or the named figure)" if any(s["kind"] == "chain" for s in plan) else "the base"
    body = (f"Which figure is {who} turned clockwise by each angle below? Write none if no figure is. "
            "Answer in order, comma-separated.\n" + "\n".join(items))
    return stem + body
