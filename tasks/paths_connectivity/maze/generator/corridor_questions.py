"""maze corridor-distance question templates and their renderer: difficulty on
CORRIDOR DISTANCE, not on component structure.

Questions ABOUT COMPONENTS ("which start reaches the target", "how many markers
are connected to it") need a fragmented board, and component size is visible:
on deep boards the answer's component is the biggest one on the page, and "the
largest group" predicts the answer.  A question that lives in a SINGLE
component and asks about corridor distance has no such cue.  So two question
forms are distance questions on one connected board:

    kth     "which start is the SECOND closest"   (needs all four distances)
    within  "how many are within R corridor steps" (answer is a ball count)

Neither can be answered by reading component sizes, because there is only one
component, and neither by finding the single nearest thing: the second-closest
is not a salient percept, and a ball count needs every marker checked.  The
boards and cases are built by detour_core.py; this module supplies the question
sentences and render, which puts them on a board_law.render record.
"""

from __future__ import annotations

from . import board_law, lattice, questions


# K IS HELD NEARLY FIXED, and that is a constraint rather than an oversight.
# K is the number of extra passages, and it is the strongest lever on corridor
# length there is -- measured eccentricity p10/p50/p90 on a 17x17 board:
#
#     K=0   36/81/151      K=30  26/34/52      K=100  20/26/32
#
# so it is tempting to give each tier its own K. It must not: K also sets how
# braided the maze looks, and braidedness is VISIBLE: tying K to the tier would
# print the tier on the page. So K stays
# in a narrow band for every tier and the tier is carried by D alone, which is
# invisible: it is a property of where the markers were put, not of the maze.


def euclid_rank(lat, cells, target):
    order = sorted(range(len(cells)),
                   key=lambda i: lattice.distance_px(lat, cells[i], target))
    return tuple(order)


QUESTION_KTH = ("Counting by corridor distance (the number of squares you walk "
                "through), which start is the {ord} closest to {desc}? "
                "Answer A, B, C or D.")
ORD = {2: "second", 3: "third"}
QUESTION_WITHIN = ("Walking only along corridors, how many of the marked starts "
                   "are within {R} squares of {desc}? (Answer with a single "
                   "number.)")


def render(lat, case, task, rng, palettes, idx, image_rel, tier, K):
    """board_law.render, with the question replaced for the two rebuilt tasks.

    The record SHAPE is board_law.render's -- same fields, same answer alphabets;
    only the sentence differs, because board_law.render's asks about components
    and these boards have one.
    """
    base, rec = board_law.render(lat, case, task, rng, palettes, idx, image_rel, tier, K)
    # Ask the base helper for the descriptor, using the same two inputs it
    # was given -- both are already on the record (slicing board_law.render's
    # sentence would break on any later " to ").
    st = rec.get("style") or {}
    shape, pstyle = st.get("target_shape"), st.get("target_prompt_style")
    tail = (questions.target_descriptor(shape, pstyle) if shape
            else "the unlabeled target marker")
    if task == "mc":
        rec["question"] = QUESTION_KTH.format(ord=ORD[case["kth"]], desc=tail)
        rec["qa_template"] = 100 + case["kth"]
    elif task == "count":
        rec["question"] = QUESTION_WITHIN.format(R=case["R"], desc=tail)
        rec["qa_template"] = 200
    rec["axis"]["D"] = case.get("D")
    rec["axis"]["margin"] = case.get("margin")
    rec["axis"]["kth"] = case.get("kth")
    rec["axis"]["R"] = case.get("R")
    return base, rec
