"""Five public tiers; xxhard mixes two internal scene profiles equally."""

TIERS = ("easy", "medium", "hard", "xhard", "xxhard")

# The 'ood' key names the extended xxhard construction internally; it is used in
# both asset splits and is part of the overlap-answer seed namespace.

COUNT_RANGES = {
    "easy": (8, 12),
    "medium": (13, 20),
    "hard": (21, 28),
    "xhard": (29, 38),
    "xxhard": (39, 48),
    "ood": (52, 64),
}

TARGET_RANGES = {
    "easy": (2, 4),
    "medium": (4, 7),
    "hard": (6, 12),
    "xhard": (10, 18),
    "xxhard": (14, 24),
    "ood": (20, 32),
}

OVERLAP_TARGET_RANGES = {
    "easy": (2, 4),
    "medium": (4, 6),
    "hard": (6, 9),
    "xhard": (10, 14),
    "xxhard": (15, 20),
    "ood": (22, 28),
}
