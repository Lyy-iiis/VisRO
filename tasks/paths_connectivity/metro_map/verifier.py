"""metro_map verifier: re-run the map validator on the serialized lines and stations."""
from visro import runners

TASK = 'metro_map'
CORRECTNESS = 'Recompute stops, transfers, physical station counts or nearest ring from the line cells, including crop checks.'
UNIQUENESS = 'The validator checks the selected minimum or unique nearest ring.'


def _check(record, meta, original, image):
    from tasks.paths_connectivity.metro_map.generator import metro_core
    metro_core.validate_record(original, image)


def verify(record, image):
    return runners.run_checks(TASK, record, image, _check)
